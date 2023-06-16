import argparse
import torch
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import time
import os


class MyModel(torch.nn.Module):
    def __init__(self) -> None:
        super(MyModel, self).__init__()
        self.l1 = torch.nn.Linear(2, 2, bias=False)
        self.l2 = torch.nn.Linear(2, 2, bias=False)
        self.l3 = torch.nn.Linear(2, 2, bias=False)

    def mid_block(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x)
        x3 = self.l3(x)
        h = 2
        x1 = x1.reshape(x1.shape[0], x1.shape[1], h, x1.shape[2]//h).permute(0, 2, 1, 3).reshape(x1.shape[0] * h, x1.shape[1], x1.shape[2]//h)
        x2 = x2.reshape(x2.shape[0], x2.shape[1], h, x2.shape[2]//h).permute(0, 2, 1, 3).reshape(x2.shape[0] * h, x2.shape[1], x2.shape[2]//h)
        x3 = x3.reshape(x3.shape[0], x3.shape[1], h, x3.shape[2]//h).permute(0, 2, 1, 3).reshape(x3.shape[0] * h, x3.shape[1], x3.shape[2]//h)
        x = x1 @ x2.transpose(2, 1) * (1/8)
        x = torch.nn.functional.softmax(x, dim=-1)
        out = x @ x3
        out = out.reshape(out.shape[0]//h, h, out.shape[1], out.shape[2]).transpose(1, 2).reshape(out.shape[0]//h, out.shape[1], out.shape[2]*h)
        return out

    def forward(self, x):
        b, c, h, w = x.shape
        # 'b c h w -> b (h w) c'
        x = x.reshape(b, c, -1)
        x = x.transpose(1, 2)

        x = self.mid_block(x)

        # 'b (h w) c -> b c h w'
        x = x.reshape(b, h, w, c)
        x = x.permute(0, 3, 1, 2)
        return x


class TRTModel:
    class HostDeviceMem(object):
        def __init__(self, host_mem, device_mem, nbytes=None):
            self.host = host_mem
            self.device = device_mem
            self.nbytes = nbytes

        def __str__(self):
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

        def __repr__(self):
            return self.__str__()

    input_names = ["input"]
    output_names = ["output"]
    trt_logger = trt.Logger(trt.Logger.ERROR)

    def __init__(self, engine_file_path, gpu_id=0):
        cuda.init()
        self.cuda_ctx = cuda.Device(gpu_id).make_context()
        self.engine = self._load_engine(engine_file_path)
        self.binding_names = self.input_names + self.output_names
        self.context = self.engine.create_execution_context()
    
    @classmethod
    def build_engine(cls,
                     onnx_file_path,
                     engine_file_path,
                     *,
                     use_fp16=True,
                     dynamic_shapes={},
                     dynamic_batch_size=1):
        """Build TensorRT Engine
        :use_fp16: set mixed flop computation if the platform has fp16.
        :dynamic_shapes: {binding_name: (min, opt, max)}, default {} represents not using dynamic.
        :dynamic_batch_size: set it to 1 if use fixed batch size, else using max batch size
        """
        builder = trt.Builder(cls.trt_logger)
        network = builder.create_network(
            1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        config.set_tactic_sources(trt.TacticSource.CUBLAS_LT)

        # Default workspace is 2G
        config.max_workspace_size = 2 << 30
        # Uncomment this, To Solve Tensorrt BUG with Dynamic shape, from V8.5.1
        # config.set_preview_feature(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805, True)

        if builder.platform_has_fast_fp16 and use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # parse ONNX
        parser = trt.OnnxParser(network, cls.trt_logger)
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read(), path=onnx_file_path):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        print("===> Completed parsing ONNX file")

        # default = 1 for fixed batch size
        builder.max_batch_size = 1

        if len(dynamic_shapes) > 0:
            builder.max_batch_size = dynamic_batch_size
            profile = builder.create_optimization_profile()
            for binding_name, dynamic_shape in dynamic_shapes.items():
                print(f"===> set dynamic shape {binding_name}: {dynamic_shape}")
                min_shape, opt_shape, max_shape = dynamic_shape
                profile.set_shape(
                    binding_name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)

        # Remove existing engine file
        if os.path.isfile(engine_file_path):
            try:
                os.remove(engine_file_path)
            except Exception:
                print(f"Cannot remove existing file: {engine_file_path}")

        print("===> Creating Tensorrt Engine...")
        engine = builder.build_engine(network, config)
        if engine:
            print("===> Serializing engine...")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            print("===> Serialized Engine Saved at: ", engine_file_path)
        else:
            print("===> build engine error")
        return engine

    def _load_engine(self, engine_file_path):
        # Force init TensorRT plugins
        trt.init_libnvinfer_plugins(None, '')
        with open(engine_file_path, "rb") as f, \
                trt.Runtime(self.trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def _allocate_buffer(self, dynamic_factors):
        """Allocate buffer
        :dynamic_factor: normally expand the buffer size for dynamic shape
        """
        inputs = []
        outputs = []
        bindings = [None] * len(self.binding_names)
        stream = cuda.Stream()

        for binding in self.binding_names:
            binding_idx = self.engine[binding]
            if binding_idx == -1:
                print("Error Binding Names!")
                continue

            # trt.volume() return negtive volue if -1 in shape
            dynamic_factor = dynamic_factors.get(binding, 1)
            size = abs(trt.volume(self.engine.get_binding_shape(binding))) * dynamic_factor
            # print(f"===> bind_name:{binding}, max_batsz:{self.engine.max_batch_size}, shape:{self.engine.get_binding_shape(binding)} dynamic_factor:{dynamic_factor}")

            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings[binding_idx] = int(device_mem)
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                inputs.append(self.HostDeviceMem(host_mem, device_mem, host_mem.nbytes))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem, host_mem.nbytes))
        return inputs, outputs, bindings, stream

    def __call__(self, x):
        x = x.astype(np.float32, copy=False)
        binding_shape_map = {"input": x.shape}
        output = self.do_inference([x], binding_shape_map=binding_shape_map)
        return output
    
    def do_inference(self, inf_in_list, *, binding_shape_map=None):
        """Main function for inference
        :inf_in_list: input list.
        :binding_shape_map: {<binding_name>: <shape>}, leave it to None for fixed shape
        """
        self.cuda_ctx.push()
        _, _, h, w = inf_in_list[0].shape
        dynamic_factors = {
            "input": h * w,
            "output": h * w
        }
        self.buffers = self._allocate_buffer(dynamic_factors)
        inputs, outputs, bindings, stream = self.buffers
        if binding_shape_map:
            self.context.active_optimization_profile = 0
            for binding_name, shape in binding_shape_map.items():
                binding_idx = self.engine[binding_name]
                self.context.set_binding_shape(binding_idx, shape)
        # transfer input data to device
        for i in range(len(inputs)):
            inputs[i].host = inf_in_list[i]
            # print(f"===>{i}: ", inputs[i].host.shape, inputs[i].host.dtype)
            cuda.memcpy_htod_async(inputs[i].device, inputs[i].host, stream)
        # do inference
        # context.profiler = trt.Profiler()
        self.context.execute_async_v2(bindings=bindings,
                                      stream_handle=stream.handle)
        # copy data from device to host
        for i in range(len(outputs)):
            cuda.memcpy_dtoh_async(outputs[i].host, outputs[i].device, stream)

        stream.synchronize()
        trt_outputs = [out.host.copy() for out in outputs]
        
        self.cuda_ctx.pop()
        return trt_outputs
    
    def __del__(self):
        if self.cuda_ctx is not None:
            self.cuda_ctx.pop()
            del self.cuda_ctx


def build_onnx(torch_model, inp, onnx_path):
    print(f"Building onnx to {onnx_path}... ")
    dynamic_axes = {"input": {2: "h", 3: "w"}, "output": {2: "h", 3: "w"}}
    torch.onnx.export(
        torch_model,
        inp,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        opset_version=16,
        dynamic_axes=dynamic_axes
    )


def build_engine(onnx_path, engine_path):
    print("Building engine...")
    dynamic_shapes = {"input": ((1, 2, 32, 32), (1, 2, 32, 32), (1, 2, 64, 64))}
    TRTModel.build_engine(onnx_path, engine_path, use_fp16=False, dynamic_shapes=dynamic_shapes)


def torch_run(torch_model, x):
    with torch.no_grad():
        y = torch_model(x)
    return y


def trt_run(trt_model, x):
    trt_output = trt_model(x.numpy())
    trt_out = trt_output[0]
    return trt_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    onnx_path = "dytest.onnx"
    engine_path = "dytest.engine"
    torch_model = MyModel()

    if args.build:
        x = torch.randn(1, 2, 32, 32)
        torch.save(torch_model.state_dict(),'dytest_weight.pth')
        build_onnx(torch_model, (x), onnx_path)
        build_engine(onnx_path, engine_path)
    
    if args.run:
        trt_model = TRTModel(engine_path)
        w = torch.load("dytest_weight.pth")
        torch_model.load_state_dict(w)
        for i in range(5):
            for j in range(5):
                width = 32 + i * 8
                height = 32 + j * 8
                x = torch.randn(1, 2, height, width)
                y = torch_run(torch_model, x)
                trt_out = trt_run(trt_model, x)
                output_size = 1
                for s in y.shape:
                    output_size = output_size * s
                trt_out = trt_out[:output_size]
                trt_out = trt_out.reshape(y.shape)
                print(f"===> test for height: {height}, width: {width}, MSE: {np.square(trt_out - y.numpy()).mean()}")

    # time.sleep(10)
