import argparse
import torch
import os
import numpy as np
import tensorrt as trt
from polygraphy.json import save_json
from cuda import cudart, cuda
import time

DEBUG = False
C_CHANNEL = 2


class MyModel(torch.nn.Module):
    def __init__(self) -> None:
        super(MyModel, self).__init__()
        self.l1 = torch.nn.Linear(C_CHANNEL, C_CHANNEL, bias=False)
        self.l2 = torch.nn.Linear(C_CHANNEL, C_CHANNEL, bias=False)
        self.l3 = torch.nn.Linear(C_CHANNEL, C_CHANNEL, bias=False)

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

    trt_logger = trt.Logger(trt.Logger.WARNING)

    def __init__(self, plan_file_path, gpu_id=0):
        cudart.cudaSetDevice(gpu_id)
        profiles_max_shapes = [
            {0: (1, C_CHANNEL, 64, 32), 1: (1, C_CHANNEL, 64, 32)},
            {0: (1, C_CHANNEL, 64, 40), 1: (1, C_CHANNEL, 64, 40)},
            {0: (1, C_CHANNEL, 64, 48), 1: (1, C_CHANNEL, 64, 48)},
            {0: (1, C_CHANNEL, 64, 56), 1: (1, C_CHANNEL, 64, 56)},
            {0: (1, C_CHANNEL, 64, 64), 1: (1, C_CHANNEL, 64, 64)},
        ]
        self.engine = self._load_engine(plan_file_path)
        self.nBinding = self.engine.num_bindings
        self.nProfile = self.engine.num_optimization_profiles
        self.nInputBinding = np.sum([self.engine.binding_is_input(i) for i in range(self.nBinding)])
        self.nOutputBinding = self.nBinding - self.nInputBinding
        self.nInput = self.nInputBinding // self.nProfile
        self.nOutput = self.nOutputBinding // self.nProfile
        self.nBindingPerProfile = self.nInput + self.nOutput
        self.bufferD = self._allocate_buffer(profiles_max_shapes)
        self.context = self.engine.create_execution_context()
    
    def __call__(self, inp):
        profile_num = (inp.shape[-1] - 32) // 8
        bufferH = [np.ascontiguousarray(inp)]
        bufferH.append(np.empty(inp.shape, dtype=inp.dtype))
        trt_outputs = self.do_inference(bufferH, profile_num)
        return trt_outputs
    
    def _load_engine(self, plan_file_path):
        # Force init TensorRT plugins
        trt.init_libnvinfer_plugins(None, '')
        with open(plan_file_path, "rb") as f:
            engineString = f.read()
            engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engineString)
        print(plan_file_path)
        return engine
    
    def _allocate_buffer(self, profiles_max_shapes):
        """Allocate buffer
        开辟binding的显存空间，并准备好输出的buffer，输入可以先空着，但是主要真正输入的时候要保证内存空间连续
        :profiles_max_shapes: [{binding_idx: shape}] 对于动态尺寸，维度尺寸为-1，可以给每个profile开辟一个最大值
        """
        # bufferH中存放numpy，bufferD中存放显存地址(int)，数组长度都是binding的长度
        # nBinding = (input+output)*nprofile
        # bufferH = []
        bufferD = []
        assert len(profiles_max_shapes) == self.nProfile
        for i in range(self.nProfile):
            dynamic_shapes = profiles_max_shapes[i]
            for j in range(self.nBindingPerProfile):
                # 获取每一个binding的shape和dtype
                binding_idx = len(bufferD)
                dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
                if j in dynamic_shapes:
                    shape = dynamic_shapes[j]
                else:
                    shape = self.engine.get_binding_shape(binding_idx)
                nbytes = self._get_nbytes(shape, dtype)
                bufferD.append(cudart.cudaMalloc(nbytes)[1])
        return bufferD

    def _get_nbytes(self, shape, dtype):
        return np.empty(shape, dtype=dtype).nbytes
    
    def do_inference(self, bufferH, profile_num):
        """Main function for inference
        :bufferH: [input0, input1.. output0, output1]
                  CPU内存空间单个profile的数据，包括input和output，numpy数据类型，
                  input需要保证c continuous，output使用np.empty()占位。
        """
        assert len(bufferH) == self.nBindingPerProfile
        _, stream = cudart.cudaStreamCreate()

        # activate profile
        self.context.set_optimization_profile_async(profile_num, stream)
        cudart.cudaStreamSynchronize(stream)

        bindingBias = profile_num * self.nBindingPerProfile
        for i in range(self.nInput):
            # print(f"===> set_binding_shape: i{i} shape:{bufferH[i].shape}")
            self.context.set_binding_shape(bindingBias + i, bufferH[i].shape)

        assert self.context.all_binding_shapes_specified
        if DEBUG:
            for i in range(self.nBinding):
                print(f"Bind:{i}", self.engine.get_binding_dtype(i), self.engine.get_binding_shape(i), self.context.get_binding_shape(i), self.engine.get_binding_name(i))

        bufferD_context_inp = [int(0)] * self.nBinding
        for i in range(self.nBindingPerProfile):
            bufferD_context_inp[bindingBias + i] = self.bufferD[bindingBias + i]

        # 将input数据拷贝到对应的显存地址
        for i in range(self.nInput):
            cudart.cudaMemcpyAsync(bufferD_context_inp[bindingBias + i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
            # cudart.cudaMemcpy(bufferD_context_inp[bindingBias + i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        # 执行推理
        # print("===> bufferD_contxt_inp: ", bufferD_context_inp)
        self.context.execute_async_v2(bufferD_context_inp, stream)
        # self.context.execute_v2(bufferD_context_inp)

        # 推理结果从GPU拷到CPU对应的bufferH
        for i in range(self.nInput, self.nBindingPerProfile):
            cudart.cudaMemcpyAsync(bufferH[i].ctypes.data, bufferD_context_inp[bindingBias + i],
                                   bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
            # cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD_context_inp[bindingBias + i],
            #                        bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        cudart.cudaStreamSynchronize(stream)
        # trt_outputs = [bufferH[i].copy() for i in range(self.nInput, self.nBindingPerProfile)]
        return bufferH[self.nInput : self.nBindingPerProfile]

    @classmethod
    def build_engine(cls,
                     onnx_file_path,
                     plan_file_path,
                     *,
                     use_fp16=True,
                     optimization_profiles=[],
                     workspace=2):
        """Build TensorRT Engine
        :use_fp16: set mixed flop computation if the platform has fp16.
        :optimization_profiles: [{input_index: (min, opt, max)}], default [] represents not using dynamic.
        :workspace: maximum workspace in GB, default 2G
        """
        builder = trt.Builder(cls.trt_logger)
        network = builder.create_network(
            1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        config.set_tactic_sources(trt.TacticSource.CUBLAS_LT)

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)

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

        # Multi profiles for dynamic shape
        for dynamic_shapes in optimization_profiles:
            iprofile = builder.create_optimization_profile()
            for input_idx, dynamic_shape in dynamic_shapes.items():
                inputTi = network.get_input(input_idx)
                min_shape, opt_shape, max_shape = dynamic_shape
                iprofile.set_shape(
                    inputTi.name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(iprofile)

        print("===> Building serialized network...")
        engine = builder.build_serialized_network(network, config)
        if engine:
            print("===> Saving engine...")
            with open(plan_file_path, "wb") as f:
                f.write(engine)
            print("===> Serialized Engine is Saved at: ", plan_file_path)
        else:
            print("===> build engine error")
        return engine


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
    optimization_profiles = [
        {0: ((1, C_CHANNEL, 32, 32), (1, C_CHANNEL, 32, 32), (1, C_CHANNEL, 64, 32))},
        {0: ((1, C_CHANNEL, 32, 40), (1, C_CHANNEL, 32, 40), (1, C_CHANNEL, 64, 40))},
        {0: ((1, C_CHANNEL, 32, 48), (1, C_CHANNEL, 32, 48), (1, C_CHANNEL, 64, 48))},
        {0: ((1, C_CHANNEL, 32, 56), (1, C_CHANNEL, 32, 56), (1, C_CHANNEL, 64, 56))},
        {0: ((1, C_CHANNEL, 32, 64), (1, C_CHANNEL, 32, 64), (1, C_CHANNEL, 64, 64))},
    ]
    TRTModel.build_engine(onnx_path, engine_path, use_fp16=False, optimization_profiles=optimization_profiles)


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
        x = torch.randn(1, C_CHANNEL, 32, 32)
        torch.save(torch_model.state_dict(),'dytest_weight.pth')
        build_onnx(torch_model, (x), onnx_path)
        build_engine(onnx_path, engine_path)
    
    if args.run:
        trt_model = TRTModel(engine_path, gpu_id=2)
        w = torch.load("dytest_weight.pth")
        torch_model.load_state_dict(w)
        for i in range(5):
            for j in range(5):
                width = 32 + i * 8
                height = 32 + j * 8
                x = torch.randn(1, C_CHANNEL, height, width)
                y = torch_run(torch_model, x)
                trt_out = trt_run(trt_model, x)
                y = y.numpy()
                print(f"===> test for height: {height}, width: {width}, MSE: {np.square(trt_out - y).mean()}")

    # time.sleep(10)