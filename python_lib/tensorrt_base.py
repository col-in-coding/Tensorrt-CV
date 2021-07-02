import os.path
import tensorrt as trt
import pycuda.driver as cuda


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TensorrtBase:
    """
    Parent Class
    """
    trt_logger = trt.Logger(trt.Logger.ERROR)
    # make order consistency via onnx.
    input_names = []
    output_names = []

    def __init__(self, engine_file_path, *, gpu_id=0, dynamic_factor=1):
        cuda.init()
        # Create CUDA context
        self.cuda_ctx = cuda.Device(gpu_id).make_context()
        # Prepare the runtine engine
        self.engine = self._load_engine(engine_file_path)
        self.binding_names = self.input_names + self.output_names
        self.context = self.engine.create_execution_context()
        self.buffers = self._allocate_buffer(dynamic_factor)

    @classmethod
    def build_engine(cls,
                     onnx_file_path,
                     engine_file_path,
                     *,
                     use_fp16=True,
                     dynamic_shape=None,
                     dynamic_batch_size=1):
        """Build TensorRT Engine

        :use_fp16: set mixed flop computation if the platform has fp16.
        :dynamic_shape: [min, opt, max], default None represents not using dynamic.
        :dynamic_batch_size: set it to 1 if use fixed batch size, else using max batch size
        """
        builder = trt.Builder(cls.trt_logger)
        network = builder.create_network(
            1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        config.set_tactic_sources(trt.TacticSource.CUBLAS_LT)

        # Default workspace is 2G
        config.max_workspace_size = 2 << 30

        if builder.platform_has_fast_fp16 and use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # parse ONNX
        parser = trt.OnnxParser(network, cls.trt_logger)
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        print("===> Completed parsing ONNX file")

        # default = 1 for fixed batch size
        builder.max_batch_size = 1

        if dynamic_shape and len(dynamic_shape) == 3:
            print(f"===> using dynamic shape: {str(dynamic_shape)}")
            builder.max_batch_size = dynamic_batch_size
            profile = builder.create_optimization_profile()
            min_shape, opt_shape, max_shape = dynamic_shape
            profile.set_shape(
                network.get_input(0).name, min_shape, opt_shape, max_shape)
            # profile.set_shape(network.get_input(0).name, (1, 3, 224, 224), (2, 3, 224, 224), (16, 3, 224, 224))
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

    def _allocate_buffer(self, dynamic_factor):
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
            size = abs(trt.volume(self.engine.get_binding_shape(binding))) * \
                self.engine.max_batch_size * dynamic_factor
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings[binding_idx] = int(device_mem)
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def do_inference(self, inf_in_list, *, binding_shape_map=None):
        """Main function for inference

        :inf_in_list: input list.
        :binding_shape_map: {<binding_name>: <shape>}, leave it to None for fixed shape
        """
        inputs, outputs, bindings, stream = self.buffers
        if binding_shape_map:
            self.context.active_optimization_profile = 0
            for binding_name, shape in binding_shape_map.items():
                binding_idx = self.engine[binding_name]
                self.context.set_binding_shape(binding_idx, shape)
        # transfer input data to device
        for i in range(len(inputs)):
            inputs[i].host = inf_in_list[i]
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
        return trt_outputs

    def __del__(self):
        self.cuda_ctx.pop()
        del self.cuda_ctx
