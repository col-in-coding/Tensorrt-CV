# 适配Tensorrt V8.4
# 使用NVIDIA官方提供的cuda-python代替原先的pycuda，（cuda runtime分离）
# 支持Multi profile，每个profile的每个binding都需要单独开辟显存
#
import numpy as np
import tensorrt as trt
from cuda import cudart

DEBUG = False


class TensorrtBaseV2:

    trt_logger = trt.Logger(trt.Logger.INFO)

    def __init__(self, plan_file_path, profiles_max_shapes=[], gpu_id=0):
        cudart.cudaSetDevice(gpu_id)
        self.engine = self._load_engine(plan_file_path)
        self.nBinding = self.engine.num_bindings
        self.nProfile = self.engine.num_optimization_profiles
        self.nInputBinding = np.sum(
            [self.engine.binding_is_input(i) for i in range(self.nBinding)])
        self.nOutputBinding = self.nBinding - self.nInputBinding
        self.nInput = self.nInputBinding // self.nProfile
        self.nOutput = self.nOutputBinding // self.nProfile
        self.nBindingPerProfile = self.nInput + self.nOutput
        self.bufferD = self._allocate_buffer(profiles_max_shapes)
        self.context = self.engine.create_execution_context()
        if DEBUG:
            print("===> num of profiles: ", self.nProfile)
            print("===> nBindingPerProfile: ", self.nBindingPerProfile)

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
        # config.set_tactic_sources(trt.TacticSource.CUBLAS_LT)

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                     workspace << 30)

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
                iprofile.set_shape(inputTi.name, min_shape, opt_shape,
                                   max_shape)
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

    def _load_engine(self, plan_file_path):
        # Force init TensorRT plugins
        trt.init_libnvinfer_plugins(None, '')
        print("===> load engine: ", plan_file_path)
        with open(plan_file_path, "rb") as f:
            engineString = f.read()
            engine = trt.Runtime(
                self.trt_logger).deserialize_cuda_engine(engineString)
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
                print(f"Bind:{i}", self.engine.get_binding_dtype(i),
                      self.engine.get_binding_shape(i),
                      self.context.get_binding_shape(i),
                      self.engine.get_binding_name(i))

        bufferD_context_inp = [int(0)] * self.nBinding
        for i in range(self.nBindingPerProfile):
            bufferD_context_inp[bindingBias + i] = self.bufferD[bindingBias +
                                                                i]

        # 将input数据拷贝到对应的显存地址
        for i in range(self.nInput):
            cudart.cudaMemcpyAsync(
                bufferD_context_inp[bindingBias + i], bufferH[i].ctypes.data,
                bufferH[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
            # cudart.cudaMemcpy(bufferD_context_inp[bindingBias + i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        # 执行推理
        # print("===> bufferD_contxt_inp: ", bufferD_context_inp)
        self.context.execute_async_v2(bufferD_context_inp, stream)
        # self.context.execute_v2(bufferD_context_inp)

        # 推理结果从GPU拷到CPU对应的bufferH
        for i in range(self.nInput, self.nBindingPerProfile):
            cudart.cudaMemcpyAsync(
                bufferH[i].ctypes.data, bufferD_context_inp[bindingBias + i],
                bufferH[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
            # cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD_context_inp[bindingBias + i],
            #                        bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        cudart.cudaStreamSynchronize(stream)
        cudart.cudaStreamDestroy(stream)
        # trt_outputs = [bufferH[i].copy() for i in range(self.nInput, self.nBindingPerProfile)]
        return bufferH[self.nInput:self.nBindingPerProfile]
