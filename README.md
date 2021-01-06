# TensorRT 踩坑日志（Bullshit Diary）

<b><i>2020-11-08:  </i></b>  
Description:  
Parsing ONNX file of GAN model.
```
ERROR: Failed to parse the ONNX file.
In node 60 (importInstanceNormalization): UNSUPPORTED_NODE: Assertion failed: !isDynamic(tensor_ptr->getDimensions()) && "InstanceNormalization does not support dynamic inputs!"
```
Solution:  
Update to V7.2!!!  

<b><i>2020-11-10:  </i></b>  
Description:  
Desrilizing the engine file on excate same environment.
```
[TensorRT] ERROR: INVALID_ARGUMENT: getPluginCreator could not find plugin InstanceNormalization_TRT version 1
[TensorRT] ERROR: safeDeserializationUtils.cpp (322) - Serialization Error in load: 0 (Cannot deserialize plugin since corresponding IPluginCreator not found in Plugin Registry)
[TensorRT] ERROR: INVALID_STATE: std::exception
[TensorRT] ERROR: INVALID_CONFIG: Deserialize the cuda engine failed.
```
Solution:  
Doing parsing the ONNX model before desirlizing the engine file.  
```
...
with open(onnx_file_path, 'rb') as model:
if not parser.parse(model.read()):
    print('ERROR: Failed to parse the ONNX file.')
    for error in range(parser.num_errors):
        print(parser.get_error(error))
    return None, None
...
if os.path.isfile(ENGINE_FILE_PATH):
    with open(ENGINE_FILE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

```

<b><i>2020-11-11:  </i></b>  
Description:  
Got an error when allocating memory.
```
...
    self.stream = cuda.Stream()
pycuda._driver.LogicError: explicit_context_dependent failed: invalid device context - no currently active context?
```
Solution:  
```
import pycuda.driver as cuda
## Add autoinit to init pycuda driver
import pycuda.autoinit

```
(pycuda.autoinit would take a some GPU memory)

<b><i>2020-12-03:  </i></b>  
Description:  
transfer pytorch model to ONNX file
```
RuntimeError: Failed to export an ONNX attribute, since it's not constant, please try to make things (e.g., kernel size) static if possible
```
Solution:  
find the error raising place from .../envs/pytorch1.6/lib/python3.7/site-packages/torch/onnx/
```
print(v.node())
# to get the error node location. 
# avg = F.avg_pool2d(feat32, feat32.size()[2:])
# add print(feat32.size()[2:]) to get the value
# set it to constant
```

<b><i>2020-12-07:  </i></b>  
Description:  
Parsing ONNX in tensorrt
```
[TensorRT] INTERNAL ERROR: Assertion failed: cublasStatus == CUBLAS_STATUS_SUCCESS
../rtSafe/cublas/cublasLtWrapper.cpp:279
Aborting...
[TensorRT] ERROR: ../rtSafe/cublas/cublasLtWrapper.cpp (279) - Assertion Error in getCublasLtHeuristic: 0 (cublasStatus == CUBLAS_STATUS_SUCCESS)
```
Solution:  
This is caused by cublas LT 10.2 BUG. Solved by disabling cublasLT
```
trtexec --onnx=xxx.onnx --tacticSources=-cublasLt,+cublas --workspace=2048 --fp16 --saveEngine=xxx.engine
```

<b><i>2020-12-08:  </i></b>  
Description:  
Allocate Buffer. Memory location bindings should be in order of binding index from engine. 
Sometimes, it is not the same as input/output order


<b><i>2020-12-11:  </i></b>  
Description:  
When run tensorrt with saved engine
```
pycuda._driver.LogicError: cuMemcpyHtoDAsync failed: invalid argument
```
Solution: This may caused by input memory error. Check if the input dtype is Float64

<b><i>2021-01-06:  </i></b>  
Description:  
I got this error when using Tensorrt and PyTorch together. I used PyTorch GPU calculation for the preprocessing.
```
[TensorRT] ERROR: safeContext.cpp (184) - Cudnn Error in configure: 7 (CUDNN_STATUS_MAPPING_ERROR)
[TensorRT] ERROR: FAILED_EXECUTION: std::exception
```
I also found when splitting them to different processes, the error disappears. But in my case, I have a very large image data to be used in post-process, which would add extra latency during passing it between the processes.  
I tried to use cupy instead of PyTorch and also got this error ;(

Solution: adding cuda context push and pop on the two ends of doing inference

```
cuda.init()
cuda_ctx = cuda.Device(gpu_id).make_context()
cuda_ctx.push()
... doing inferennce
cuda_ctx.pop()
```
