[![Documentation](https://img.shields.io/badge/Pytorch-documentation-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![Documentation](https://img.shields.io/badge/TensorRT-documentation-brightgreen.svg)](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html) 
[![Documentation](https://img.shields.io/badge/TensorRT--Python-api-brightgreen)](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html)
[![Documentation](https://img.shields.io/badge/Onnx--Graphsurgeon-docs-brightgreen)](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html)
[![Documentation](https://img.shields.io/badge/PyCUDA-documentation-brightgreen)](https://documen.tician.de/pycuda/)

# Tensorrt-CV
This implementation is totally for deployment concern. In most cases, we got an DNN model trained by python (like Pytorch) and the goal is low latency (measured by FPS) and high accuracy (measured by MSE) in product. Here, Tensorrt is in use.  

For my expericence, the most efficient achievement is to convert Pytorch model to ONNX and use onnx parser in tensorrt, over rebuilding the whole net and loading the weights on tensorrt, particularly when the preprocess and postprocess are in consideration. The calculation of preprocess or postprocess will be treated as a part of the net, saved in tensorrt engine and runing on gpu.  

But in some cases, there may have operator-unsupport issues (like grid_sampler) and not all operations are good for gpu (like inversing).

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

<b><i>2021-05-29:  </i></b>  
Description:  
When I deserializing an saved engine that I build from different place and it faild.

Solution: We first need to load all custom plugins shipped with TensorRT manually.

```
# Force init TensorRT plugins
trt.init_libnvinfer_plugins(None,'')
with open(engine_file_path, "rb") as f, \
        trt.Runtime(self.trt_logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
return engine
```

<b><i>2021-05-31:  </i></b>  
Description:
Pytorch to onnx success, but onnx to tensorrt engine faild,and throw an error  
```
[05/28/2021-20:30:26] [I] [TRT] /training/colin/Github/TensorRT/parsers/onnx/ModelImporter.cpp:139: No importer registered for op: ScatterND. Attempting to import as plugin.
[05/28/2021-20:30:26] [I] [TRT] /training/colin/Github/TensorRT/parsers/onnx/builtin_op_importers.cpp:3775: Searching for plugin: ScatterND, plugin_version: 1, plugin_namespace: 
[05/28/2021-20:30:26] [E] [TRT] INVALID_ARGUMENT: getPluginCreator could not find plugin ScatterND version 1
```

Solution: ScatterND is for indexing, when you got operations like A[:, 0:2] = B. So the solotion is to substitute them with splits and concatinations.  

<b><i>2021-07-05:  </i></b>  
Description:
This is caused by BatchNorm1d, upstreamed by full connection layer  
```
[TensorRT] ERROR: (Unnamed Layer* 11) [Shuffle]: at most one dimension may be inferred
ERROR: Failed to parse the ONNX file.
In node 1 (scaleHelper): UNSUPPORTED_NODE: Assertion failed: dims.nbDims == 4 || dims.nbDims == 5
```

Solution: From Pytorch documentation, the input of nn.BatchNorm1d could be (N, C, L) or (N, L). Unsqueeze the output of fc layer to (N, C, L), and it works.
```
...
fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
features = nn.BatchNorm1d(num_features)
...
x = fc(x)
x = x.unsqueeze(-1)
x = features(x)
```
