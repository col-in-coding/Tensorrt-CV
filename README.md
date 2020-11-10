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
