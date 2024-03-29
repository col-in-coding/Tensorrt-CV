[![Documentation](https://img.shields.io/badge/Pytorch-documentation-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![Documentation](https://img.shields.io/badge/TensorRT-documentation-brightgreen.svg)](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html)
[![Documentation](https://img.shields.io/badge/TensorRT--Python-api-brightgreen)](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html)
[![Documentation](https://img.shields.io/badge/Onnx--Graphsurgeon-docs-brightgreen)](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html)
[![Documentation](https://img.shields.io/badge/PyCUDA-documentation-brightgreen)](https://documen.tician.de/pycuda/)

# Tensorrt-CV
This implementation is totally for deployment concerns. In most cases, we got a DNN model trained by Python (like Pytorch) and the goal is low latency (measured by FPS) and high accuracy (measured by MSE) in the product. Here, Tensorrt is in use.

From my experience, the most efficient achievement is to convert the Pytorch model to ONNX and use onnx parser in tensorrt, over rebuilding the whole net and loading the weights on tensorrt, mainly when the preprocess and post-process are in consideration. The calculation of preprocess or post-process will be treated as a part of the net, saved in tensorrt engine and running on gpu.

But in some cases, there may have operator-unsupport issues (like grid_sampler) and not all operations are good for gpu (like inversing).

# Tensorrt Plugin
Current Repo is based on Tensorrt OSS, which consists of plugins, parsers and samples.
Parsers and samples are removed in this repo, and plugins will be built in separated shared libraries.

# Build the plugins

## Environment

### 1. Download Tensorrt-CV
```
git clone -b main https://github.com/col-in-coding/Tensorrt-CV.git Tensorrt-CV
cd Tensorrt-CV
git submodule update --init --recursive
```

### 2. Build plugin

Check cuda version.
```
nvcc --version
```
Check cudnn version.
```
whereis cudnn_version.h
cat path_to_cudnn/cudnn_version.h
```
Build plugin
```
mkdir build
cd build
cmake .. \
-DCUDA_VERSION=11.8.89 -DCUDNN_VERSION=8.7 \
-DGPU_ARCHS=86 \
-DBUILD_PARSERS=OFF -DBUILD_SAMPLES=OFF \
-DTRT_OUT_DIR=`pwd`/out
```

### 3. Transplant a Tensorrt-OSS plugin

- Copy the plugin from Tensorrt-OSS to the plugin folder in this repo
- Call `REGISTER_TENSORRT_PLUGIN` to register the plugin creator
- Refunction the CMakeLists.txt in it to create a shared library
- Build a shared library
- Check if the trt plugin can be registered correctly
    ```
    python samples/workbanch/display_registed_plugins.py --so-path=xxx.so
    ```
