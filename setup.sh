# Tensorrt Lib
export TRT_LIBPATH='/workspace/TensorRT-8.5.1.7/lib'
export LD_LIBRARY_PATH="${TRT_LIBPATH}:/usr/local/cuda/lib64"

# Tensorrt OSS
export TRT_OSSPATH='/workspace/TensorRT'

# # Tensorrt Plugin
# export PLUGIN_LIBS="$TRT_OSSPATH/build/out/libnvinfer_plugin.so"
# export LD_PRELOAD=${LD_PRELOAD}:${PLUGIN_LIBS}