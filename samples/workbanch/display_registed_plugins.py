import ctypes
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

if __name__ == "__main__":
    soFilePath = "/workspace/Github/Tensorrt-CV/build/out/libbatchednms_plugin.so"
    ctypes.cdll.LoadLibrary(soFilePath)

    plg_registry = trt.get_plugin_registry()
    for c in plg_registry.plugin_creator_list:
        print("===> ", c.name, c.plugin_version)
