import ctypes
import argparse
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--so-path", type=str, required=True, help="shared library path (libxxx.so)")
    args = parser.parse_args()

    soFilePath = args.so_path
    ctypes.cdll.LoadLibrary(soFilePath)

    plg_registry = trt.get_plugin_registry()
    for c in plg_registry.plugin_creator_list:
        print("===> ", c.name, c.plugin_version)
