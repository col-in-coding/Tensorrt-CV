import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import sys
from iresnet import iresnet100
from torchvision import transforms
import cv2
import os.path
import time

ONNX_FILE_PATH = "./iresnet100.onnx"
ENGINE_FILE_PATH = "./iresnet100.engine"
LOGGER = trt.Logger()


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def build_engine(onnx_file_path=ONNX_FILE_PATH, engine_file_path=ENGINE_FILE_PATH):
    builder = trt.Builder(LOGGER)
    network = builder.create_network(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, LOGGER)
    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None, None
    print('Completed parsing of ONNX file')

    builder.max_workspace_size = 1 << 30
    builder.max_batch_size = 1
    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True
    if os.path.isfile(engine_file_path):
        print("Loading Engine...")
        with open(engine_file_path, "rb") as f, \
                trt.Runtime(LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
    else:
        print("Creating Engine...")
        engine = builder.build_cuda_engine(network)
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
    context = engine.create_execution_context() 
    return engine, context


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * \
            engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


if __name__ == "__main__":

    torch_model = iresnet100("./iresnet100-73e07ba7.pth")
    torch_model.eval()
    torch_model.cuda()

    image_shape = (112,112,3)
    origin = np.random.randint(0, 255, image_shape).astype('uint8')

    x = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, 0)
    x = torch.from_numpy(x)
    x = (x - 127.5) / 128.0

    x_cu = x.cuda()
    # 15.38s, 1.7G
    # start = time.time()
    # for i in range(1000):
    #     torch_embedding = torch_model(x_cu).cpu().detach().numpy()
    # print("time consuming: ", time.time() - start) 
    # print(torch_embedding.shape)

    if not os.path.isfile(ONNX_FILE_PATH):
        print("Export ONNX")
        torch.onnx.export(
            torch_model,
            x_cu,
            ONNX_FILE_PATH,
            input_names=['input'],
            output_names=['output'],
            export_params=True,
            verbose=True
        )

    engine, context = build_engine()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # 3s, 2G
    # start = time.time()
    # for i in range(1000):

    inputs[0].host = np.array(x, dtype=np.float32, order='C')
    cuda.memcpy_htod_async(inputs[0].device, inputs[0].host, stream)
    context.execute_async_v2(
        bindings=bindings,
        stream_handle=stream.handle
    )
    cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, stream)
    trt_outputs = [out.host for out in outputs]
    stream.synchronize()
        # trt_outputs = trt_outputs.copy()

    # print("time consuming: ", time.time() - start)
    print(trt_outputs[0].shape)
    trt_embeddings = trt_outputs[0].reshape((-1, 512))

    # print("MSE: ", np.square(torch_embedding - trt_embeddings).mean())



