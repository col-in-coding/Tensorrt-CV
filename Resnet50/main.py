import cv2
import torch
import onnx
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import common
import os.path
from torchvision import models
from albumentations import Resize, Compose
from albumentations.pytorch.transforms import ToTensor
from albumentations.augmentations.transforms import Normalize

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()
ONNX_FILE_PATH = 'resnet50.onnx'
ENGINE_FILE_PATH = "resnet50.engine"


def preprocess_image(img_path):
    # transformations for the input data
    transforms = Compose([
        Resize(224, 224, interpolation=cv2.INTER_NEAREST),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensor(),
    ])
    
    # read input image
    input_img = cv2.imread(img_path)
    # do transformations
    input_data = transforms(image=input_img)["image"]

    batch_data = torch.unsqueeze(input_data, 0)
    return batch_data


def postprocess(output_data):
    # get class names
    with open("imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]
    # calculate human-readable value by softmax
    confidences = torch.nn.functional.softmax(output_data, dim=1)[0] * 100
    # find top predicted classes
    _, indices = torch.sort(output_data, descending=True)
    i = 0
    # print the top classes predicted by the model
    while confidences[indices[0][i]] > 0.5:
        class_idx = indices[0][i]
        print(
            "class:",
            classes[class_idx],
            ", confidence:",
            confidences[class_idx].item(),
            "%, index:",
            class_idx.item(),
        )
        i += 1


def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch) # generate empty tensorrt.INetworkDefinition
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None, None
    print('Completed parsing of ONNX file')

    # print("****** num layers: ", network.num_layers)
    # last_layer = network.get_layer(network.num_layers - 1)
    # print("******", last_layer.get_output(0))
    # network.mark_output(last_layer.get_output(0))

    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    builder.max_workspace_size = 1 << 30
    # we have only one image in batch
    builder.max_batch_size = 1
    # use FP16 mode if possible
    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True
    # generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    if os.path.isfile(ENGINE_FILE_PATH):
        with open(ENGINE_FILE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
    else:
        # network.mark_output(network.get_layer(network.num_layers - 1).get_output(0))
        engine = builder.build_cuda_engine(network)
        with open(ENGINE_FILE_PATH, "wb") as f:
            f.write(engine.serialize())

    print("Building context")
    context = engine.create_execution_context()
    print("Completed creating Engine")

    return engine, context


def allocate_mem(engine):
    # get sizes of input and output and allocate memory required for input data and for output data
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)
    return device_input, device_output, host_output


def onnx2tensorrt(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    engine, context = build_engine(onnx_file_path)

    device_input, device_output, host_output  = allocate_mem(engine)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    # preprocess input data, C Format
    host_input = np.array(preprocess_image("turkish_coffee.jpg").numpy(), dtype=np.float32, order='C')
    cuda.memcpy_htod_async(device_input, host_input, stream)
    # run inference
    context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()
    # postprocess results
    output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, -1)
    postprocess(output_data)


if __name__ == "__main__":
    model = models.resnet50(pretrained=True)
    inp = preprocess_image("turkish_coffee.jpg").cuda()
    print(inp.shape)
    model.eval()
    model.cuda()

    # output = model(input)
    # print(output.shape)
    # postprocess(output)

    # # Convert the Pytorch Model to ONNX format
    torch.onnx.export(
        model, inp, ONNX_FILE_PATH, input_names=['input'],
        output_names=['output'], export_params=True)

    # onnx_model = onnx.load(ONNX_FILE_PATH)
    # onnx.checker.check_model(onnx_model)

    # Tensorrt Test
    onnx2tensorrt(ONNX_FILE_PATH)