
# Retinaface

## Description
This project is based on https://github.com/biubug6/Pytorch_Retinaface

Our procedure is still PyTorch => Onnx => TensorRT

But the problem is  

* NMS (NonMaxSuppression) inconsistency. [torchvision.ops.nms](https://pytorch.org/vision/stable/ops.html) is unrecognizable for tensorrt
* TopK filtering unsopported

So, at the begining, I gave up involving postprocess in tensorrt inference.
But soon, I find [batchedNMSPlugin](https://github.com/NVIDIA/TensorRT/blob/master/plugin/batchedNMSPlugin/README.md). I implemented it in current version of tensorrt inference, the trade off is there is not landmarks output.

To make prior boxes as a constant, the detection network input image resolution is fixed to 1024x1024.

## To be solved
* make landmarks output available
* dynamic batch input
* dynamic image shape input