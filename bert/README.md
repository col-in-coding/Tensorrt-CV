## Download Bert Model

1. install git lfs
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```
2. download model
```
git clone https://huggingface.co/bert-base-uncased
```

## Build ONNX
run `python torch2onnx.py` to generate onnx model, and run `python onnxrun.py` for onnxruntime-gpu inference.

## Build Tensorrt Engine
run `python onnx2trt.py` to build tensorrt engine and do inference.