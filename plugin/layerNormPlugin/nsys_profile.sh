nsys profile --stats=true --sample=cpu \
--trace=cuda,nvtx,cublas,cudnn \
python testLayerNormPlugin.py