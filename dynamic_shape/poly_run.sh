polygraphy run dytest.onnx --trt --onnxrt \
--onnx-outputs mark all \
--trt-outputs mark all \
--verbose \
--trt-min-shapes input:[1,2,32,32] --trt-opt-shapes input:[1,2,64,64] --trt-max-shapes input:[1,2,64,64] \
--load-inputs custom_inputs.json \
--atol 1e-05 --rtol 1e-05 &>log.log