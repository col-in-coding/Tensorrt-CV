polygraphy run xxx.onnx --trt --onnxrt \
--fp16 \
--onnx-outputs mark all \
--trt-outputs mark all \
--input-shapes sample:[2,4,64,64] encoder_hidden_states:[2,4,64,64] \
--atol 1e-5 --rtol 1e-5 \
--save-engine test.plan \
--pool-limit workspace:20G
