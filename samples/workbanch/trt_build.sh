trtexec --onnx=xxx.onnx \
--minShapes=input_ids:1x1,token_type_ids:1x1,attention_mask:1x1 \
--optShapes=input_ids:1x16,token_type_ids:1x16,attention_mask:1x16 \
--maxShapes=input_ids:1x512,token_type_ids:1x512,attention_mask:1x512 \
--saveEngine=test.plan