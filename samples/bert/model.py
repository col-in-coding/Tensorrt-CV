import sys
import numpy as np
sys.path.append("../python_lib")

from tensorrt_base_v3 import TensorrtBaseV3

DEBUG = False


class BertForMaskedLMTRT(TensorrtBaseV3):

    def __init__(self, plan_file_path, gpu_id=0):
        profiles_max_shapes = [{
            0: (1, 512),
            1: (1, 512),
            2: (1, 512),
            3: (1, 512, 30522),
        }]
        super().__init__(plan_file_path, profiles_max_shapes=profiles_max_shapes, gpu_id=gpu_id)
    
    def __call__(self, input_ids, token_type_ids, attention_mask):
        profile_num = 0
        output_shape = (input_ids.shape[0], input_ids.shape[1], 30522)
        # print(output_shape)
        bufferH = [
            np.ascontiguousarray(input_ids.astype(np.int32, copy=False)),
            np.ascontiguousarray(token_type_ids.astype(np.int32, copy=False)),
            np.ascontiguousarray(attention_mask.astype(np.int32, copy=False)),
            np.empty(output_shape, dtype=np.float32)
        ]
        trt_outputs = self.do_inference(bufferH, profile_num)
        return trt_outputs