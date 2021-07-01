import numpy as np
from .tensorrt_base import TensorrtBase


class ResnetInferenceErr(Exception):
    def __init__(self, e):
        self.code = 1
        self.message = "Resnet Inference Error"
        super().__init__(self.message, str(e))


class Resnet(TensorrtBase):
    input_names = ["input"]
    output_names = ["output"]

    def __init__(self, engine_file_path, gpu_id=0):
        super().__init__(engine_file_path, gpu_id=gpu_id)

    def __call__(self, inp):
        self.cuda_ctx.push()
        try:
            assert inp.shape[-3:] == (3, 224, 224)
            assert inp.flags['C_CONTIGUOUS']
            inp = inp.astype(np.float32, copy=False)
            inf_in_list = [inp]
            binding_shape_map = {'input': inp.shape}
            outputs = self.do_inference(
                inf_in_list, binding_shape_map=binding_shape_map)
            return outputs
        except Exception as e:
            raise ResnetInferenceErr(e)
        finally:
            self.cuda_ctx.pop()

    @classmethod
    def postprocess(cls, outputs):
        return outputs[0].reshape(-1, 1000)
