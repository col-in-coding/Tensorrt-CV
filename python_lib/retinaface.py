import numpy as np
from .tensorrt_base import TensorrtBase


class FaceDetectErr(Exception):
    def __init__(self, e):
        self.code = 1
        self.message = "Resnet Inference Error"
        super().__init__(self.message, str(e))


class RetinafaceTrt(TensorrtBase):
    input_names = ["input"]
    output_names = ["nms_num_detections",
                    "nms_boxes", "nms_scores", "nms_classes"]
    batch_size = 8
    box_scale = 1024

    def __init__(self, engine_file_path, gpu_id=0):
        super().__init__(engine_file_path, gpu_id=gpu_id)

    def __call__(self, inp):
        self.cuda_ctx.push()
        try:
            assert inp.flags['C_CONTIGUOUS']
            inp = inp.astype(np.float32, copy=False)
            inf_in_list = [inp]
            outputs = self.do_inference(
                inf_in_list)
            return outputs
        except Exception as e:
            raise FaceDetectErr(e)
        finally:
            self.cuda_ctx.pop()

    @classmethod
    def postprocess(cls, outputs):
        face_counts = outputs[0]
        bboxes = outputs[1].reshape(cls.batch_size, -1, 4) * cls.box_scale
        scores = outputs[2].reshape(cls.batch_size, -1)
        return face_counts, bboxes, scores
