import os
import cv2
import onnx
import torch
import argparse
import numpy as np
from torch import nn
import onnx_graphsurgeon as gs
from pytorch.config import cfg_mnet, cfg_re50
from pytorch.retinaface import RetinaFace, load_model
from pytorch.retinaface import postprocess as torch_postprocess
from pytorch.prior_box import PriorBox
from utils import resize_image_with_ratio, draw_bbox

import sys
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), '..'))
from python_lib.retinaface import RetinafaceTrt

KEEP_TOP_K = 750
NMS_THRESHOLD = 0.4


def append_nms(graph, num_classes, scoreThreshold, iouThreshold, keepTopK):
    out_tensors = graph.outputs
    bs = out_tensors[0].shape[0]

    nms_attrs = {
        'shareLocation': True,
        'backgroundLabelId': -1,
        'numClasses': num_classes,
        'topK': 1024,
        'keepTopK': keepTopK,
        'scoreThreshold': scoreThreshold,
        'iouThreshold': iouThreshold,
        'isNormalized': True,
        'clipBoxes': True
    }

    nms_num_detections = gs.Variable(name="nms_num_detections",
                                     dtype=np.int32,
                                     shape=(bs, 1))
    nms_boxes = gs.Variable(name="nms_boxes",
                            dtype=np.float32,
                            shape=(bs, keepTopK, 4))
    nms_scores = gs.Variable(name="nms_scores",
                             dtype=np.float32,
                             shape=(bs, keepTopK))
    nms_classes = gs.Variable(name="nms_classes",
                              dtype=np.float32,
                              shape=(bs, keepTopK))

    nms = gs.Node(
        op="BatchedNMSDynamic_TRT",
        attrs=nms_attrs,
        inputs=out_tensors,
        outputs=[nms_num_detections, nms_boxes, nms_scores, nms_classes])
    graph.nodes.append(nms)
    graph.outputs = [nms_num_detections, nms_boxes, nms_scores, nms_classes]

    return graph


class MyModel(nn.Module):
    def __init__(self, cfg):
        super(MyModel, self).__init__()
        self.retinaface = RetinaFace(cfg=cfg, phase='test')

        im_h = 1024
        im_w = 1024
        self.cfg = cfg

        # 先验框
        priorbox = PriorBox(cfg, image_size=(im_h, im_w))
        priors = priorbox.forward()
        self.prior_data = priors.data.unsqueeze(0)

        self.mean = torch.Tensor((104.0, 117.0, 123.0))

    def preprocess(self, imgs):
        x = imgs - self.mean
        x = x.transpose(1, 2).transpose(1, 3)
        return x

    def postprocess(self, loc, conf):
        """
        人脸检测后处理部分只能batch_size为1
        """
        priors = self.prior_data
        variances = self.cfg['variance']
        # decode boxes
        boxes = torch.cat(
            (priors[:, :, :2] + loc[:, :, :2] * variances[0] * priors[:, :, 2:],
             priors[:, :, 2:] * torch.exp(loc[:, :, 2:] * variances[1])), 2)

        tempA = boxes[:, :, 0:2]
        tempB = boxes[:, :, 2:4]
        boxes = torch.cat((tempA - tempB / 2, tempA + tempB / 2), 2)

        scores = conf[:, :, 1]
        return boxes.reshape(-1, 43008, 1, 4), scores.reshape(-1, 43008, 1)

    def forward(self, imgs):
        x = self.preprocess(imgs)
        loc, conf, _ = self.retinaface(x)
        return self.postprocess(loc, conf)


def main(args):
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50

    torch_model = MyModel(cfg)
    torch_model.retinaface = load_model(torch_model.retinaface,
                                        args.trained_model, True)
    torch_model.retinaface.eval()
    print('Finished loading model!')

    imgs = torch.rand((args.batch_size, 1024, 1024, 3), dtype=torch.float32)

    if args.build_onnx:
        print("build onnx...", args.onnx_path)
        torch.onnx.export(torch_model,
                          imgs,
                          args.onnx_path,
                          export_params=True,
                          input_names=['input'],
                          output_names=['output1', 'output2'],
                          verbose=True,
                          opset_version=11,
                          enable_onnx_checker=False)

        print("add nms...")
        graph = gs.import_onnx(onnx.load(args.onnx_path))
        graph = append_nms(graph,
                           num_classes=1,
                           scoreThreshold=0.95,
                           iouThreshold=0.4,
                           keepTopK=20)

        # Remove unused nodes, and topologically sort the graph.
        graph.cleanup().toposort().fold_constants().cleanup()

        # Export the onnx graph from graphsurgeon
        out_name = 'retinaface_via_nms.onnx'
        onnx.save_model(gs.export_onnx(graph), out_name)
        print("Saving the ONNX model to {}".format(out_name))

    if args.build_engine:
        os.system(
            f"trtexec --onnx=retinaface_via_nms.onnx --saveEngine={args.engine_path} --fp16 --workspace=3000"
        )

    if args.test:
        # Using test image
        img = cv2.imread("../data/buffett.png")
        _, img1024 = resize_image_with_ratio(img)

        # ######  Pytorch Run  ##############################################
        # imgs = torch.from_numpy(img1024).unsqueeze(0)
        # with torch.no_grad():
        #     x = torch_model.preprocess(imgs)
        #     torch_out = torch_model.retinaface(x)
        # batch_locs, batch_confs, batch_landms = torch_out
        # boxes, landms = torch_postprocess([batch_locs[0], batch_confs[0], batch_landms[0]])

        # ######  TensorRT Run  ############################################
        imgs = np.expand_dims(img1024, 0)
        net = RetinafaceTrt(engine_file_path=args.engine_path)
        outputs = net(imgs)
        face_counts, bboxes, scores = RetinafaceTrt.postprocess(outputs)
        draw_bbox(img, bboxes[0])
        cv2.imwrite("test.png", img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retinaface')
    parser.add_argument(
        '-m', '--trained_model',
        default='/training/colin/Github/Tensorrt-CV/models/retinaface/Resnet50_Final.pth',
        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network',
                        default='resnet50',
                        help='Backbone network mobile0.25 or resnet50')
    parser.add_argument("--batch-size", default=8)
    parser.add_argument("--build-onnx", action="store_true")
    parser.add_argument("--onnx-path", type=str, default='retinaface_no_nms.onnx')
    parser.add_argument("--build-engine", action="store_true")
    parser.add_argument("--engine-path", type=str, default='retinaface.engine')
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    main(args)
