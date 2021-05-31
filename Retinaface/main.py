import cv2
import torch
import torchvision
import argparse
import numpy as np
from torch import nn
from pytorch.config import cfg_mnet, cfg_re50
from pytorch.retinaface import RetinaFace
from pytorch.prior_box import PriorBox

KEEP_TOP_K = 750
NMS_THRESHOLD = 0.4


def MSE(v1, v2):
    print("Mean Square Error: ", np.square(v1 - v2).mean())


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def postprocess(outputs, confidence_threshold=0.7, cfg=cfg_re50, device='cpu'):
    loc = outputs[0].reshape((43008, 4))
    conf = outputs[1].reshape((43008, 2))
    landms = outputs[2].reshape((43008, 10))

    priorbox = PriorBox(cfg_re50, (1024, 1024))
    priors = priorbox.forward()

    # decode boxes
    boxes = decode(
        loc,
        priors.data,
        cfg_re50['variance'])
    scale = torch.Tensor([1024, 1024, 1024, 1024])
    scale = scale.to(device)
    boxes = boxes * scale

    scores = conf[:, 1]

    landms = decode_landm(
        landms,
        priors.data,
        cfg_re50['variance'])
    scale1 = np.array([1024] * 10)
    # scale1 = torch.from_numpy(scale1).to(device)
    landms = landms * scale1

    # ignore low scores
    inds = torch.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    top_k = 5000
    order = scores.topk(min(top_k, len(scores))).indices
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    keep = torchvision.ops.nms(boxes, scores, NMS_THRESHOLD)
    landms = landms[keep]
    boxes = boxes[keep]
    scores = scores[keep]
    dets = torch.hstack((boxes, scores.reshape(-1, 1)))

    # keep top-K faster NMS
    dets = dets[:KEEP_TOP_K, :]
    landms = landms[:KEEP_TOP_K, :]
    return dets.cpu().numpy(), landms.cpu().numpy()


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms


def resize_image_with_ratio(im, desired_size=1024):
    old_size = im.shape[:2]  # old_size is in (height, width) format

    im_size_max = np.max(old_size)

    ratio = 1.0
    # new_size should be in (width, height) format
    new_size = old_size
    new_im = np.zeros((desired_size, desired_size, 3), np.uint8)
    if im_size_max > desired_size:
        ratio = float(desired_size) / im_size_max
        new_size = tuple([int(x * ratio) for x in old_size])
        im = cv2.resize(im, (new_size[1], new_size[0]))

    new_im[:new_size[0], :new_size[1], :] = im
    if new_size[0] < desired_size:
        new_im[new_size[0]:, :, :] = 255
    if new_size[1] < desired_size:
        new_im[:, new_size[1]:, :] = 255
    return ratio, new_im


class MyModel(nn.Module):
    def __init__(self, cfg):
        super(MyModel, self).__init__()
        self.retinaface = RetinaFace(cfg=cfg, phase='test')

        im_h = 1024
        im_w = 1024
        self.im_height = im_h
        self.im_width = im_w
        self.scale = torch.Tensor([im_w, im_h, im_w, im_h])
        self.scale1 = torch.Tensor([
            im_w, im_h, im_w, im_h, im_w,
            im_h, im_w, im_h, im_w, im_h
        ])

        self.cfg = cfg
        self.im_h = 1024
        self.im_w = 1024

        # 先验框
        priorbox = PriorBox(cfg, image_size=(im_h, im_w))
        priors = priorbox.forward()
        self.prior_data = priors.data

        self.tempA = torch.zeros(43008, 2)
        self.tempB = torch.zeros(43008, 2)
        self.scale = torch.Tensor([im_w, im_h, im_w, im_h])

    def preprocess(self, img):
        mean = torch.Tensor((104.0, 117.0, 123.0))
        x = img.unsqueeze(0)
        x = x - mean
        x = x.transpose(1, 2).transpose(1, 3)
        return x

    def postprocess(self, loc, conf, landms, confidence_threshold=0.7):
        """
        人脸检测后处理部分只能batch_size为1
        """
        # priors = self.prior_data
        # variances = self.cfg['variance']
        # # decode boxes
        # loc = loc.reshape((43008, 4))
        # boxes = torch.cat((
        #     priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        #     priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)

        # tempA = boxes[:, 0:2]
        # tempB = boxes[:, 2:4]
        # boxes = torch.cat((tempA - tempB/2, tempA + tempB/2), 1)
        # boxes = boxes * self.scale

        # # decode landmarks
        landms = landms.reshape((43008, 10))
        # pre = landms.reshape((-1, 10))
        # landms = torch.cat((
        #     priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
        #     priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
        #     priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
        #     priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
        #     priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        #     ), dim=1)
        # landms = landms * self.scale1

        # # ignore low scores (NonZero tensorrt not support)
        conf = conf.reshape((43008, 2))
        scores = conf[:, 1]
        # inds = torch.where(scores > confidence_threshold)
        boxes = loc.reshape((43008, 4))
        # inds = torch.gt(scores, confidence_threshold)
        # boxes = boxes[inds]
        # landms = landms[inds]
        # scores = scores[inds]
        # return boxes

        # (NMS tensorrt not support)
        keep = torchvision.ops.nms(boxes, scores, NMS_THRESHOLD)
        landms = landms[keep]
        boxes = boxes[keep]
        scores = scores[keep]
        return boxes, landms, scores

    def forward(self, loc, conf, landms):
        return self.postprocess(loc, conf, landms)


def main(args):
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50

    torch_model = MyModel(cfg)
    torch_model.retinaface = load_model(torch_model.retinaface, args.trained_model, True)
    torch_model.retinaface.eval()
    print('Finished loading model!')

    # image_path = "../data/buffett.png"
    # img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # _, resized_img = resize_image_with_ratio(img_raw)
    # img = resized_img.astype(np.float32, copy=False)
    # img = torch.from_numpy(img)

    loc = torch.rand((1, 43008, 4))
    conf = torch.rand((1, 43008, 2))
    landms = torch.rand((1, 43008, 10))

    with torch.no_grad():
        boxes, _, _ = torch_model(loc, conf, landms)

    print(boxes.shape)

    # boxes = outputs
    # print(boxes.shape)
    # dets, lmdms = postprocess(outputs)
    # print(dets.shape, lmdms.shape)

    if args.build_onnx:
        torch.onnx.export(
            torch_model,
            (loc, conf, landms),
            args.onnx_path,
            export_params=True,
            input_names=['input1', 'input2', 'input3'],
            output_names=['output'],
            verbose=True,
            opset_version=11,
            enable_onnx_checker=False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retinaface')
    parser.add_argument('-m', '--trained_model', default='/training/colin/Github/Tensorrt-CV/models/retinaface/Resnet50_Final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--build-onnx', action="store_true")
    parser.add_argument("--onnx-path", type=str, default='test.onnx', help="path to pytorch model checkpoint")

    args = parser.parse_args()
    main(args)

# trtexec --onnx=test.onnx --saveEngine=test.engine --fp16 --workspace=2048
