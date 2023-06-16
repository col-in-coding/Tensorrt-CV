import cv2
import numpy as np


def MSE(v1, v2):
    print("Mean Square Error: ", np.square(v1 - v2).mean())


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


def draw_bbox(image, bboxes):
    for box in bboxes:
        x1, y1, x2, y2 = box[:4]
        cv2.rectangle(
            image,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (255, 0, 0),
            1
        )
