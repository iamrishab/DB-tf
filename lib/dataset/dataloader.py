import os
import cv2
import math
import numpy as np



def load_gt_labels(gt_path):
    """
    load pts
    :param gt_path:
    :return:
    """
    assert os.path.exists(gt_path), '{} is not exits'.format(gt_path)
    boxes = []
    with open(gt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(',')
            pts = [float(parts[i]) for i in range(len(parts))]
            poly = np.array(pts).reshape((-1, 2))
            boxes.append(poly)

    return boxes