import numpy as np

def get_rect(bbox):
    x1 = float(bbox[0]) # w
    y1 = float(bbox[1]) # h
    x2 = float(bbox[2])
    y2 = float(bbox[3])
    x3 = float(bbox[4])
    y3 = float(bbox[5])
    x4 = float(bbox[6])
    y4 = float(bbox[7])
    xmin = min([x1, x2, x3, x4])
    xmax = max([x1, x2, x3, x4])
    ymin = min([y1, y2, y3, y4])
    ymax = max([y1, y2, y3, y4])
    return ymin, xmin, ymax, xmax

def bboxes_jaccard(bboxes1, bboxes2):
    """Computing jaccard index between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
    jaccard = int_vol / (vol1 + vol2 - int_vol)
    return jaccard

def bboxes_sort(scores, bboxes):
    """Sort bounding boxes by decreasing order and keep only the top_k
    """
    # if priority_inside:
    #     inside = (bboxes[:, 0] > margin) & (bboxes[:, 1] > margin) & \
    #         (bboxes[:, 2] < 1-margin) & (bboxes[:, 3] < 1-margin)
    #     idxes = np.argsort(-scores)
    #     inside = inside[idxes]
    #     idxes = np.concatenate([idxes[inside], idxes[~inside]])
    idxes = np.argsort(-scores)
    scores = scores[idxes]
    bboxes = bboxes[idxes]
    keep_bboxes = np.ones(scores.shape, dtype = np.bool)
    return bboxes[keep_bboxes]

def bboxes_nms(bboxes, nms_threshold = 0.45):
    """Apply non-maximum selection to bounding boxes.
    """
    bboxes_rect = []
    for bbox in bboxes:
        ymin, xmin, ymax, xmax = get_rect(bbox)
        bboxes_rect.append([ymin, xmin, ymax, xmax])
    scores = bboxes[:,8]
    keep_bboxes = np.ones(scores.shape, dtype = np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            overlap = bboxes_jaccard(bboxes_rect[i], bboxes_rect[(i+1):])
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], overlap < nms_threshold)
    idxes = np.where(keep_bboxes)
    return bboxes[idxes]