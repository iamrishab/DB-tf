import os
import cv2
import numpy as np
from shapely.geometry import Polygon
import pyclipper
from db_config import cfg

import warnings
warnings.filterwarnings('ignore')

def _distance(xs, ys, point_1, point_2):
    '''
    compute the distance from point to a line
    ys: coordinates in the first axis
    xs: coordinates in the second axis
    point_1, point_2: (x, y), the end of the line
    '''
    square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
    square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
    square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

    cosin = (square_distance - square_distance_1 - square_distance_2) / (2 * np.sqrt(square_distance_1 * square_distance_2))
    square_sin = 1 - np.square(cosin)
    square_sin = np.nan_to_num(square_sin)

    result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / square_distance)
    result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
    return result

def _extend_line(point1, point2, result):
    ex_point_1 = (int(round(point1[0] + (point1[0] - point2[0]) * (1 + cfg.SHRINK_RATIO))),
                  int(round(point1[1] + (point1[1] - point2[1]) * (1 + cfg.SHRINK_RATIO))))
    cv2.line(result, tuple(ex_point_1), tuple(point1), 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
    ex_point_2 = (int(round(point2[0] + (point2[0] - point1[0]) * (1 + cfg.SHRINK_RATIO))),
                  int(round(point2[1] + (point2[1] - point1[1]) * (1 + cfg.SHRINK_RATIO))))
    cv2.line(result, tuple(ex_point_2), tuple(point2), 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
    return ex_point_1, ex_point_2

def _validate_polygons(polys, tags, h, w):

    if len(polys) == 0:
        return polys, tags
    for poly in polys:
        poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)

    for i in range(len(polys)):
        area = _polygon_area(polys[i])
        if abs(area) < 1:
            tags[i] = True
        if area > 0:
            polys[i] = polys[i][::-1, :]
    return polys, tags

def _polygon_area(poly):
    edge = 0
    for i in range(poly.shape[0]):
        next_index = (i + 1) % poly.shape[0]
        edge += (poly[next_index, 0] - poly[i, 0]) * (poly[next_index, 1] - poly[i, 1])
    return edge / 2.


def make_score_map(text_polys, tags, h, w):
    min_text_size = cfg.MIN_TEXT_SIZE
    shrink_ratio = cfg.SHRINK_RATIO

    text_polys, ignore_tags = _validate_polygons(text_polys, tags, h, w)
    score_map = np.zeros((h, w), dtype=np.float32)
    mask = np.ones((h, w), dtype=np.float32)

    for i in range(len(text_polys)):
        polygon = text_polys[i]
        height = max(polygon[:, 1]) - min(polygon[:, 1])
        width = max(polygon[:, 0]) - min(polygon[:, 0])
        if ignore_tags[i] or min(height, width) < min_text_size:
            cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
            ignore_tags[i] = True
        else:
            polygon_shape = Polygon(polygon)
            distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
            subject = [tuple(l) for l in text_polys[i]]
            padding = pyclipper.PyclipperOffset()
            padding.AddPath(subject, pyclipper.JT_ROUND,
                            pyclipper.ET_CLOSEDPOLYGON)
            shrinked = padding.Execute(-distance)
            if shrinked == []:
                cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
                continue
            shrinked = np.array(shrinked[0]).reshape(-1, 2)
            cv2.fillPoly(score_map, [shrinked.astype(np.int32)], 1)

    return score_map, mask

def make_border_map(text_polys, tags, h, w):

    canvas = np.zeros([h, w], dtype=np.float32)
    mask = np.zeros([h, w], dtype=np.float32)

    for i in range(len(text_polys)):
        if tags[i]:
            continue
        canvas, mask = _draw_border_map(text_polys[i], canvas, mask)
    threshold_map = canvas * (cfg.THRESH_MAX - cfg.THRESH_MIN) + cfg.THRESH_MIN

    return threshold_map, mask

def _draw_border_map(poly, canvas, mask):
    poly = np.array(poly).copy()
    assert poly.ndim == 2
    assert poly.shape[1] == 2

    poly_shape = Polygon(poly)
    if poly_shape.area <= 0:
        return
    distance = poly_shape.area * (1 - np.power(cfg.SHRINK_RATIO, 2)) / poly_shape.length
    subject = [tuple(l) for l in poly]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND,
                    pyclipper.ET_CLOSEDPOLYGON)

    padded_polygon = np.array(padding.Execute(distance)[0])
    cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

    xmin = padded_polygon[:, 0].min()
    xmax = padded_polygon[:, 0].max()
    ymin = padded_polygon[:, 1].min()
    ymax = padded_polygon[:, 1].max()
    width = xmax - xmin + 1
    height = ymax - ymin + 1

    poly[:, 0] = poly[:, 0] - xmin
    poly[:, 1] = poly[:, 1] - ymin

    xs = np.broadcast_to(
        np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
    ys = np.broadcast_to(
        np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

    distance_map = np.zeros(
        (poly.shape[0], height, width), dtype=np.float32)
    for i in range(poly.shape[0]):
        j = (i + 1) % poly.shape[0]
        absolute_distance = _distance(xs, ys, poly[i], poly[j])
        distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
    distance_map = distance_map.min(axis=0)

    xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
    xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
    ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
    ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
    canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
        1 - distance_map[
            ymin_valid - ymin:ymax_valid - ymax + height,
            xmin_valid - xmin:xmax_valid - xmax + width],
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

    return canvas, mask