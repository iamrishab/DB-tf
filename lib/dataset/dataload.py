# coding:utf-8
import glob
import cv2
import time
import os
import random
import json
import numpy as np
from shapely.geometry import Polygon

from lib.dataset.generator_enqueuer import GeneratorEnqueuer
from lib.dataset.geo_map_cython_lib import gen_geo_map
from east_config import cfg
from lib.dataset.img_aug import det_aug

CYTHON = True

def exp_box_by_box(box, right_threshold=0.02, left_threshold=0.01):
    right_index = [0, 1, 0, 1, 6, 7, 6, 7]  #
    left_index = [2, 3, 2, 3, 4, 5, 4, 5]
    # right
    vector = box - box[right_index]
    new_box = right_threshold * vector + box

    # left
    vector = new_box - new_box[left_index]
    new_box = left_threshold * vector + new_box

    return new_box

def exp_box_by_muti_boxes(box, right_threshold=0.02, left_threshold=0.01):
    right_index = [0, 1, 0, 1, 6, 7, 6, 7]  #
    left_index = [2, 3, 2, 3, 4, 5, 4, 5]
    # right
    vector = box - box[:, right_index]
    new_box = right_threshold * vector + box

    # left
    vector = new_box - new_box[:,left_index]
    new_box = left_threshold * vector + new_box

    return new_box

def exp_thin_box(box):
    """
    扩充瘦框
    :param box:
    :return:
    """
    width = box[1][0]-box[0][0]
    height = box[3][1] - box[0][1]
    if height > width:
        ratio = (height-width)/width
        return exp_box_by_box(box.reshape([8,]), ratio/2.5, ratio/2.5)
    else:
        return box


def sort_points(xy_np):
    """
    返回四个点顺序
    :param xy_np: numpy
    :return:
    """
    xy_list = xy_np.reshape(4, 2).tolist()
    sort_x_list = sorted(xy_list, key=lambda x: x[0])
    left_point = sort_x_list[0:2]
    right_point = sort_x_list[2:4]

    sort_left_y_list = sorted(left_point, key=lambda x: x[1])
    sort_right_y_list = sorted(right_point, key=lambda x: x[1])

    left_top_p = sort_left_y_list[0]
    left_down_p = sort_left_y_list[1]

    right_top_p = sort_right_y_list[0]
    right_down_p = sort_right_y_list[1]

    new_point_list = [left_top_p, right_top_p, right_down_p, left_down_p]

    return np.array(new_point_list)

def correct_box(boxes_np, img_height, img_width):
    """
    修正box范围，防止box超出图片边界范围
    :param boxes_np: [-1, 4, 2]
    :param img_height:
    :param img_width:
    :return:
    """
    # 首先检查x,y是否小于0
    index_x_zero = np.where(boxes_np[:, :, 0] < 0)
    if len(index_x_zero[0]) != 0:
        boxes_np[index_x_zero[0], index_x_zero[1], 0] = 0
    index_y_zero = np.where(boxes_np[:, :, 1] < 0)
    if len(index_y_zero[0]) != 0:
        boxes_np[index_y_zero[0], index_y_zero[1], 1] = 0

    # 检查x,y是否超出图像边界框
    index_x_img = np.where(boxes_np[:, :, 0] > img_width)
    if len(index_x_img[0]) != 0:
        boxes_np[index_x_img[0], index_x_img[1], 0] = img_width
    index_y_img = np.where(boxes_np[:, :, 1] > img_height)
    if len(index_y_img[0]) != 0:
        boxes_np[index_y_img[0], index_y_img[1], 1] = img_height

    return boxes_np

def get_images(train_img_dir):
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(
            os.path.join(train_img_dir, '*.{}'.format(ext))))
    return files

def context_has_one(context):
    """
    判断文本的开头和末尾是否有１
    :param context:
    :return:
    """
    context = context.replace('$', '')

    begin = False
    end = False
    if len(context) == 0:
        return begin, end

    if context[0] == '1':
        begin = True
    if context[-1] == '1':
        end = True

    return begin, end

def load_annoataion(label_file, h, w):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''

    def load_data(datas, label, total_text_polys, total_text_tags):
        for data in datas:
            # total_text_polys = []
            # 转为最小包围矩形
            rect = cv2.minAreaRect(np.array(data['bbox'][0:8]).reshape((4, 2)))
            box = cv2.boxPoints(rect)

            # 是否将所有框进行外扩
            if cfg.TRAIN.EXP_ALL_BOX or cfg.TRAIN.EXP_THIN:
                box = sort_points(box)

            # 对瘦框进行外扩
            if cfg.TRAIN.EXP_THIN:
                box = exp_thin_box(box)
                box = box.reshape((4,2))

            # 是否进行１的外扩
            if cfg.TRAIN.EXP_1_BOX and len(data['context']) !=0 :
                begin, end = context_has_one(data['context'])
                if begin and end:
                    box = sort_points(box)
                    box = exp_box_by_box(box.reshape(8,), 0.02, 0.01)
                    box = box.reshape(4, 2)
                elif begin or end:
                    box = sort_points(box)
                    if begin is True:
                        box = exp_box_by_box(box.reshape(8, ), 0, 0.01)
                    else:
                        box = exp_box_by_box(box.reshape(8, ), 0.02, 0.0)
                    box = box.reshape(4, 2)
                else:
                    pass

            total_text_polys.append(box)
            total_text_tags.append(False)

    if not os.path.exists(label_file):
        return None, None, None

    total_text_polys = []
    total_text_tags = []
    try:
        json_data = json.loads(open(label_file, encoding='utf-8').read(),
                               encoding='bytes')

        load_data(json_data['text'], 1, total_text_polys, total_text_tags)

        if len(total_text_polys) == 0:
            return None, None, None

        if cfg.TRAIN.EXP_ALL_BOX:
            box_np = np.array(total_text_polys).reshape([-1, 8])
            # 外扩检测框
            exp_boxes = exp_box_by_muti_boxes(box_np)
            # 修正box的边缘
            text_polys = correct_box(exp_boxes.reshape([-1, 4, 2]), h, w)
        else:
            text_polys = total_text_polys

        if 'mask_data' in json_data.keys():
            mask_data = json_data['mask_data']
        else:
            mask_data = []

        return np.array(text_polys, dtype=np.float32), np.array(total_text_tags, dtype=np.bool), mask_data
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('errror label ', label_file)
        return None, None, None


def polygon_area(poly):
    '''
    compute area of a polygon
    :param poly:
    :return:
    '''

    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge) / 2.


def check_and_validate_polys(polys, tags, img_size):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    (h, w) = img_size
    if polys.shape[0] == 0:
        return polys

    # 检查数据是否有越界的
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)

    validated_polys = []
    validated_tags = []
    for poly, tag in zip(polys, tags):
        p_area = polygon_area(poly)
        if abs(p_area) < 1:
            # TODO: print poly
            print('invalid poly:', poly)
            continue
        if p_area > 0:
            print('poly in wrong direction')
            poly = poly[(0, 3, 2, 1), :]
        validated_polys.append(poly)
        validated_tags.append(tag)
    return np.array(validated_polys), np.array(validated_tags)


def crop_area(im, polys, tags, crop_background=False, max_tries=50):
    '''
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    '''
    h, w, _ = im.shape
    pad_h = h // 10
    pad_w = w // 10
    h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
    w_array = np.zeros((w + pad_w * 2), dtype=np.int32)
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx + pad_w:maxx + pad_w] = 1 #投影后宽的投影位置都设置为1,其余位置为0
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny + pad_h:maxy + pad_h] = 1 #投影后高的位置设置为1,其余为0
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0] #返回索引
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0: #检查索引的长度是否大于0
        return im, polys, tags
    for i in range(max_tries): #最大尝试数量
        xx = np.random.choice(w_axis, size=2)#从w中以概率P，随机选择2个, p没有指定的时候相当于是一致的分布
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w - 1)#强制转换边界 <0,变成0,大于w-1,变成w-1
        xmax = np.clip(xmax, 0, w - 1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h - 1)
        ymax = np.clip(ymax, 0, h - 1)

        # 最大的x-最小的小于设定比例乘于图片的宽
        if xmax - xmin < cfg["TRAIN"]["MIN_CROP_SIDE_RATIO"] * w or \
                ymax - ymin < cfg["TRAIN"]["MIN_CROP_SIDE_RATIO"] * h:
            # area too small
            continue

        if polys.shape[0] != 0:
            #[[ True  True  True  True] [ True  True  True  True] [ True  True  True  True] [ True  True  True  True]]
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            #[0 1 2 3] 返回索引,则证明四个地址都有
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = []
        if len(selected_polys) == 0:
            # no text in this area
            if crop_background:
                return im[ymin:ymax + 1, xmin:xmax + 1, :], polys[selected_polys], tags[selected_polys]
            else:
                continue
        im = im[ymin:ymax + 1, xmin:xmax + 1, :]
        polys = polys[selected_polys]
        tags = tags[selected_polys]
        polys[:, :, 0] -= xmin
        polys[:, :, 1] -= ymin
        return im, polys, tags

    return im, polys, tags


def shrink_poly(poly, r):
    '''
    fit a poly inside the origin poly, maybe bugs here...
    used for generate the score map
    :param poly: the text poly
    :param r: r in the paper
    :return: the shrinked poly
    '''
    # shrink ratio
    R = cfg.TRAIN.SHRINK_RATIO
    # find the longer pair
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
            np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
        ## p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
    else:
        ## p0, p3
        # print poly
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
    return poly


def point_dist_to_line(p1, p2, p3):
    """
    点到两点间直线距离，p3到p1和p2两点间直线距离
    :param p1:
    :param p2:
    :param p3:
    :return:
    """
    # compute the distance from p3 to p1-p2
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def fit_line(p1, p2):
    # fit a line ax+by+c = 0
    if p1[0] == p1[1]:
        return [1., 0., -p1[0]]
    else:
        [k, b] = np.polyfit(p1, p2, deg=1)
        return [k, -1., b]


def line_cross_point(line1, line2):
    # line1 0= ax+by+c, compute the cross point of line1 and line2
    if line1[0] != 0 and line1[0] == line2[0]:
        print('Cross point does not exist')
        return None
    if line1[0] == 0 and line2[0] == 0:
        print('Cross point does not exist')
        return None
    if line1[1] == 0:
        x = -line1[2]
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else:
        k1, _, b1 = line1
        k2, _, b2 = line2
        x = -(b1 - b2) / (k1 - k2)
        y = k1 * x + b1
    return np.array([x, y], dtype=np.float32)


def line_verticle(line, point):
    # get the verticle line from line across point
    if line[1] == 0:
        verticle = [0, -1, point[1]]
    else:
        if line[0] == 0:
            verticle = [1, 0, -point[0]]
        else:
            verticle = [-1. / line[0], -1, point[1] - (-1 / line[0] * point[0])]
    return verticle


def rectangle_from_parallelogram(poly):
    '''
    fit a rectangle from a parallelogram
    :param poly:
    :return:
    '''
    p0, p1, p2, p3 = poly
    angle_p0 = np.arccos(np.dot(p1 - p0, p3 - p0) / (np.linalg.norm(p0 - p1) * np.linalg.norm(p3 - p0)))
    if angle_p0 < 0.5 * np.pi:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
            # p0 and p2
            ## p0
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p0)

            new_p3 = line_cross_point(p2p3, p2p3_verticle)
            ## p2
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p2)

            new_p1 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
        else:
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p0)

            new_p1 = line_cross_point(p1p2, p1p2_verticle)
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p2)

            new_p3 = line_cross_point(p0p3, p0p3_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
    else:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
            # p1 and p3
            ## p1
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p1)

            new_p2 = line_cross_point(p2p3, p2p3_verticle)
            ## p3
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p3)

            new_p0 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
        else:
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p1)

            new_p0 = line_cross_point(p0p3, p0p3_verticle)
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p3)

            new_p2 = line_cross_point(p1p2, p1p2_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)


def sort_rectangle(poly):
    # sort the four coordinates of the polygon, points in poly should be sorted clockwise
    # First find the lowest point
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        # 底边平行于X轴, 那么p0为左上角 - if the bottom line is parallel to x-axis, then p0 must be the upper-left corner
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
    else:
        # 找到最低点右边的点 - find the point that sits right to the lowest point
        p_lowest_right = (p_lowest - 1) % 4
        p_lowest_left = (p_lowest + 1) % 4
        angle = np.arctan(
            -(poly[p_lowest][1] - poly[p_lowest_right][1]) / (poly[p_lowest][0] - poly[p_lowest_right][0]))
        # assert angle > 0
        if angle <= 0:
            print(angle, poly[p_lowest], poly[p_lowest_right])
        if angle / np.pi * 180 > 45:
            # 这个点为p2 - this point is p2
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi / 2 - angle)
        else:
            # 这个点为p3 - this point is p3
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], angle

def generate_rbox(im_size, polys, tags):
    h, w = im_size
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.uint8)
    geo_map = np.zeros((h, w, 5), dtype=np.float32)
    # mask used during traning, to ignore some hard areas
    training_mask = np.ones((h, w), dtype=np.uint8)
    for poly_idx, poly_tag in enumerate(zip(polys, tags)):
        poly = poly_tag[0]
        tag = poly_tag[1]

        r = [None, None, None, None]
        for i in range(4):
            r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                       np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
        # score map
        shrinked_poly = shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
        cv2.fillPoly(score_map, shrinked_poly, 1)
        cv2.fillPoly(poly_mask, shrinked_poly, poly_idx + 1)
        # if the poly is too small, then ignore it during training
        poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
        poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
        if min(poly_h, poly_w) < cfg["TRAIN"]["MIN_TEXT_SIZE"]:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
        if tag:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

        xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))
        # if geometry == 'RBOX':
        # 对任意两个顶点的组合生成一个平行四边形 - generate a parallelogram for any combination of two vertices
        fitted_parallelograms = []
        for i in range(4):
            p0 = poly[i]
            p1 = poly[(i + 1) % 4]
            p2 = poly[(i + 2) % 4]
            p3 = poly[(i + 3) % 4]
            edge = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            backward_edge = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            forward_edge = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            if point_dist_to_line(p0, p1, p2) > point_dist_to_line(p0, p1, p3):
                # 平行线经过p2 - parallel lines through p2
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p2[0]]
                else:
                    edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
            else:
                # 经过p3 - after p3
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p3[0]]
                else:
                    edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]
            # move forward edge
            new_p0 = p0
            new_p1 = p1
            new_p2 = p2
            new_p3 = p3
            new_p2 = line_cross_point(forward_edge, edge_opposite)
            if point_dist_to_line(p1, new_p2, p0) > point_dist_to_line(p1, new_p2, p3):
                # across p0
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p0[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
            else:
                # across p3
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p3[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]
            new_p0 = line_cross_point(forward_opposite, edge)
            new_p3 = line_cross_point(forward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
            # or move backward edge
            new_p0 = p0
            new_p1 = p1
            new_p2 = p2
            new_p3 = p3
            new_p3 = line_cross_point(backward_edge, edge_opposite)
            if point_dist_to_line(p0, p3, p1) > point_dist_to_line(p0, p3, p2):
                # across p1
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p1[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
            else:
                # across p2
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p2[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]
            new_p1 = line_cross_point(backward_opposite, edge)
            new_p2 = line_cross_point(backward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
        areas = [Polygon(t).area for t in fitted_parallelograms]
        parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype=np.float32)
        # sort thie polygon
        parallelogram_coord_sum = np.sum(parallelogram, axis=1)
        min_coord_idx = np.argmin(parallelogram_coord_sum)
        parallelogram = parallelogram[
            [min_coord_idx, (min_coord_idx + 1) % 4, (min_coord_idx + 2) % 4, (min_coord_idx + 3) % 4]]

        rectange = rectangle_from_parallelogram(parallelogram)
        rectange, rotate_angle = sort_rectangle(rectange)

        if not CYTHON:
            p0_rect, p1_rect, p2_rect, p3_rect = rectange
            for y, x in xy_in_poly:
                point = np.array([x, y], dtype=np.float32)
                # top
                geo_map[y, x, 0] = point_dist_to_line(p0_rect, p1_rect, point)
                # right
                geo_map[y, x, 1] = point_dist_to_line(p1_rect, p2_rect, point)
                # down
                geo_map[y, x, 2] = point_dist_to_line(p2_rect, p3_rect, point)
                # left
                geo_map[y, x, 3] = point_dist_to_line(p3_rect, p0_rect, point)
                # angle
                geo_map[y, x, 4] = rotate_angle
        else:
            gen_geo_map.gen_geo_map(geo_map, xy_in_poly, rectange, rotate_angle)

    return score_map, geo_map, training_mask

def resize_img(img, max_side_len=720):
    """
    将图像进行缩放,最大边大于2400,按照最大边进行resize,然后判断每个边是否能够被32整除,再进行一次resize
    :param im:
    :param max_side_len:
    :return:
    """
    h, w, _ = img.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resized_img = cv2.resize(img, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return resized_img, (ratio_h, ratio_w)

def mask_img(img, mask_data):
    for mask in mask_data:
        # print(mask)
        im = np.ones(img.shape[:2], dtype="uint8")
        mask = np.array(mask['bbox'], np.int32).reshape([1, 4, 2])
        cv2.polylines(im, mask, 1, 0)
        cv2.fillPoly(im, mask, 0)
        img[:, :, 0] = img[:, :, 0] * im
        img[:, :, 1] = img[:, :, 1] * im
        img[:, :, 2] = img[:, :, 2] * im
    return img

def generator(input_size=512, batch_size=32,
              background_ratio=1. / 8,
              random_scale=np.array(cfg.TRAIN.IMG_SCALE),
              vis=False,
              train_jsons=cfg.TRAIN.TRAIN_JSONS_LIST):
    json_data = []
    for json_file in train_jsons.keys():
        data = json.loads(open(os.path.join(json_file), encoding='utf-8').read(), encoding='bytes')
        np.random.shuffle(data)
        use = data[0: int(len(data) * train_jsons[json_file])]
        print('{} has {} data'.format(json_file, len(use)))
        json_data.extend(use)

    np.random.shuffle(json_data)
    print('{} training images'.format(len(json_data)))
    index = np.arange(0, len(json_data))
    epoch = -1
    while True:
        epoch += 1
        np.random.shuffle(index)
        images = []
        # image_fns = []
        score_maps = []
        geo_maps = []
        training_masks = []
        for i in index:
            try:
                data = json_data[i]

                im_fn = data['img_path']
                im = cv2.imread(im_fn)
                h, w, _ = im.shape
                txt_fn = data['label_path']
                if not os.path.exists(txt_fn):
                    print('text file {} does not exists'.format(txt_fn))
                    continue
                text_polys, text_tags, mask_data = load_annoataion(txt_fn, h, w)

                if text_polys is None:
                    continue

                if len(mask_data) != 0:
                    im = mask_img(im, mask_data)

                im, ratio = resize_img(im)
                ratio_h, ratio_w = ratio

                text_polys[:, :, 0] *= ratio_w
                text_polys[:, :, 1] *= ratio_h

                # 进行数据增强
                if random.random() < cfg.TRAIN.DATA_AUG_PROB:
                    im, text_polys = det_aug(im, text_polys)
                    h, w, _ = im.shape
                    text_polys = correct_box(text_polys, h, w)

                text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))
                rd_scale = np.random.choice(random_scale)
                im = cv2.resize(im, dsize=None,  fx=rd_scale, fy=rd_scale)
                text_polys *= rd_scale

                is_rotate = False
                if cfg.TRAIN.ROTATE180 is not None:
                    if random.random() < cfg.TRAIN.ROTATE180:
                        im = np.rot90(im)
                        im = np.rot90(im)
                        text_polys = np.array([])
                        text_tags = np.array([])
                        is_rotate = True

                # random crop a area from image
                if np.random.rand() < background_ratio:
                    # 没有将image进行resize,所以label不需要进行修正
                    # crop background
                    if cfg.TRAIN.IS_CROP:
                        im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=True)
                    if text_polys.shape[0] > 0:
                        # cannot find background
                        continue
                    # pad and resize image
                    new_h, new_w, _ = im.shape
                    max_h_w_i = np.max([new_h, new_w, input_size])
                    im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                    im_padded[:new_h, :new_w, :] = im.copy()
                    im = cv2.resize(im_padded, dsize=(input_size, input_size))
                    score_map = np.zeros((input_size, input_size), dtype=np.uint8)
                    geo_map_channels = 5 if cfg["GEOMETRY"] == 'RBOX' else 8
                    geo_map = np.zeros((input_size, input_size, geo_map_channels), dtype=np.float32)
                    training_mask = np.ones((input_size, input_size), dtype=np.uint8)
                else:
                    if is_rotate:
                        im = cv2.resize(im, dsize=(input_size, input_size))
                        score_map = np.zeros((input_size, input_size), dtype=np.uint8)
                        geo_map_channels = 5 if cfg["GEOMETRY"] == 'RBOX' else 8
                        geo_map = np.zeros((input_size, input_size, geo_map_channels), dtype=np.float32)
                        training_mask = np.ones((input_size, input_size), dtype=np.uint8)
                    else:
                        # 将image进行了resize,所以需要将label进行修正
                        if cfg.TRAIN.IS_CROP:
                            im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)
                        if text_polys.shape[0] == 0:
                            continue

                        # pad the image to the training input size or the longer side of image
                        new_h, new_w, _ = im.shape
                        max_h_w_i = np.max([new_h, new_w, input_size])
                        im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                        im_padded[:new_h, :new_w, :] = im.copy()
                        im = im_padded
                        # resize the image to input size
                        new_h, new_w, _ = im.shape
                        resize_h = input_size
                        resize_w = input_size
                        im = cv2.resize(im, dsize=(resize_w, resize_h))
                        resize_ratio_3_x = resize_w / float(new_w)
                        resize_ratio_3_y = resize_h / float(new_h)
                        text_polys[:, :, 0] *= resize_ratio_3_x
                        text_polys[:, :, 1] *= resize_ratio_3_y
                        new_h, new_w, _ = im.shape
                        score_map, geo_map, training_mask = generate_rbox((new_h, new_w), text_polys, text_tags)
                if vis:
                    save_path = ''
                    # vs_map = score_map.copy()
                    # vs_map[np.where(vs_map == 1)] = 255
                    # cv2.imwrite(os.path.join(save_path, img_name + '_score.jpg'), vs_map * training_mask[::, ::])
                    # cv2.imwrite(os.path.join('./test/geo/', img_name + '_g0.jpg'), geo_map[::, ::, 0])
                    # cv2.imwrite(os.path.join('./test/geo/', img_name + '_g1.jpg'), geo_map[::, ::, 1])
                    # cv2.imwrite(os.path.join('./test/geo/', img_name + '_g2.jpg'), geo_map[::, ::, 2])
                    # cv2.imwrite(os.path.join('./test/geo/', img_name + '_rbg.jpg'), geo_map[::, ::, 0:3])
                    # cv2.imwrite(os.path.join('./test/mask/', img_name + '_mask.jpg'), training_mask[::, ::] * 255)

                    # fig, axs = plt.subplots(3, 2, figsize=(20, 30))
                    # axs[0, 0].imshow(im[:, :, ::-1])
                    # axs[0, 0].set_xticks([])
                    # axs[0, 0].set_yticks([])
                    # for poly in text_polys:
                    #     poly_h = min(abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
                    #     poly_w = min(abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
                    #     axs[0, 0].add_artist(Patches.Polygon(
                    #         poly, facecolor='none', edgecolor='green', linewidth=2, linestyle='-', fill=True))
                    #     axs[0, 0].text(poly[0, 0], poly[0, 1], '{:.0f}-{:.0f}'.format(poly_h, poly_w), color='purple')
                    # axs[0, 1].imshow(score_map[::, ::])
                    # axs[0, 1].set_xticks([])
                    # axs[0, 1].set_yticks([])
                    # axs[1, 0].imshow(geo_map[::, ::, 0])
                    # axs[1, 0].set_xticks([])
                    # axs[1, 0].set_yticks([])
                    # axs[1, 1].imshow(geo_map[::, ::, 1])
                    # axs[1, 1].set_xticks([])
                    # axs[1, 1].set_yticks([])
                    # axs[2, 0].imshow(geo_map[::, ::, 2])
                    # axs[2, 0].set_xticks([])
                    # axs[2, 0].set_yticks([])
                    # axs[2, 1].imshow(training_mask[::, ::])
                    # axs[2, 1].set_xticks([])
                    # axs[2, 1].set_yticks([])
                    # plt.tight_layout()
                    # plt.show()
                    # plt.close()

                images.append(im[:, :, ::-1].astype(np.float32))
                # image_fns.append(im_fn)
                score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
                geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
                training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))

                if len(images) == batch_size:
                    yield images, epoch, score_maps, geo_maps, training_masks
                    images = []
                    # image_fns = []
                    score_maps = []
                    geo_maps = []
                    training_masks = []
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(data['img_path'])
                continue


def get_batch(num_workers, **kwargs):
    try:
        # 利用多线程 获取数据
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        print('Generator use 10 batches for buffering, this may take a while, you can tune this yourself.')
        enqueuer.start(max_queue_size=10, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


if __name__ == '__main__':
    data_generator = get_batch(num_workers=cfg["TRAIN"]["NUM_READERS"],
                               input_size=cfg['TRAIN']["INPUT_SIZE"],
                               batch_size=64,
                               train_img_dir='./test/img/',
                               train_label_dir='./test/label/')
    data = next(data_generator)

    a = time.time()
    for i in range(100):
        s = time.time()
        data = next(data_generator)
        print(time.time()-s)
        time.sleep(0.7)
    print('b=', (time.time()-a)/100)
    # 7.078546857833862
    # 0.8469973683357239

    #
    # 1.64
    # 16.86
