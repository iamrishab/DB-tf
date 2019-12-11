# -*- coding:utf-8 -*-
"""
用于检测模型的数据增广，包括：
1、图片旋转，-15度~15度
2、图片剪裁，1/2、1/4、1/8、1/16
3、图片缩放

"""

import os
import cv2
import numpy as np
import random
import tqdm


def crop_image(img, boxes, threshold_num=3):
    """
    对图片剪裁，最多切为4份。
    :param img: 读取出的图片（RGB）
    :param boxes: 检测框标签，shape = [N, 8], N为检测框个数，8为4点坐标值。
    :param threshold_num: 截取每张小图中最小的框的数目
    :return: 返回切割的图片和每个切割后的图片框的label
    """

    def find_crop_dot(line):
        """
        找到切割点
        :param line:
        :return:
        """
        dots_index = np.where(line==0)
        if len(dots_index[0]) == 0:
            return [0]
        midle = int(len(line)/2)
        dots_index = dots_index[0]
        distance = abs(dots_index-midle)
        dot = dots_index[np.where(distance == min(distance))]

        return dot[0]

    def tune_img_labels(img, bboxes, point, type):

        if type == 1:
            cut_img = img[0:point[0] + 1, 0:point[1] + 1, :]
            res_bboxes = bboxes
        elif type == 2:
            cut_img = img[0:point[0], point[1]:, :]
            res_bboxes = []
            for bbox in bboxes:
                bbox[0] -= point[0]
                bbox[2] -= point[0]
                bbox[4] -= point[0]
                bbox[6] -= point[0]
                res_bboxes.append(bbox)
        elif type == 3:
            cut_img = img[point[0]:, 0:point[1] + 1, :]
            res_bboxes = []
            for bbox in bboxes:
                bbox[1] -= point[1]
                bbox[3] -= point[1]
                bbox[5] -= point[1]
                bbox[7] -= point[1]
                res_bboxes.append(bbox)
        else:
            cut_img = img[point[0]:, point[1]:, :]
            res_bboxes = []
            for bbox in bboxes:
                bbox[1] -= point[1]
                bbox[3] -= point[1]
                bbox[5] -= point[1]
                bbox[7] -= point[1]
                bbox[0] -= point[0]
                bbox[2] -= point[0]
                bbox[4] -= point[0]
                bbox[6] -= point[0]
                res_bboxes.append(bbox)
        return cut_img, res_bboxes

    h, w, _ = img.shape

    y_flag = np.zeros([h])
    x_flag = np.zeros([w])

    for box in boxes:
        min_x = min(box[0], box[2], box[4], box[6])
        max_x = max(box[0], box[2], box[4], box[6])

        min_y = min(box[1], box[3], box[5], box[7])
        max_y = max(box[1], box[3], box[5], box[7])

        x_flag[min_x: max_x+1] = 1
        y_flag[min_y: max_y+1] = 1

    crop_x_dot = find_crop_dot(x_flag)
    crop_y_dot = find_crop_dot(y_flag)

    bboxes_top_l = []
    bboxes_top_r = []
    bboxes_bot_l = []
    bboxes_bot_r = []

    # 将box进行分类
    for box in boxes:
        if box[1] < crop_y_dot:
            if box[0] < crop_x_dot:
                bboxes_top_l.append(box)
            else:
                bboxes_top_r.append(box)
        else:
            if box[0] < crop_x_dot:
                bboxes_bot_l.append(box)
            else:
                bboxes_bot_r.append(box)

    point = [crop_x_dot, crop_y_dot]
    crop_dataset_list = []

    if len(bboxes_top_l) > threshold_num:
        data_dict = {}
        cut_img, res_bboxes = tune_img_labels(img.copy(), bboxes_top_l, point, 1)
        data_dict["img"] = cut_img
        data_dict["bbox"] = res_bboxes
        crop_dataset_list.append(data_dict)

    if len(bboxes_top_r) > threshold_num:
        data_dict = {}
        cut_img, res_bboxes = tune_img_labels(img.copy(), bboxes_top_r, point, 2)
        data_dict["img"] = cut_img
        data_dict["bbox"] = res_bboxes
        crop_dataset_list.append(data_dict)

    if len(bboxes_bot_l) > threshold_num:
        data_dict = {}
        cut_img, res_bboxes = tune_img_labels(img.copy(), bboxes_bot_l, point, 3)
        data_dict["img"] = cut_img
        data_dict["bbox"] = res_bboxes
        crop_dataset_list.append(data_dict)

    if len(bboxes_bot_r) > threshold_num:
        data_dict = {}
        cut_img, res_bboxes = tune_img_labels(img.copy(), bboxes_bot_r, point, 4)
        data_dict["img"] = cut_img
        data_dict["bbox"] = res_bboxes
        crop_dataset_list.append(data_dict)

    return crop_dataset_list


def rotate_image(img, boxes, rotate):
    """
    根据旋转角度，旋转图片和对应的boxes
    :param img: [h,w,3]
    :param boxes: [n, 4, 2]
    :param rotate: -15~15度
    :return:
    """

    pass


if __name__ == "__main__":
    test_img_dir = './test/img'
    test_label_dir = './test/label'

    img_path_list = os.listdir(test_img_dir)
    label_path_list = os.listdir(test_label_dir)

    for img_path in img_path_list:
        img = cv2.imread()


