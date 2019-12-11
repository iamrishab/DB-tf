# -*- coding:utf-8 -*-
import os
import json
import cv2
import random
import numpy as np
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon


from east_config import cfg


def det_aug(image, polys_np):
    """
    随机对图像做以下的增强操作
    :param image: cv2 read
    :param polys_np:[N, 4, 2]
    :return:
    """
    aug_sample = random.sample(cfg.TRAIN.AUG_TOOL, 1)[0]  #从数组中随机取出一个增强的功能

    ######################################################################################################
    # blur-模糊
    aug = None
    # 高斯滤波 sigma 为1-10的保留小数点后一位的float的随机值,可根据情况调整
    if aug_sample == 'GaussianBlur':
        sigma = random.uniform(1, 2)
        sigma = round(8, 10)
        aug = iaa.GaussianBlur(sigma)

    # 平均模糊 k 为1-10的随机 奇 数,范围根据情况调整
    if aug_sample == 'AverageBlur':
        k = random.randint(8, 10) * 2 + 1
        aug = iaa.AverageBlur(k)

    # 中值滤波 k 为1-10的随机 奇 数,范围根据情况调整
    if aug_sample == 'MedianBlur':
        k = random.randint(8, 10) * 2 + 1
        aug = iaa.MedianBlur(k)

    # 双边滤波 d=1 为 奇 数, sigma_color=(10, 250), sigma_space=(10, 250)
    if aug_sample == 'BilateralBlur':
        d = random.randint(0, 2) * 2 + 1
        sigma_color = random.randint(10, 250)
        sigma_space = random.randint(10, 250)
        aug = iaa.BilateralBlur(d, sigma_color, sigma_space)

    # 运动模糊 k=5 一定大于3 的 奇 数, angle=(0, 360), direction=(-1.0, 1.0)
    if aug_sample == 'MotionBlur':
        k = random.randint(15, 20) * 2 + 1
        angle = random.randint(0, 360)
        direction = random.uniform(-1, 1)
        direction = round(direction, 1)
        aug = iaa.MotionBlur(k, angle, direction)

    ######################################################################################################
    # geometric  几何学

    # 弹性变换
    if aug_sample == 'ElasticTransformation':
        alpha = random.uniform(10, 20)
        alpha = round(alpha, 1)
        sigma = random.uniform(5, 10)
        sigma = round(sigma, 1)
        # print(alpha, sigma)
        aug = iaa.ElasticTransformation(alpha, sigma)

    # 透视
    if aug_sample == 'PerspectiveTransform':
        scale = random.uniform(0, 0.2)
        scale = round(scale, 3)
        aug = iaa.PerspectiveTransform(scale)

    # 旋转角度
    # if aug_sample == 'Affine_rot':
    #     rotate = random.randint(-20, 20)
    #     aug = iaa.Affine(rotate=rotate)

    # 缩放
    # if aug_sample == 'Affine_scale':
    #     scale = random.uniform(0, 2)
    #     scale = round(scale, 1)
    #     aug = iaa.Affine(scale=scale)
    ######################################################################################################
    # flip 镜像

    # 水平镜像
    # if aug_sample == 'Fliplr':
    #     aug = iaa.Fliplr(1)
    #
    # 垂直镜像
    # if aug_sample == 'Flipud':
    #     aug = iaa.Flipud(1)

    ######################################################################################################
    # size 尺寸

    # if aug_sample == 'CropAndPad':
    #     top = random.randint(0, 10)
    #     right = random.randint(0, 10)
    #     bottom = random.randint(0, 10)
    #     left = random.randint(0, 10)
    #     aug = iaa.CropAndPad(px=(top, right, bottom, left))  # 上 右 下 左 各crop多少像素,然后进行padding

    if aug_sample == 'Crop':
        top = random.randint(0, 10)
        right = random.randint(0, 10)
        bottom = random.randint(0, 10)
        left = random.randint(0, 10)
        aug = iaa.Crop(px=(top, right, bottom, left))  # 上 右 下 左

    if aug_sample == 'Pad':
        top = random.randint(0, 10)
        right = random.randint(0, 10)
        bottom = random.randint(0, 10)
        left = random.randint(0, 10)
        aug = iaa.Pad(px=(top, right, bottom, left))  # 上 右 下 左

    # if aug_sample == 'PadToFixedSize':
    #     height = image.shape[0] + 32
    #     width = image.shape[1] + 100
    #     aug = iaa.PadToFixedSize(width=width, height=height)z

    # if aug_sample == 'CropToFixedSize':
    #     height = image.shape[0] - 32
    #     width = image.shape[1] - 100
    #     aug = iaa.CropToFixedSize(width=width, height=height)
    if aug is not None:
        # print(aug_sample)
        h, w, _ = image.shape
        boxes_info_list = []
        for box in polys_np:
            boxes_info_list.append(Polygon(box))

        psoi = ia.PolygonsOnImage(boxes_info_list, shape=image.shape)  # 生成单个图像上所有多边形的对象
        image, psoi_aug = aug(image=image, polygons=psoi)

        pts_list = []
        for each_poly in psoi_aug.polygons:
            pts_list.append(np.array(each_poly.exterior).reshape((4, 2)))
        return image, np.array(pts_list, np.float32).reshape((-1, 4, 2))
    else:

        return image, polys_np
