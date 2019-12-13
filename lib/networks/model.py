# coding:utf-8
import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
from east_config import cfg

import lib.network.resnet_v1 as resnet_v1


def unpool(inputs):
    # 使用双线性插值调整
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2])


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    图像归一化
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def model(images, weight_decay=1e-5, is_training=True):
    """
    构建EAST网络模型,使用resnet-50
    :param images:
    :param weight_decay:
    :param is_training:
    :return:
    """

    # 图像归一化,使用像素均值
    images = mean_image_subtraction(images)

    # 通过这个函数将不想重复写的参数通过这个函数自动赋值
    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')

    # 管理传给get_variable()的变量名称的作用域
    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {'decay': cfg["TRAIN"]["MOVING_AVERAGE_DECAY"],
                             'epsilon': 1e-5,
                             'scale': True,
                             'is_training': is_training}

        # FPN层,合并多层卷积
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):

            # # 提取四个级别的特征图
            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2']]

            # ASPP
            if cfg.IS_ASPP:
                with tf.variable_scope('ASPP_branch'):
                    aspp_conv = []
                    aspp_rate = [3, 6, 9, 12]
                    for i in range(4):
                        aspp_c = slim.conv2d(f[0], 256, 3, rate=aspp_rate[i], )
                        aspp_conv.append(aspp_c)
                    f[0] = slim.conv2d(tf.concat(aspp_conv, axis=-1), 256 * len(aspp_rate), 3)
                    f[0] = slim.conv2d(f[0], 2048, 3)
                    #f[0] = slim.conv2d(f[0], 256, 3)

            # 基础特征图
            g = [None, None, None, None]
            # 合并后特征图
            h = [None, None, None, None]

            # 合并每一步的通道数
            num_outputs = [None, 128, 64, 32]

            # 自下而上合并特征图
            for i in range(len(f)):
                if i == 0:
                    h[i] = f[i]  # 计算h
                else:
                    # 先合并特征图，再进行1*1卷积 3*3卷积
                    c1_1 = slim.conv2d(tf.concat([g[i - 1], f[i]], axis=-1), num_outputs[i], 1)
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                if i <= 2:
                    # 上池化特征图
                    g[i] = unpool(h[i])  # 计算g
                else:
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                # print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

            # 输出层
            # 获得预测分数的特征图，与原图尺寸一致，每一个值代表此处是否有文字的可能性
            # 使用sigmod来完成得分和角度的数值回归
            F_score = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)

            # 4 channel of axis aligned bbox and 1 channel rotation angle
            # 获得旋转框像素偏移的特征图，有四个通道，分别代表每个像素点到文本矩形框上，右，底，左边界的距离
            geo_map = slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * cfg.TRAIN.INPUT_SIZE

            # 获得旋转框角度特征图 angle is between [-45, 45]
            angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi / 2

            # 按深度连接旋转框特征图和角度特征图
            F_geometry = tf.concat([geo_map, angle_map], axis=-1)

    return F_score, F_geometry


# 计算score_map的loss
def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('classification_dice_loss', loss)
    return loss


# 几何误差函数,总的loss
def loss(y_true_cls, y_pred_cls,
         y_true_geo, y_pred_geo,
         training_mask):
    '''
    define the loss used for training, contraning two part,
    the first part we use dice loss instead of weighted logloss,
    the second part is the iou loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    '''
    classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    # scale classification loss to match the iou loss part
    classification_loss *= 0.01

    # d1 -> top, d2->right, d3->bottom, d4->left
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)

    # 计算真实旋转框、预测旋转框面积
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)

    # 计算相交部分的高度和宽度  面积
    w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)

    # 计算R_true与R_pred的交集
    area_intersect = w_union * h_union

    # 计算R_true与R_pred的并集
    area_union = area_gt + area_pred - area_intersect

    # IoU loss,加1为了防止交集为0，log0没意义
    L_AABB = -tf.log((area_intersect + 1.0) / (area_union + 1.0))

    # 夹角的loss
    L_theta = 1 - tf.cos(theta_pred - theta_gt)
    # tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
    # tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))

    valid_pts_nums = tf.cast(tf.count_nonzero(y_true_cls * training_mask), dtype=tf.float32)
    tf.summary.scalar('geometry_AABB', tf.reduce_sum(L_AABB * y_true_cls * training_mask) / valid_pts_nums)
    tf.summary.scalar('geometry_theta', tf.reduce_sum(L_theta * y_true_cls * training_mask) / valid_pts_nums)

    L_g = L_AABB + 20 * L_theta  # geometry_map loss

    return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss
