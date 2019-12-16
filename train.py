# coding=utf-8
import time
import numpy as np
import logging
import os
import tensorflow as tf
from tensorflow.contrib import slim

from db_config import cfg

import lib.networks.model as model
from lib.networks.losses import compute_loss
import lib.dataset.dataload as dataload

import warnings
warnings.filterwarnings("ignore")


gpus = cfg.TRAIN.GPU_LIST.split(',')




def tower_loss(images, gt_score_maps, gt_threshold_map, gt_score_mask,
               gt_thresh_mask, reuse_variables=None):

    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        binarize_map, threshold_map, thresh_binary = model.model(images, is_training=True)

    model_loss = compute_loss(binarize_map, threshold_map, thresh_binary,
                              gt_score_maps, gt_threshold_map, gt_score_mask, gt_thresh_mask)

    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # add summary
    if reuse_variables is None:
        tf.summary.image('input', images)
        tf.summary.image('score_map', score_maps)
        tf.summary.image('score_map_pred', f_score * 255)
        tf.summary.image('geo_map_0', geo_maps[:, :, :, 0:1])
        tf.summary.image('geo_map_0_pred', f_geometry[:, :, :, 0:1])
        tf.summary.image('training_masks', training_masks)
        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)

    return total_loss, model_loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def _train_logger_init():
    """
    初始化log日志
    :return:
    """
    train_logger = logging.getLogger('train')
    train_logger.setLevel(logging.DEBUG)

    # 添加文件输出
    log_file = cfg["TRAIN"]["TRAIN_LOGS"] + time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '.logs'
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    file_handler.setFormatter(file_formatter)
    train_logger.addHandler(file_handler)

    # 添加控制台输出
    consol_handler = logging.StreamHandler()
    consol_handler.setLevel(logging.DEBUG)
    consol_formatter = logging.Formatter('%(message)s')
    consol_handler.setFormatter(consol_formatter)
    train_logger.addHandler(consol_handler)
    return train_logger


