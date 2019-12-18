# coding:utf-8
import tensorflow as tf
from tensorflow.contrib import slim
from db_config import cfg

import lib.networks.resnet.resnet_v1 as resnet_v1
import lib.networks.resnet.resnet_v1_tiny as resnet_v1_tiny


def unpool(inputs, ratio=2):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * ratio, tf.shape(inputs)[2] * ratio])


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
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

def backbone(input, weight_decay, is_training, backbone_name=cfg.BACKBONE):
    # ['resnet_v1_50', 'resnet_v1_18', 'resnet_v2_50', 'resnet_v2_18', 'mobilenet_v2', 'mobilenet_v3']

    if backbone_name == 'resnet_v1_50':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v1.resnet_v1_50(input, is_training=is_training, scope=backbone_name)
        return logits, end_points
    elif backbone_name == 'resnet_v1_18':
        with slim.arg_scope(resnet_v1_tiny.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v1_tiny.resnet_v1_18(input, is_training=is_training, scope=backbone_name)
        return logits, end_points
    else:
        print('{} is error backbone name, not support!'.format(backbone_name))
        assert 0


def model(images, weight_decay=1e-5, is_training=True):
    """
    resnet-50
    :param images:
    :param weight_decay:
    :param is_training:
    :return:
    """

    images = mean_image_subtraction(images)

    logits, end_points = backbone(images, weight_decay, is_training)

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {'decay': cfg["TRAIN"]["MOVING_AVERAGE_DECAY"],
                             'epsilon': 1e-5,
                             'scale': True,
                             'is_training': is_training}

        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):

            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2']]

            g = [None, None, None, None]
            h = [None, None, None, None]

            num_outputs = [None, 128, 64, 32]

            for i in range(len(f)):
                if i == 0:
                    h[i] = f[i]
                else:
                    c1_1 = slim.conv2d(tf.concat([g[i - 1], f[i]], axis=-1), num_outputs[i], 1)
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                if i <= 2:
                    g[i] = unpool(h[i])
                else:
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)

            with tf.variable_scope('concat_branch'):
                features = [g[3], h[2], h[1], h[0]]

                concat_feature = None

                for i, f in enumerate(features):
                    if i is 0:
                        conv_f = slim.conv2d(f, 64, 3)
                        concat_feature = conv_f
                    else:
                        up_f = slim.conv2d(f, 64, 3)
                        up_f = unpool(up_f, 2**i)
                        concat_feature = tf.concat([concat_feature, up_f], axis=-1)

                final_f = slim.conv2d(concat_feature, 64, 3)

            with tf.variable_scope('binarize_branch'):
                b_conv = slim.conv2d(final_f, 64, 3)
                b_conv = slim.conv2d_transpose(b_conv, 64, 2, 2)
                binarize_map = slim.conv2d_transpose(b_conv, 1, 2, 2, activation_fn=tf.nn.sigmoid)

            with tf.variable_scope('threshold_branch'):
                b_conv = slim.conv2d(final_f, 64, 3)
                b_conv = slim.conv2d_transpose(b_conv, 256, 2, 2)
                threshold_map = slim.conv2d_transpose(b_conv, 1, 2, 2, activation_fn=tf.nn.sigmoid)

            with tf.variable_scope('thresh_binary_branch'):
                thresh_binary = tf.reciprocal(1 + tf.exp(-cfg.K * (binarize_map-threshold_map)), name='thresh_binary')

    return binarize_map, threshold_map, thresh_binary

