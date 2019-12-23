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
from lib.dataset.dataloader import get_batch

import warnings
warnings.filterwarnings("ignore")

def tower_loss(images, gt_score_maps, gt_threshold_map, gt_score_mask,
               gt_thresh_mask, reuse_variables):

    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        binarize_map, threshold_map, thresh_binary = model.model(images, is_training=True)

    model_loss = compute_loss(binarize_map, threshold_map, thresh_binary,
                              gt_score_maps, gt_threshold_map, gt_score_mask, gt_thresh_mask)

    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # add summary
    if reuse_variables is None:
        tf.summary.image('gt/input_imgs', images)
        tf.summary.image('gt/score_map', gt_score_maps)
        tf.summary.image('gt/threshold_map', gt_threshold_map * 255)
        tf.summary.image('gt/score_mask', gt_score_mask)
        tf.summary.image('gt/thresh_mask', gt_thresh_mask)

        tf.summary.image('pred/binarize_map', binarize_map)
        tf.summary.image('pred/threshold_map', threshold_map * 255)
        tf.summary.image('pred/thresh_binary', thresh_binary)

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
    log_file = os.path.join(cfg["TRAIN"]["TRAIN_LOGS"], time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '.logs')
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


def main():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.TRAIN.VIS_GPU
    if not tf.gfile.Exists(cfg["TRAIN"]["CHECKPOINTS_OUTPUT_DIR"]):
        tf.gfile.MkDir(cfg["TRAIN"]["CHECKPOINTS_OUTPUT_DIR"])

    train_logger = _train_logger_init()

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_score_maps')
    input_threshold_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_threshold_maps')

    input_score_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_score_masks')
    input_threshold_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_threshold_masks')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    learning_rate = tf.train.exponential_decay(cfg["TRAIN"]["LEARNING_RATE"], global_step, decay_steps=10000,
                                               decay_rate=0.94, staircase=True)

    if cfg.TRAIN.OPT == 'adam':
        # learning_rate = tf.constant(cfg["TRAIN"]["LEARNING_RATE"], tf.float32)
        opt = tf.train.AdamOptimizer(learning_rate)
    elif cfg.TRAIN.OPT == 'momentum':
        opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
    else:
        assert 0, 'error optimzer'
    print('use ', cfg.TRAIN.OPT)

    # add summary
    tf.summary.scalar('learning_rate', learning_rate)

    gpus = [str(i) for i in range(len(cfg.TRAIN.VIS_GPU.split(',')))]
    input_images_split = tf.split(input_images, len(gpus))
    input_score_maps_split = tf.split(input_score_maps, len(gpus))
    input_threshold_maps_split = tf.split(input_threshold_maps, len(gpus))
    input_score_masks_split = tf.split(input_score_masks, len(gpus))
    input_threshold_masks_split = tf.split(input_threshold_masks, len(gpus))


    tower_grads = []
    reuse_variables = None
    for i, gpu_id in enumerate(gpus):
        print('gpu_id', gpu_id)
        with tf.device('/gpu:' + gpu_id):
            with tf.name_scope('model_' + gpu_id) as scope:
                gt_imgs = input_images_split[i]
                gt_scores = input_score_maps_split[i]
                gt_thresholds = input_threshold_maps_split[i]
                gt_score_masks = input_score_masks_split[i]
                gt_threshold_masks = input_threshold_masks_split[i]
                total_loss, model_loss = tower_loss(gt_imgs, gt_scores, gt_thresholds, gt_score_masks, gt_threshold_masks, reuse_variables)
                reuse_variables = True

                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))

                grads = opt.compute_gradients(total_loss)
                tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()

    variable_averages = tf.train.ExponentialMovingAverage(cfg["TRAIN"]["MOVING_AVERAGE_DECAY"], global_step)

    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=cfg.TRAIN.SAVE_MAX)

    if not tf.gfile.Exists(cfg["TRAIN"]["TRAIN_LOGS"]):
        tf.gfile.MkDir(cfg["TRAIN"]["TRAIN_LOGS"])
    summary_writer = tf.summary.FileWriter(cfg["TRAIN"]["TRAIN_LOGS"], tf.get_default_graph())

    init = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        try:

            if cfg["TRAIN"]["RESTORE"]:
                train_logger.info('continue training from previous checkpoint')
                ckpt = tf.train.get_checkpoint_state(cfg["TRAIN"]["RESTORE_CKPT_PATH"])
                train_logger.info('restore model path:', ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                train_logger.info("done")
            elif cfg["TRAIN"]["PRETRAINED_MODEL_PATH"] is not None:
                sess.run(init)
                print(cfg["TRAIN"]["PRETRAINED_MODEL_PATH"])
                train_logger.info('load pretrain model:{}', str(cfg["TRAIN"]["PRETRAINED_MODEL_PATH"]))
                variable_restore_op = slim.assign_from_checkpoint_fn(cfg["TRAIN"]["PRETRAINED_MODEL_PATH"],
                                                                     slim.get_trainable_variables(),
                                                                     ignore_missing_vars=True)
                variable_restore_op(sess)
                train_logger.info("done")

            else:
                sess.run(init)
        except:
            assert 0, 'load error'

        train_data_generator = get_batch(num_workers=cfg.TRAIN.NUM_READERS,
                                   img_dir=cfg.TRAIN.IMG_DIR,
                                   label_dir=cfg.TRAIN.LABEL_DIR,
                                   batchsize=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(gpus))

        start = time.time()
        for step in range(cfg["TRAIN"]["MAX_STEPS"]):
            train_data = next(train_data_generator)

            feed_dict = {input_images: train_data[0],
                         input_score_maps: train_data[1],
                         input_threshold_maps: train_data[3],
                         input_score_masks: train_data[2],
                         input_threshold_masks: train_data[4]}

            ml, tl, _ = sess.run([model_loss, total_loss, train_op], feed_dict=feed_dict)
            if np.isnan(tl):
                train_logger.info('Loss diverged, stop training')
                break

            if step % 10 == 0:
                avg_time_per_step = (time.time() - start) / 10
                avg_examples_per_second = (10 * cfg["TRAIN"]["BATCH_SIZE_PER_GPU"] * len(gpus)) / (time.time() - start)
                start = time.time()
                train_logger.info(
                    '{}->Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, {:.2f} examples/second'.format(
                        cfg.TRAIN.VERSION, step, ml, tl, avg_time_per_step, avg_examples_per_second))

            if step % cfg["TRAIN"]["SAVE_CHECKPOINT_STEPS"] == 0:
                saver.save(sess, os.path.join(cfg["TRAIN"]["CHECKPOINTS_OUTPUT_DIR"],
                                              'DB_' + cfg.BACKBONE + '_' + cfg.TRAIN.VERSION + '_model.ckpt'),
                           global_step=global_step)

            if step % cfg["TRAIN"]["SAVE_SUMMARY_STEPS"] == 0:
                _, tl, summary_str = sess.run([train_op, total_loss, summary_op], feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step=step)


if __name__ == '__main__':

    main()

