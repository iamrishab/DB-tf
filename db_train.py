# coding=utf-8
import time
import numpy as np
import logging
import os
import tensorflow as tf
from tensorflow.contrib import slim

from east_config import cfg

import lib.network.model as model
import lib.dataset.dataload as dataload

import warnings
warnings.filterwarnings("ignore")


gpus = cfg.TRAIN.GPU_LIST.split(',')


def tower_loss(images, score_maps, geo_maps, training_masks, reuse_variables=None):
    # Build inference graph
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        f_score, f_geometry = model.model(images, is_training=True)

    model_loss = model.loss(score_maps, f_score, geo_maps, f_geometry, training_masks)

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


def main():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.TRAIN.VIS_GPU
    if not tf.gfile.Exists(cfg["TRAIN"]["CHECKPOINTS_OUTPUT_DIR"]):
        tf.gfile.MkDir(cfg["TRAIN"]["CHECKPOINTS_OUTPUT_DIR"])

    train_logger = _train_logger_init()

    # resize后的图像
    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    # resize后的score map
    input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_score_maps')

    if cfg["GEOMETRY"] == 'RBOX':
        # 每个bbox
        input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='input_geo_maps')
    else:
        # 不支持任意四边形的train
        input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 8], name='input_geo_maps')

    # mask后的图像
    input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_masks')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    # 实现指数衰减学习步骤：1.首先使用较大学习率(目的：为快速得到一个比较优的解);2.然后通过迭代逐步减小学习率(目的：为使模型在训练后期更加稳定);
    learning_rate = tf.train.exponential_decay(cfg["TRAIN"]["LEARNING_RATE"], global_step, decay_steps=10000,
                                               decay_rate=0.94, staircase=True)

    # Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。
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

    # 切分数据
    input_images_split = tf.split(input_images, len(gpus))
    input_score_maps_split = tf.split(input_score_maps, len(gpus))
    input_geo_maps_split = tf.split(input_geo_maps, len(gpus))
    input_training_masks_split = tf.split(input_training_masks, len(gpus))

    # 存储每个GPU计算的梯度
    tower_grads = []
    reuse_variables = None
    for i, gpu_id in enumerate(gpus):
        # 给每个GPU构图并训练对应的batch,得到每个GPU计算的梯度
        print('gpu_id', gpu_id)
        with tf.device('/gpu:' + gpu_id):
            with tf.name_scope('model_' + gpu_id) as scope:
                iis = input_images_split[i]
                isms = input_score_maps_split[i]
                igms = input_geo_maps_split[i]
                itms = input_training_masks_split[i]
                total_loss, model_loss = tower_loss(iis, isms, igms, itms, reuse_variables)
                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                reuse_variables = True

                grads = opt.compute_gradients(total_loss)
                tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(cfg["TRAIN"]["MOVING_AVERAGE_DECAY"], global_step)

    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # batch norm updates
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=cfg.TRAIN.SAVE_MAX)

    if not tf.gfile.Exists(cfg["TRAIN"]["TRAIN_LOGS"]):
        tf.gfile.MkDir(cfg["TRAIN"]["TRAIN_LOGS"])
    summary_writer = tf.summary.FileWriter(cfg["TRAIN"]["TRAIN_LOGS"], tf.get_default_graph())

    init = tf.global_variables_initializer()
    #
    # if os.path.exists(cfg["TRAIN"]["PRETRAINED_MODEL_DIR"]):
    #     variable_restore_op = slim.assign_from_checkpoint_fn(cfg["TRAIN"]["PRETRAINED_MODEL_DIR"],
    #                                                          slim.get_trainable_variables(),
    #                                                          ignore_missing_vars=True)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        try:

            if cfg["TRAIN"]["RESTORE"]:
                # 加载上次保存的模型
                train_logger.info('continue training from previous checkpoint')
                ckpt = tf.train.get_checkpoint_state(cfg["TRAIN"]["RESTORE_CKPT_PATH"])
                train_logger.info('restore model path:', ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                train_logger.info("done")
            elif cfg["TRAIN"]["PRETRAINED_MODEL_PATH"] is not None:
                # 加载预训练模型
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

        data_generator = dataload.get_batch(num_workers=cfg["TRAIN"]["NUM_READERS"],
                                            input_size=cfg['TRAIN']["INPUT_SIZE"],
                                            batch_size=cfg["TRAIN"]["BATCH_SIZE_PER_GPU"] * len(gpus))

        start = time.time()
        for step in range(cfg["TRAIN"]["MAX_STEPS"]):
            data = next(data_generator)
            ml, tl, _ = sess.run([model_loss, total_loss, train_op], feed_dict={input_images: data[0],
                                                                                input_score_maps: data[2],
                                                                                input_geo_maps: data[3],
                                                                                input_training_masks: data[4]})
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
                saver.save(sess, os.path.join(cfg["TRAIN"]["CHECKPOINTS_OUTPUT_DIR"], cfg.TRAIN.VERSION + 'east_model.ckpt'), global_step=global_step)

            if step % cfg["TRAIN"]["SAVE_SUMMARY_STEPS"] == 0:
                _, tl, summary_str = sess.run([train_op, total_loss, summary_op], feed_dict={input_images: data[0],
                                                                                             input_score_maps: data[2],
                                                                                             input_geo_maps: data[3],
                                                                                             input_training_masks: data[
                                                                                                 4]})
                summary_writer.add_summary(summary_str, global_step=step)


if __name__ == '__main__':

    main()
