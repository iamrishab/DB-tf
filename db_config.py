import os
from easydict import EasyDict as edict

cfg = edict()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~inference~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cfg.MEANS = [123.68, 116.78, 103.94]
cfg.INPUT_MAX_SIZE = 720
cfg.K = 10
cfg.SHRINK_RATIO = 0.4
cfg.THRESH_MIN = 10
cfg.THRESH_MAX = 10
cfg.MIN_TEXT_SIZE = 10

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~train config~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cfg.TRAIN = edict()
cfg.TRAIN.VERSION = '' + '_'
# 多gpu训练
cfg.TRAIN.GPU_LIST = '0,1'
cfg.TRAIN.VIS_GPU = '1,2'
# batchsize 大小
cfg.TRAIN.BATCH_SIZE_PER_GPU = 12

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~dataload & aug~~~~~~~~~~~~~~~~~~~~~~~~~~
# 图像缩放
cfg.TRAIN.IMG_SCALE = [0.5, 1, 1, 1, 1.5, 2.0]
# crop
cfg.TRAIN.IS_CROP = True
# 线程数
cfg.TRAIN.NUM_READERS = 20
# 数据增强概率
cfg.TRAIN.DATA_AUG_PROB = 0.0
# 数据增强方式
cfg.TRAIN.AUG_TOOL = ['GaussianBlur',
                'AverageBlur',
                'MedianBlur',
                'BilateralBlur',
                'MotionBlur',
                #'ElasticTransformation',
                #'PerspectiveTransform',
                      ]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~save ckpt and log~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 训练次数
cfg.TRAIN.MAX_STEPS = 10000000
# 保存ckpt的步数
cfg.TRAIN.SAVE_CHECKPOINT_STEPS = 2000
# tensorboard
cfg.TRAIN.SAVE_SUMMARY_STEPS = 100
# 保存ckpt最大数
cfg.TRAIN.SAVE_MAX = 100
# log存放地址
cfg.TRAIN.TRAIN_LOGS = os.path.join('/hostpersistent/zzh/ikkyyu/train_data/', save_data_name, 'tf_logs')
# ckpt存放地址
cfg.TRAIN.CHECKPOINTS_OUTPUT_DIR = os.path.join('/hostpersistent/zzh/ikkyyu/train_data/', save_data_name, 'ckpt')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~restore and pretrain~~~~~~~~~~~~~~~~~~~~~
cfg.TRAIN.RESTORE = True
cfg.TRAIN.RESTORE_CKPT_PATH = os.path.join('/hostpersistent/zzh/ikkyyu/train_data/', save_data_name, 'ckpt')
cfg.TRAIN.PRETRAINED_MODEL_PATH = '/hostpersistent/zzh/ikkyyu/east_models/V0.9.29'


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~super em~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cfg.TRAIN.AABB_LOSS = 1
cfg.TRAIN.LEARNING_RATE = 0.0001
cfg.TRAIN.OPT = 'momentum'#'adam'#
cfg.TRAIN.MOVING_AVERAGE_DECAY = 0.997
cfg.TRAIN.MIN_CROP_SIDE_RATIO = 0.00001

