import os
from easydict import EasyDict as edict

cfg = edict()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~inference~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cfg.MEANS = [123.68, 116.78, 103.94]
cfg.INPUT_MAX_SIZE = 736
cfg.K = 10
cfg.SHRINK_RATIO = 0.4
cfg.THRESH_MIN = 0.3
cfg.THRESH_MAX = 0.7
cfg.MIN_TEXT_SIZE = 1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~train config~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cfg.TRAIN = edict()
cfg.TRAIN.VERSION = '1216'
# 多gpu训练
cfg.TRAIN.VIS_GPU = '5,6'
cfg.TRAIN.BATCH_SIZE_PER_GPU = 2
cfg.TRAIN.LOSS_ALPHA = 1.0
cfg.TRAIN.LOSS_BETA = 10.0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~dataload & aug~~~~~~~~~~~~~~~~~~~~~~~~~~
cfg.TRAIN.IMG_DIR = '/hostpersistent/zzh/dataset/open_data/ctw1500/train/text_image'
cfg.TRAIN.LABEL_DIR = '/hostpersistent/zzh/dataset/open_data/ctw1500/train/text_label_curve'

cfg.TRAIN.IMG_SIZE = 736
cfg.TRAIN.IMG_SCALE = [0.5, 1, 1, 1, 1.5, 2.0]
cfg.TRAIN.IS_CROP = True
cfg.TRAIN.NUM_READERS = 20
cfg.TRAIN.DATA_AUG_PROB = 0.0
cfg.TRAIN.AUG_TOOL = ['GaussianBlur',
                'AverageBlur',
                'MedianBlur',
                'BilateralBlur',
                'MotionBlur',
                #'ElasticTransformation',
                #'PerspectiveTransform',
                      ]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~save ckpt and log~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cfg.TRAIN.MAX_STEPS = 10000000
cfg.TRAIN.SAVE_CHECKPOINT_STEPS = 2000
cfg.TRAIN.SAVE_SUMMARY_STEPS = 100
cfg.TRAIN.SAVE_MAX = 100
cfg.TRAIN.TRAIN_LOGS = os.path.join('/hostpersistent/zzh/lab/DB-tf', 'tf_logs')
cfg.TRAIN.CHECKPOINTS_OUTPUT_DIR = os.path.join('/hostpersistent/zzh/lab/DB-tf', 'ckpt')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~restore and pretrain~~~~~~~~~~~~~~~~~~~~~
cfg.TRAIN.RESTORE = False
cfg.TRAIN.RESTORE_CKPT_PATH = os.path.join('/hostpersistent/zzh/lab/DB-tf', 'ckpt')
cfg.TRAIN.PRETRAINED_MODEL_PATH = '/hostpersistent/zzh/pretrain_models/resnet_v1_50.ckpt'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~super em~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cfg.TRAIN.LEARNING_RATE = 0.0001
cfg.TRAIN.OPT = 'adam'#'momentum'#
cfg.TRAIN.MOVING_AVERAGE_DECAY = 0.997
cfg.TRAIN.MIN_CROP_SIDE_RATIO = 0.00001

