import os
from easydict import EasyDict as edict

cfg = edict()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~inference~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cfg.MEANS = [123.68, 116.78, 103.94]
cfg.INPUT_MAX_SIZE = 720
cfg.GEOMETRY = 'RBOX'
# -1采用原始lanms, >=0 <=1使用最新的lanms
cfg.LONG_MERGE_THRESHOLD = 0.5
# 进行合并的阈值
cfg.LANMS_THRESHOLD = 0.4
# 进行nms的阈值
cfg.NMS_THRESHOLD = 0.3
cfg.EXP_BOX = {'right_ratio': 0, 'left_ratio': 0, 'is_exp': False}
cfg.IS_ASPP = False

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~train config~~~~~~~~~~~~~~~~~~~~~~~~~~~~
save_data_name = '1211'

cfg.TRAIN = edict()
cfg.TRAIN.VERSION = save_data_name + '_'
cfg.TRAIN.INPUT_SIZE = 512
cfg.TRAIN.TEXT_SCALE = 512
# 多gpu训练
cfg.TRAIN.GPU_LIST = '0,1'
cfg.TRAIN.VIS_GPU = '1,2'
# batchsize 大小
cfg.TRAIN.BATCH_SIZE_PER_GPU = 12
# 训练json,存储了图片地址和对应的label地址
cfg.TRAIN.TRAIN_JSONS_LIST = {
                              # # '/share/zzh/train_json/q1_15k.json',
                              # # '/share/zzh/train_json/0909_12k.json',
                              # # '/share/zzh/train_json/0916_12k_train.json',
                              # # '/share/zzh/train_json/kousaun_6k_train.json',
                              # # '/share/zzh/train_json/pdf_train.json',
                              # # '/share/zzh/train_json/pdf_aug.json',
                              # # '/share/zzh/train_json/1017_data.json',
                              # # '/share/zzh/train_json/1017_aug_data.json',
                              # # '/share/zzh/train_json/long_8k.json',
                              # '/share/zzh/train_json/1017_manual_ex_pot_big.json',
                              # # '/share/zzh/train_json/q1_4k.json',
                              # '/share/zzh/train_json/badcase_1029.json',
                              # '/share/zzh/train_json/1017_unit_convert.json',
                              # # '/share/zzh/train_json/1017_manual_goux.json',

    # q1,q2数据
    '/hostpersistent/zzh/dataset/text/train_file/q1_data.json': 0.5,
    '/hostpersistent/zzh/dataset/text/train_file/q2_data.json': 0.5,

    # v3标注数据
    # '/hostpersistent/zzh/dataset/text/train_file/v3_0909.json': 0.5,
    '/hostpersistent/zzh/dataset/text/train_file/v3_0916.json': 1,
    '/hostpersistent/zzh/dataset/text/train_file/1114_24k.json': 1,

    # 小文本数据：pdf数据，口算数据
    '/hostpersistent/zzh/dataset/text/train_file/pdf_data.json': 1,
    '/hostpersistent/zzh/dataset/text/train_file/0808_kousuan.json': 0.5,

    # 增强数据
    # '/hostpersistent/zzh/dataset/text/train_file/aug_kousuan_10k.json': 1,
    '/hostpersistent/zzh/dataset/text/train_file/aug_pdf-8k.json': 0.1,
    '/hostpersistent/zzh/dataset/text/train_file/aug_badcase_1k.json': 1,
    '/hostpersistent/zzh/dataset/text/train_file/aug_long_10k.json': 0.25,
    '/hostpersistent/zzh/dataset/text/train_file/aug_danwei_2k.json': 1,
    '/hostpersistent/zzh/dataset/text/train_file/aug_small_2k.json': 1,
    '/hostpersistent/zzh/dataset/text/train_file/aug_tihao_2k.json': 0.5,

    # '/hostpersistent/zzh/dataset/text/train_file/1017_manual_tihao_pot.json': 1,
    '/hostpersistent/zzh/dataset/text/train_file/1017_manual_away.json': 1,
    '/hostpersistent/zzh/dataset/text/train_file/1017_manual_biglong.json': 1,
    '/hostpersistent/zzh/dataset/text/train_file/1017_manual_danwei.json': 1,
    '/hostpersistent/zzh/dataset/text/train_file/1017_manual_goux.json': 1,
    '/hostpersistent/zzh/dataset/text/train_file/1017_manual_short.json': 1,
    '/hostpersistent/zzh/dataset/text/train_file/1017_manual_small.json': 1,
    '/hostpersistent/zzh/dataset/text/train_file/1017_manual_tihao.json': 1,

    '/hostpersistent/zzh/dataset/text/train_file/1027_15_beitou.json': 1,
    '/hostpersistent/zzh/dataset/text/train_file/1027_15_bigjianju.json': 1,
    '/hostpersistent/zzh/dataset/text/train_file/1027_15_bigziti.json': 1,
    '/hostpersistent/zzh/dataset/text/train_file/1027_15_danziti.json': 1,
    # '/hostpersistent/zzh/dataset/text/train_file/1027_15_goux.json': 1,
    # '/hostpersistent/zzh/dataset/text/train_file/1027_15_liaocao.json': 1,
    # '/hostpersistent/zzh/dataset/text/train_file/1027_15_mohu.json': 1,
    '/hostpersistent/zzh/dataset/text/train_file/1027_15_net.json': 1,
    '/hostpersistent/zzh/dataset/text/train_file/1027_15_near.json': 1,
    '/hostpersistent/zzh/dataset/text/train_file/1027_15_short.json': 1,
    '/hostpersistent/zzh/dataset/text/train_file/1027_15_shushi.json': 1,
    '/hostpersistent/zzh/dataset/text/train_file/1027_15_tihao.json': 1,
    '/hostpersistent/zzh/dataset/text/train_file/1027_15_wanqu.json': 1,

    # 竖式．脱式．解方程
    '/hostpersistent/zzh/dataset/text/train_file/shushi_49367_.json': 0.1,
    '/hostpersistent/zzh/dataset/text/train_file/tuoshi_37965_.json': 0.1,
    '/hostpersistent/zzh/dataset/text/train_file/jiefangcheng_7130_.json': 0.1,

}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~dataload & aug~~~~~~~~~~~~~~~~~~~~~~~~~~
# 图像缩放
cfg.TRAIN.IMG_SCALE = [0.5, 1, 1, 1, 1.5, 2.0]
# crop
cfg.TRAIN.IS_CROP = True
# 颠倒概率
cfg.TRAIN.ROTATE180 = 0.1
# 生成score map的缩放尺度
cfg.TRAIN.SHRINK_RATIO = 0.3
# 是否进行边框外扩
cfg.TRAIN.EXP_ALL_BOX = False
# 外扩边界为１的框
cfg.TRAIN.EXP_1_BOX = False
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

# 对瘦的框进行扩充
cfg.TRAIN.EXP_THIN = True

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
#'/hostpersistent/zzh/ikkyyu/east_models/east_model.ckpt-212101'
# '/hostpersistent/zzh/ikkyyu/east_models/V0.9.05.ckpt'
#'/hostpersistent/zzh/ikkyyu/east_models/'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~super em~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cfg.TRAIN.AABB_LOSS = 1
cfg.TRAIN.LEARNING_RATE = 0.0001
cfg.TRAIN.OPT = 'momentum'#'adam'#
cfg.TRAIN.MOVING_AVERAGE_DECAY = 0.997
# 文本过小,则训练时候忽略这个框,长宽最小值
cfg.TRAIN.MIN_TEXT_SIZE = 2
# when doing random crop from input image, the
# min length of min(H, W)
cfg.TRAIN.MIN_CROP_SIDE_RATIO = 0.00001

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~evalue~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cfg.TRAIN.EVAL_ORG_DIR = '/share/zzh/east_data/test_data/org/'
cfg.TRAIN.EVAL_RES_DIR = '/root/east_data/eval_res'

