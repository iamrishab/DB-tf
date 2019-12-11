# EAST: An Efficient and Accurate Scene Text Detector


## 一、安装
- python3
- tensorflow1.8


## 二、训练

### 1.准备训练数据
准备image文件夹和label文件夹，并修改config.yml参数
        
    TRAIN:
        TRAINING_DATA_DIR: "/home/tony/ocr/ocr_dataset/tal_ocr_data_v1/img"
        TRAINING_DATA_LABEL_DIR: "/home/tony/ocr/ocr_dataset/tal_ocr_data_v1/label"

### 2.label数据格式

[icdar2015官方数据集](https://axer.ailab.100tal.com/main/team/dataset/dataset-detail/5f6ed2081d4645fb9f06ed9d4d3c5989)

若图片名字为demo.jpg，label文件名应为gt_demo.jpg

label数据格式：
    x1,y1,x2,y2,x3,y3,x4,y4,text
例如：
    533,134,562,133,561,145,532,146,EW15

### 3.训练参数
根据config.yml文件进行调整训练和测试参数。

### 4.开始训练
    python east_train.py

## 三、测试
下载[模型]放在ftp,下载后放到checkpoints中
    
配置test参数，见config.yml
    TEST:
        CHECKPOINT_DIR: './checkpoints/' #模型加载地址
        TEST_DIR: '/home/zzh/ocr/dataset/icdar2015/val/img' #测试图片地址
        RESULT_DIR: './data/result'  #测试结果存放地址
        SCORE_MAP_DIR: './data/score_map' #score map结果存放地址
        GEO_MAP_DIR: './data/geo'  #geo结果存放地址
        WRITE_IMAGES: 1    
执行
    python east_test.py
    
## 四、接口调用
east_inference.py

使用
    CKPT_PATH = '/workspace/boby/model/EAST_tf/ckpt/model.ckpt-417144'
    east = East(CKPT_PATH) #1.传入模型
    res = east.east_process(temp_img) #2.传入图片

