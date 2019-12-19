# DB: Real-time Scene Text Detection with Differentiable Binarization


## Introduction
This is a TensorFlow implementation of ["Real-time Scene Text Detection with Differentiable Binarization"](https://arxiv.org/abs/1911.08947).

Part of the code is inherited from [DB](https://github.com/MhLiao/DB).

![net](figures/net.png)


## Requirements:
- Python3
- Tensorflow >= 1.13 
- easydict


## Dataset
This repo is train on CTW1500 dataset.
Download from [BaiduYun](https://pan.baidu.com/s/1yG_191LemrQa7K0h7Wispw) (key:yjiz) or 
[OneDrive](https://1drv.ms/u/s!Aplwt7jiPGKilH4XzZPoKrO7Aulk).


## Test

### 1.Download model.
Download from [BaiduYun]().

### 2.Start to test img.

    python inference.py --gpuid='0' --ckptpath='path' --imgpath='img.jpg'


## Samples show

| org show 	| poly show 	| bbox show 	|
|------------	|-------	|-------	|
| ![poly_img](figures/org.jpg) 	| ![poly_img](figures/1039_polyshow.jpg) 	| ![bbox_img](figures/1039_bboxshow.jpg) 	|
| binarize_map |  threshold_map	| thresh_binary |
| ![bin_map](figures/1039_binarize_map.jpg) |  ![thres_map](figures/1039_threshold_map.jpg)	| ![bin_thres_map](figures/1039_thresh_binary.jpg) | 

## Training model
#### 1. Get the CTW1500 train images path and labels path.

revise the `db_config.py`

    cfg.TRAIN.IMG_DIR = '/path/ctw1500/train/text_image'
    cfg.TRAIN.LABEL_DIR = '/path/ctw1500/train/text_label_curve'

#### 2. Muti gpu train.

revise the `db_config.py`

    cfg.TRAIN.VIS_GPU = '5,6' # single gpu -> '0'
    
#### 3. Save train logs and models.

revise the `db_config.py`

    cfg.TRAIN.TRAIN_LOGS = '/path/tf_logs'
    cfg.TRAIN.CHECKPOINTS_OUTPUT_DIR = '/path/ckpt'
    
#### 4. Pretrain or restore model.

If you want to pretrain model,

revise the `db_config.py`

    cfg.TRAIN.RESTORE = False
    cfg.TRAIN.PRETRAINED_MODEL_PATH = 'pretrain model path'
    
If you want to restore model,

revise the `db_config.py`

    cfg.TRAIN.RESTORE = True
    cfg.TRAIN.RESTORE_CKPT_PATH = 'checkpoint path'

#### 5. Start to train.

    python train.py

tensorboard show

|   binarize_loss	|   thresh_binary_loss	|
|------------	|-------	|
| ![binarize_loss](figures/1.png) 	| ![thresh_binary_loss](figures/3.png)	|
|   model_loss 	|   total_loss	|
| ![model_loss](figures/2.png) 	| ![total_loss](figures/4.png) 	|

## Experiment

Test on RTX 2080 Ti.

|   BackBone	|   Input Size	|   Infernce Time(ms)	|	PostProcess Time(ms) |
|------------	|-------	|-------	|-------	|
| ResNet-50 	| 320	| 13.3 | 2.9 |
| ResNet-50 	| 512	| 19.2 | 4.5 |
| ResNet-50 	| 736	| 33.2 | 5.7 |
| ResNet-18 	| 320	|  |  |
| ResNet-18 	| 512	|  |  |
| ResNet-18 	| 736	|  |  |



## ToDo List

- [x] Release trained models
- [x] Training code
- [x] Inference code
- [x] Muti gpu training
- [x] Tensorboard support
- [x] Threshold loss : L1 loss -> Smooth L1 loss
- [ ] Eval code
- [ ] Data augmentation
- [x] More backbones
- [ ] Add ASPP
- [ ] Deformable Convolutional Networks
