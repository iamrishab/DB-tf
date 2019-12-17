# DB: Real-time Scene Text Detection with Differentiable Binarization


## Introduction
This is a TensorFlow implementation of ["Real-time Scene Text Detection with Differentiable Binarization"](https://arxiv.org/abs/1911.08947).

Part of the code is inherited from [DB](https://github.com/MhLiao/DB).


## Requirements:
- Python3
- Tensorflow >= 1.13 
- easydict

## Dataset
This repo is train on CTW1500 dataset.
Download from [BaiduYun](https://pan.baidu.com/s/1yG_191LemrQa7K0h7Wispw) (key:yjiz) or 
[OneDrive](https://1drv.ms/u/s!Aplwt7jiPGKilH4XzZPoKrO7Aulk).


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

![dice_loss](figures/1.png)
![total_loss](figures/2.png)
![model_loss](figures/3.png)


## Test

### 1.Download model.
Download from [BaiduYun]().

### 2.Start to test img.

    python inference.py --gpuid='0' --ckptpath='path' --imgpath='img.jpg'

## Performance



## Samples

poly show 
![poly_img](figures/1039_polyshow.jpg) 

bbox show
![bbox_img](figures/1039_bboxshow.jpg)

binarize_map
![bin_map](figures/1039_binarize_map.jpg)

threshold_map
![thres_map](figures/1039_threshold_map.jpg)

thresh_binary
![bin_thres_map](figures/1039_thresh_binary.jpg)

## ToDo List

- [x] Release trained models
- [x] Training code
- [x] Inference code
- [x] Muti gpu training
- [x] Easy to read the code
- [x] Tensorboard support
- [ ] Eval code
- [ ] Data augmentation
- [ ] More backbones
- [ ] Deformable Convolutional Networks