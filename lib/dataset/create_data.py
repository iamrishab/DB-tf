import os
import shutil
import cv2
import random
import tqdm
import json
import numpy as np

"""
用于创建训练数据和验证集数据
"""

def clean_create_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def resize_img(img, max_side_len=720):
    """
    将图片进行resize，将对短边缩放到img_size大小
    :param img:
    :return:
    """
    h, w, _ = img.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resized_img = cv2.resize(img, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return resized_img, ratio_h, ratio_w


def process_data(img_list, raw_data_dir, save_dir):
    """
    处理数据, 将原始图片和boxes进行resize，写入到txt文件中
    :param img_list:
    :return:
    """
    raw_img_dir = os.path.join(raw_data_dir, 'imgs')
    raw_label_dir = os.path.join(raw_data_dir, 'jsons')
    img_dir = os.path.join(save_dir, 'img_720')
    label_dir = os.path.join(save_dir, 'label_720')
    show_dir = os.path.join(save_dir, 'show_720')

    for img_name in tqdm.tqdm(img_list):
        # 进行图像缩放
        img = cv2.imread(os.path.join(raw_img_dir, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_img, ratio_h, ratio_w = resize_img(img)

        show_img = resized_img.copy()

        # 解析json
        base_name = img_name.split('.')[0]
        json_file = base_name + '.json'
        if not os.path.exists(os.path.join(raw_label_dir, json_file)):
            continue
        boxes_dict = json.loads(open(os.path.join(raw_label_dir, json_file), encoding='utf-8').read(), encoding='bytes')
        # print(box_dict)
        if not boxes_dict:
            print(img_name, 'is error label')
            continue
        with open(os.path.join(label_dir, base_name + '.txt'), 'w') as f:
            for box_dict in boxes_dict:
                box = box_dict['bbox']
                box_np = np.array(box).reshape([4, 2])
                box_np[:, 0] = box_np[:, 0] * ratio_w
                box_np[:, 1] = box_np[:, 1] * ratio_h
                # if EXP_DATA:
                #     box_np = expansion_box(box_np.reshape([8,])).reshape([4, 2])
                box_np = box_np.astype(np.int32)

                cv2.polylines(show_img,
                              [box_np.reshape((-1, 1, 2))],
                              True,
                              (0, 255, 0))
                line = '{},{},{},{},{},{},{},{},###\n'.format(box_np[0][0],box_np[0][1],
                                                            box_np[1][0],box_np[1][1],
                                                            box_np[2][0],box_np[2][1],
                                                            box_np[3][0],box_np[3][1])
                f.writelines(line)
        # 写resized图和框图
        cv2.imwrite(os.path.join(img_dir, img_name), resized_img)
        cv2.imwrite(os.path.join(show_dir, img_name), show_img)


def create_data(raw_data_dir, train_save_dir, val_save_dir, ratio=0.99):
    """
    创建训练数据集和验证数据集
    :param ratio: 训练集占数据集的比例。
    :return:
    """

    all_img_list = os.listdir(os.path.join(raw_data_dir, 'imgs'))
    train_data_num = int(len(all_img_list) * ratio)
    random.shuffle(all_img_list)
    train_data_list = all_img_list[0:train_data_num]
    val_data_list = all_img_list[train_data_num:]

    print('start to process train data ....')
    process_data(train_data_list, raw_data_dir, train_save_dir)
    print('start to process val data ....')
    process_data(val_data_list, raw_data_dir, val_save_dir)

def create_test_data(raw_data_dir, test_save_dir):
    """
    创建训练数据集和验证数据集
    :param ratio: 训练集占数据集的比例。
    :return:
    """

    all_img_list = os.listdir(os.path.join(raw_data_dir, 'imgs'))

    print('start to process test data ....')
    process_data(all_img_list, raw_data_dir, test_save_dir)


if __name__ == '__main__':

    # handle　one　data
    '''
    raw_data_dir = '/share/zzh/raw_train_data/0808_3000'
    train_dir = '/share/zzh/east_train_data/train_data'

    val_dir = '/share/zzh/east_train_data/val_data'

    data_dirs = os.listdir(raw_data_dir)
    print('handle ', os.path.join(raw_data_dir))
    create_data(os.path.join(raw_data_dir), train_dir, val_dir)
    '''
    # handle all data

    raw_data_dir = '/share/zzh/raw_train_data/'

    train_dir = '/share/zzh/east_train_data/train_data'

    val_dir = '/share/zzh/east_train_data/val_data'

    clean_create_dir(os.path.join(train_dir, 'img_720'))
    clean_create_dir(os.path.join(train_dir, 'label_720'))
    clean_create_dir(os.path.join(train_dir, 'show_720'))

    clean_create_dir(os.path.join(val_dir, 'img_720'))
    clean_create_dir(os.path.join(val_dir, 'label_720'))
    clean_create_dir(os.path.join(val_dir, 'show_720'))

    data_dirs = os.listdir(raw_data_dir)
    for data_dir in data_dirs:
        print('handle ', os.path.join(raw_data_dir, data_dir))
        create_data(os.path.join(raw_data_dir, data_dir), train_dir, val_dir)

    #
    #
    # raw_data_dir = '/share/zzh/east_data/raw_test_data'
    #
    # test_dir = '/share/zzh/east_data/test_data'
    #
    # clean_create_dir(os.path.join(test_dir, 'img_720'))
    # clean_create_dir(os.path.join(test_dir, 'label_720'))
    # clean_create_dir(os.path.join(test_dir, 'show_720'))
    #
    #
    # create_test_data(raw_data_dir, test_dir)




