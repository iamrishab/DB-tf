# coding=utf-8
"""
EAST模型验证：
针对验证数据进行算法模型验证
1、对验证数据进行算法模型测试输出
2、对比模型输出和验证数据进行验证

"""

import os
import cv2
import shapely
import tqdm
import shutil
import json
import numpy as np
import glob
import tensorflow as tf
from PIL import Image
from shapely.geometry import Polygon, MultiPoint  # 多边形

from east_inference import East


def make_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


def read_label_file_from_txt(file_path):
    """
    读取label文件，打印的bbox信息
    :param file_path:
    :param is_gt:
    :return:
    """
    with open(file_path, 'r') as f:
        # 读入所有文本行
        lines = f.readlines()
        # 容器每一行代表一个框(8个点)
        boxes_info_list = []

        # 遍历每行,进行切分
        for line in lines:
            info = line.split(",")
            bbox_info = []
            bbox_info.append(int(info[0]))
            bbox_info.append(int(info[1]))
            bbox_info.append(int(info[2]))
            bbox_info.append(int(info[3]))
            bbox_info.append(int(info[4]))
            bbox_info.append(int(info[5]))
            bbox_info.append(int(info[6]))
            bbox_info.append(int(info[7]))
            boxes_info_list.append(bbox_info)

    return boxes_info_list

def read_label_file_from_json(file_path, exp_thin=False):
    """
    读取label文件，打印的bbox信息
    :param file_path:
    :param is_gt:
    :return:
    """

    json_data = json.loads(open(file_path, encoding='utf-8').read(),
                           encoding='bytes')
    boxes_info_list = []
    for boxes in json_data['text']:
        boxes_info_list.append(boxes['bbox'])

    return boxes_info_list



def polygon_riou(pred_box, gt_box):
    """
    计算预测和gt的riou
    :param pred_box: list [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    :param gt_box: list [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    :return:
    """
    pred_polygon_points = np.array(pred_box).reshape(4, 2)
    pred_poly = Polygon(pred_polygon_points).convex_hull
    gt_polygon_points = np.array(gt_box).reshape(4, 2)
    gt_poly = Polygon(gt_polygon_points).convex_hull
    if not pred_poly.intersects(gt_poly):
        iou = 0
    else:
        try:
            inter_area = pred_poly.intersection(gt_poly).area
            # union_area = gt_box.area
            union_area = gt_poly.area
            if union_area == 0:
                return 0
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

def quad_iou(_gt_bbox, _pre_bbox):
    # 四边形四个点坐标的一维数组表示，[x,y,x,y....]

    gt_poly = Polygon(_gt_bbox).convex_hull  # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上
    # print(Polygon(_gt_bbox).convex_hull)  # 可以打印看看是不是这样子

    pre_poly = Polygon(_pre_bbox).convex_hull
    # print(Polygon(_pre_bbox).convex_hull)

    union_poly = np.concatenate((_gt_bbox, _pre_bbox))  # 合并两个box坐标，变为8*2
    # print(MultiPoint(union_poly).convex_hull)  # 包含两四边形最小的多边形点

    if not gt_poly.intersects(pre_poly):  # 如果两四边形不相交
        iou = 0
        return iou
    else:
        try:
            inter_area = gt_poly.intersection(pre_poly).area  # 相交面积
            # print(inter_area)
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            # print(union_area)
            if union_area == 0:
                iou = 0
            # iou = float(inter_area) / (union_area-inter_area)  #错了
            iou = float(inter_area) / union_area
            # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
            # 源码中给出了两种IOU计算方式，第一种计算的是: 交集部分/包含两个四边形最小多边形的面积
            # 第二种： 交集 / 并集（常见矩形框IOU计算方式）
            return iou
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
            return iou

def exp_box_by_box(box, right_threshold=0.02, left_threshold=0.01):
    right_index = [0, 1, 0, 1, 6, 7, 6, 7]  #
    left_index = [2, 3, 2, 3, 4, 5, 4, 5]
    # right
    vector = box - box[right_index]
    new_box = right_threshold * vector + box

    # left
    vector = new_box - new_box[left_index]
    new_box = left_threshold * vector + new_box

    return new_box

def exp_thin_box(box):
    """
    扩充瘦框
    :param box:
    :return:
    """
    width = box[1][0]-box[0][0]
    height = box[3][1] - box[0][1]
    if height > width:
        ratio = (height-width)/width
        return exp_box_by_box(box.reshape([8,]), ratio/2.5, ratio/2.5)
    else:
        return box

def evaluate(gt_boxes_list, pred_boxes_list, overlap=0.5):
    """
    评价网络模型:param gt_boxes_list: [[x1,y1,x2,y2,x3,y3,x4,y4],...]
    :param pred_boxes_list: [[x1,y1,x2,y2,x3,y3,x4,y4],...]
    :param overlap:
    :param r_overlap:
    :return:
    """
    TP = 0
    pred_flag = [0] * len(pred_boxes_list)
    gt_flag = [0] * len(gt_boxes_list)
    bad_boxes_info = []

    for i, pred_box_info in enumerate(pred_boxes_list):
        pre_bbox = np.array(pred_box_info, np.int32).reshape(4, 2)

        max_iou = 0
        pair_gt_box_id = -1
        pair_gt_box = None

        for j, gt_box_info in enumerate(gt_boxes_list):
            gt_box_info = gt_box_info[0:8]
            if gt_flag[j] == 1:
                continue

            gt_bbox = np.array(gt_box_info).reshape(4, 2)  # 四边形二维坐标表示

            # 传入两个Bbox,返回iou
            iou = quad_iou(pre_bbox, gt_bbox)

            if iou > max_iou:
                max_iou = iou
                pair_gt_box_id = j
                pair_gt_box = gt_bbox
        if EVAL_OR:

            if max_iou > overlap or polygon_riou(pre_bbox, gt_boxes_list[pair_gt_box_id][0:8]) > 0.95:
                TP += 1
                pred_flag[i] = 1
                if pair_gt_box_id != -1:
                    gt_flag[pair_gt_box_id] = 1
            else:
                pred_bad_box = {}
                pred_bad_box['id'] = i
                pred_bad_box['bbox_pts'] = pre_bbox
                pred_bad_box['max_iou'] = max_iou

                if pair_gt_box_id != -1:
                    pred_bad_box['pair_box_pts'] = None
                else:
                    pred_bad_box['pair_box_pts'] = None
                bad_boxes_info.append(pred_bad_box)
        else:
            if max_iou > overlap and polygon_riou(pre_bbox, gt_boxes_list[pair_gt_box_id]) > 0.95:
                TP += 1
                pred_flag[i] = 1
                if pair_gt_box_id != -1:
                    gt_flag[pair_gt_box_id] = 1
            else:
                pred_bad_box = {}
                pred_bad_box['id'] = i
                pred_bad_box['bbox_pts'] = pre_bbox
                pred_bad_box['max_iou'] = max_iou

                if pair_gt_box_id != -1:
                    pred_bad_box['pair_box_pts'] = None
                else:
                    pred_bad_box['pair_box_pts'] = None
                bad_boxes_info.append(pred_bad_box)

    # 查找丢失的gt
    lose_gt_boxes = []
    for i, gt_box_info in enumerate(gt_boxes_list):
        if gt_flag[i] != 1:
            lose_gt_boxes.append(np.array(gt_box_info, np.int32).reshape(4, 2))

    precision = TP / (float(len(pred_boxes_list)) + 1e-5)
    recall = TP / (float(len(gt_boxes_list)) + 1e-5)
    F1_score = 2 * (precision * recall) / (precision + recall + 1e-5)
    pred_boxes = float(len(pred_boxes_list))
    gt_boxes = float(len(gt_boxes_list))

    return TP, precision, recall, F1_score, pred_boxes, gt_boxes, bad_boxes_info, lose_gt_boxes

def evaluate_all(gt_txt_path, pre_txt_path, img_path, badcase_precision_path=None, badcase_recall_path=None, show=True):
    """
    测试指标
    :param gt_txt_path:
    :param pre_txt_path:
    :return:
    """
    global thr_min, thr_max, thr_interval

    # 读取gt下所有TXT文本
    gt_file_list = os.listdir(gt_txt_path)
    # 读取预测图片下TXT文本
    pred_file_list = os.listdir(pre_txt_path)
    print(len(gt_file_list), len(pred_file_list))
    assert len(pred_file_list) == len(gt_file_list), '{}和{}中的文件数目不一致'.format(gt_txt_path, pre_txt_path)

    all_TP = 0.0
    all_pred_num = 0.0
    all_gt_num = 0.0
    img_files_name = os.listdir(img_path)

    # 遍历所有预测文本,即对一个文本(一张图片)进行处理
    for pred_file in tqdm.tqdm(pred_file_list):

        # 若预测文本在gt文本中不存在
        if pred_file not in gt_file_list:
            assert 0, '{}预测文件没有在{}找到应gt文件'.format(pred_file, pre_txt_path)

        gt_bboxes_info_list = read_label_file_from_json(os.path.join(gt_txt_path, pred_file))
        pred_bboxes_info_list = read_label_file_from_json(os.path.join(pre_txt_path, pred_file))

        TP, precision, recall, F1_score, pred_boxes, gt_boxes, bad_boxes_info, lose_gt_boxes = evaluate(gt_bboxes_info_list,
                                                                                         pred_bboxes_info_list,
                                                                                         overlap=0.5)

        all_TP += TP
        all_gt_num += len(gt_bboxes_info_list)
        all_pred_num += len(pred_bboxes_info_list)

        if show:
            if bad_boxes_info:
                basename = pred_file.split('.')[0]
                for img_name in img_files_name:
                    if basename in img_name:
                        img = cv2.imread(os.path.join(img_path, img_name))
                        precison_img = img.copy()
                        recall_img = img.copy()
                        P = False
                        R = False
                        for bad_info in bad_boxes_info:
                            P = True
                            cv2.polylines(precison_img, [bad_info['bbox_pts'].reshape((-1, 1, 2))], True, (0, 0, 255))
                            # if bad_info['pair_box_pts'] is not None:
                            #     cv2.polylines(img, [bad_info['pair_box_pts'].reshape((-1, 1, 2))], True, (0, 255, 0))
                        for lose_box in lose_gt_boxes:
                            R = True
                            cv2.polylines(recall_img, [lose_box.reshape((-1, 1, 2))], True, (0, 255, 0))
                        if P:
                            cv2.imwrite(os.path.join(badcase_precision_path, img_name), precison_img)
                        if R:
                            cv2.imwrite(os.path.join(badcase_recall_path, img_name), recall_img)

                        break

    precision = all_TP / float(all_pred_num + 0.0001)
    recall = all_TP / float(all_gt_num + 0.0001)
    F1_score = 2 * (precision * recall) / (precision + recall + 0.0001)

    print("TP num:" + str(all_TP))
    print("all_gt_num num:" + str(all_gt_num))
    print("all_pred_num num:" + str(all_pred_num))

    print("precision:" + str(precision))
    print("recall:" + str(recall))
    print("F1_score:" + str(F1_score))

    return precision, recall, F1_score, [all_TP, all_gt_num, all_pred_num]


# 判断文件是否为有效（完整）的图片
# 输入参数为文件路径
# 会出现漏检的情况1
def is_valid_image(pathfile):
    bValid = True
    try:
        Image.open(pathfile).verify()
    except:
        bValid = False
    return bValid



def train_eval():
    checkpoint_path = '/hostpersistent/zzh/ikkyyu/train_data/1211/ckpt'

    train_ckpt_list = []

    # 测试图片文件夹路径
    # val_img_dir = '/share/zzh/east_data/test_data/v2/org/imgs'
    # val_label_dir = '/share/zzh/east_data/test_data/v2/org/jsons'
    # val_img_dir = '/share/zzh/east_data/test_data/v2/long/imgs'
    # val_label_dir = '/share/zzh/east_data/test_data/v2/long/jsons'
    # val_img_dir = '/share/zzh/east_data/test_data/v2/merage/imgs'
    # val_label_dir = '/share/zzh/east_data/test_data/v2/merage/jsons'

    val_data_dirs = ['/hostpersistent/zzh/dataset/test/test_data/v2_org/v2',
                     #'/share/zzh/east_data/test_data/v2/org',
                     #'/share/zzh/east_data/test_data/v2/1017_test',
                     #'/share/zzh/east_data/test_data/v2/1017_manual'
                     ]

    res_txt_dir = '/hostpersistent/zzh/ikkyyu/train_data/1211/temp_eval_res/txt'#'/root/east_data/temp_eval_res/txt'
    # make_dir(res_txt_dir)
    res_save_dir = '/hostpersistent/zzh/ikkyyu/train_data/1211'

    # img_list = os.listdir(val_img_dir)

    while True:
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)

        for model_path in ckpt.all_model_checkpoint_paths:
            if model_path in train_ckpt_list:
                continue
            else:
                east = East(model_path)
                for val_data_dir in val_data_dirs:
                    make_dir(res_txt_dir)

                    val_img_dir = os.path.join(val_data_dir, 'imgs')
                    val_label_dir = os.path.join(val_data_dir, 'jsons')
                    img_list = os.listdir(val_img_dir)

                    east.write_img = False
                    print(model_path, ' start to test ', val_data_dir)
                    for img_name in tqdm.tqdm(img_list):
                        east.test(os.path.join(val_img_dir, img_name), res_txt_dir, res_txt_dir, write_img=False)
                    print('done')
                    print('start to evaluate models....')
                    precision, recall, F1_score, _ = evaluate_all(val_label_dir, res_txt_dir, val_img_dir, show=False)

                    data_name = os.path.split(val_data_dir)[-1]
                    with open(os.path.join(res_save_dir, 'train_eval.txt'), 'a') as f:
                        res_str = ' P:' + str(precision) + ' R:' + str(recall) + ' F:' + str(F1_score) + '\r\n'
                        f.writelines(model_path)
                        f.writelines(data_name)
                        f.writelines(res_str)
                    train_ckpt_list.append(model_path)

def find_badcase():

    org_dir = '/share/zzh/parser_data'

    data_dir_list = os.listdir(org_dir)

    img_path_list = []

    for data_dir in data_dir_list:
        img_path_list.extend(glob.glob(os.path.join(org_dir, data_dir, 'imgs', '*')))
    east = East('/share/zzh/east_data/ckpt/online_models/V0.9.29.ckpt')

    bad_img_list = []

    for img_path in tqdm.tqdm(img_path_list):

        img = cv2.imread(img_path)
        pred_boxes = east.east_process(img)

        json_path = img_path.replace('imgs', 'jsons').split('.')[0] + '.json'
        gt_boxes = read_label_file_from_json(json_path)

        try:
            res_list = evaluate(gt_boxes, pred_boxes)
        except:
            print('error label:', json_path)
            continue

        bad_boxes_info = res_list[6]
        lose_gt_boxes = res_list[7]

        if len(lose_gt_boxes) != 0:
            dict = {}
            dict['img_path'] = img_path
            dict['label_path'] = json_path
            bad_img_list.append(dict)
            for box in lose_gt_boxes:
                # box = np.array(box, np.int)
                # print(box)
                cv2.polylines(img, [box.reshape((-1, 1, 2))], True, (255, 0, 0), 2)
            for box in bad_boxes_info:
                cv2.polylines(img, [np.array(box['bbox_pts']).reshape((-1, 1, 2))], True, (0, 255, 0), 2)
            cv2.imwrite(os.path.join('/share/zzh/badcase_show', img_path.split('/')[-1]), img)


    with open('/share/zzh/train_json/1001/badddddd.json', 'w') as f:
        f.writelines(json.dumps(bad_img_list, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    #EVAL_OR = True
    #find_badcase()
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    EVAL_OR = True
    is_train_eval = True

    if is_train_eval:
        # 训练时候的模型验证
        train_eval()
    else:
        # 模型测试
        RUN_MODEL = True

        # checkpoint 文件路径
        checkpoint_path = '/share/zzh/east_data/ckpt/online_models'

        #checkpoint_path = '/share/temp/ckpt'
        #checkpoint_path = '/share/pipeline_version/V1.1.4/pipeline_for_monkey/Models/east'
        assert os.path.exists(checkpoint_path), 'checkpoint is not exists!'

        # 测试图片文件夹路径
        # val_dir = '/share/zzh/east_data/test_data/v2/org/'
        # val_dir = '/share/zzh/east_data/test_data/v2/1017_test/'
        val_dir = '/share/zzh/east_data/test_data/v2/long/'

        val_img_dir = os.path.join(val_dir, 'imgs')
        val_label_dir = os.path.join(val_dir, 'jsons')

        # 算法测试结果和验证结果存储路径
        res_dir = '/share/zzh/east_data/test_data_res/0929-1017/'

        # 读取文件夹下路径
        img_list = os.listdir(val_img_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        print(ckpt)
        print('test models number:', len(ckpt.all_model_checkpoint_paths))
        print('start to run east model...')

        models_res_list = []
        models_test_res_list = []
        models_precison_badcase_dir_list = []
        models_recall_badcase_dir_list = []

        for model_path in ckpt.all_model_checkpoint_paths:

            model_name = model_path.split('/')[-1]

            model_res_dir = os.path.join(res_dir, model_name)

            if RUN_MODEL:
                # 创建当前模型的结果文件
                if not os.path.exists(model_res_dir):
                    # shutil.rmtree(model_res_dir)
                    os.makedirs(model_res_dir)

            models_res_list.append(model_res_dir)

            # 结果图文件夹
            res_img_dir = os.path.join(model_res_dir, 'result_img')
            if RUN_MODEL:
                if os.path.exists(res_img_dir):
                    shutil.rmtree(res_img_dir)
                os.makedirs(res_img_dir)

            # 结果txt文件
            res_txt_dir = os.path.join(model_res_dir, 'result_txt')
            if RUN_MODEL:
                if os.path.exists(res_txt_dir):
                    shutil.rmtree(res_txt_dir)
                os.makedirs(res_txt_dir)
            models_test_res_list.append(res_txt_dir)

            # badcase路径
            badcase_precison_dir = os.path.join(model_res_dir, 'badcase_precison')
            if os.path.exists(badcase_precison_dir):
                shutil.rmtree(badcase_precison_dir)
            os.makedirs(badcase_precison_dir)
            models_precison_badcase_dir_list.append(badcase_precison_dir)

            badcase_recall_dir = os.path.join(model_res_dir, 'badcase_recall')
            if os.path.exists(badcase_recall_dir):
                shutil.rmtree(badcase_recall_dir)
            os.makedirs(badcase_recall_dir)
            models_recall_badcase_dir_list.append(badcase_recall_dir)

            if RUN_MODEL:
                east = East(model_path)
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print(model_name, ' start to run img....')
                for img_name in tqdm.tqdm(img_list):
                    east.test(os.path.join(val_img_dir, img_name), res_txt_dir, res_img_dir, write_img=True)
                print('done')

        print('start to evaluate models....')
        for i, model_txt_dir in enumerate(models_test_res_list):
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('eval ', model_txt_dir)
            precision, recall, F1_score, _ = evaluate_all(val_label_dir, model_txt_dir, val_img_dir, models_precison_badcase_dir_list[i], models_recall_badcase_dir_list[i])
            with open(os.path.join(models_res_list[i], 'result_and.txt'), 'w') as f:
                f.writelines('precision'+str(precision))
                f.writelines('recall'+str(recall))
                f.writelines('F1_score'+str(F1_score))
        print('done')
