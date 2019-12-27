# coding=utf-8

import os
import cv2
import tqdm
import numpy as np

from inference import DB
from db_config import cfg
from lib.utils import quad_iou, compute_f1_score, load_ctw1500_labels, make_dir


def load_pred_labels(path):
    pass

def evaluate(gt_care_list, gt_dontcare_list, pred_list, overlap=0.5):
    """

    :param gt_care_list: [-1, M, 2]
    :param gt_dontcare_list: [-1, M, 2]
    :param pred_list: [-1, M, 2]
    :param overlap:
    :return:
    """

    pred_care_list =[]
    pred_dontcare_list = []

    if len(gt_dontcare_list) != 0:
        for pred_box in pred_list:
            flag = False
            for gt_box in gt_dontcare_list:
                if quad_iou(gt_box, pred_box) > overlap:
                    flag = True
                    break

            if not flag:
                pred_care_list.append(pred_box)
            else:
                pred_dontcare_list.append(pred_box)
    else:
        pred_care_list = pred_list

    gt_care_flag_list = [False] * len(gt_care_list)
    pred_care_flag_list = [False] * len(pred_care_list)
    pairs_list = []
    gt_not_pair_list = []
    pred_not_pair_list = []

    for gt_i, gt_box in enumerate(gt_care_list):
        for pred_i, pred_box in enumerate(pred_care_list):
            if pred_care_flag_list[pred_i]:
                continue
            else:
                iou = quad_iou(gt_box, pred_box)
                if iou > overlap:
                    pair_dict = {}
                    pair_dict['gt'] = gt_box
                    pair_dict['pred'] = pred_box
                    pair_dict['iou'] = iou
                    pairs_list.append(pair_dict)
                    pred_care_flag_list[pred_i] = True
                    gt_care_flag_list[gt_i] = True

    TP = len(pairs_list)

    if len(gt_care_list) == 0:
        recall = 1.0
        precision = 1.0 if len(pred_care_list) == 0 else 0.0
    elif len(pred_care_list) == 0:
        recall = 0.0
        precision = 0.0
    else:
        recall = 1.0 * TP / len(gt_care_list)
        precision = 1.0 * TP / len(pred_care_list)

    f1_score = compute_f1_score(precision, recall)

    return precision, recall, f1_score, TP, len(gt_care_list), len(pred_care_list), pairs_list


def evaluate_all(gt_file_dir, gt_img_dir, ckpt_path, gpuid='0'):
    db = DB(ckpt_path, gpuid)

    img_list = os.listdir(gt_img_dir)

    show = './eva'
    make_dir(show)

    total_TP = 0
    total_gt_care_num = 0
    total_pred_care_num = 0
    for img_name in tqdm.tqdm(img_list):
        img = cv2.imread(os.path.join(gt_img_dir, img_name))

        pred_box_list, pred_score_list, _ = db.detect_img(os.path.join(gt_img_dir, img_name),
                                                          ispoly=True,
                                                          show_res=False)

        gt_file_name = os.path.splitext(img_name)[0] + '.txt'

        gt_boxes, tags = load_ctw1500_labels(os.path.join(gt_file_dir, gt_file_name))

        gt_care_list = []
        gt_dontcare_list = []

        for i, box in enumerate(gt_boxes):
            box = box.reshape((-1, 2)).tolist()
            if tags[i] == False:
                gt_care_list.append(box)
            else:
                gt_dontcare_list.append(box)

        precision, recall, f1_score, TP, gt_care_num, pred_care_num, pairs_list = evaluate(gt_care_list,
                                                                               gt_dontcare_list,
                                                                               pred_box_list,
                                                                               overlap=0.5)

        for pair in pairs_list:
            cv2.polylines(img, [np.array(pair['gt'], np.int).reshape([-1, 1, 2])], True, (0, 255, 0))
            cv2.polylines(img, [np.array(pair['pred'], np.int).reshape([-1, 1, 2])], True, (255, 0, 0))

        cv2.imwrite(os.path.join(show, img_name), img)

        total_TP += TP
        total_gt_care_num += gt_care_num
        total_pred_care_num += pred_care_num

    total_precision = float(total_TP) / total_pred_care_num
    total_recall = float(total_TP) / total_gt_care_num
    total_f1_score = compute_f1_score(total_precision, total_recall)

    return total_precision, total_recall, total_f1_score

if __name__ == '__main__':

    ckpt_path = '/hostpersistent/zzh/lab/DB-tf/ckpt/DB_resnet_v1_50_aspp_model.ckpt-303001'
    gt_img_dir = cfg.EVAL.IMG_DIR
    gt_file_dir = cfg.EVAL.LABEL_DIR

    precision, recall, f1_score = evaluate_all(gt_file_dir, gt_img_dir, ckpt_path)
    print(precision, recall, f1_score)






