import os
import shutil
import cv2
import random
import tqdm
import json
import numpy as np
import re

"""
用于创建长文本和颠倒文本
"""

def clean_create_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def get_boxes_info_from_txt(label_path):
    # text_polys = []
    label = []
    if not os.path.exists(label_path):
        return None

    bbox_list = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # info = line.split(',')
            # bbox = []
            pattern = "\d+"
            bbox = [int(i) for i in re.findall(pattern, line)]
            bbox_list.append(bbox)

    if len(bbox_list) == 0:
        return None

    boxes_temp_list = []
    for bbox in bbox_list:
        new_bbox = []
        new_bbox.append([bbox[0], bbox[1]])
        new_bbox.append([bbox[2], bbox[3]])
        new_bbox.append([bbox[4], bbox[5]])
        new_bbox.append([bbox[6], bbox[7]])
        # 转为最小包围矩形
        rect = cv2.minAreaRect(np.array(new_bbox))
        # text_polys.append(cv2.boxPoints(rect))
        boxes_temp_list.append(cv2.boxPoints(rect))
        label.append(0)

    return np.array(boxes_temp_list, dtype=np.int), label


def get_boxes_info(label_path):
    data = json.loads(open(label_path, encoding='utf-8').read(),
                      encoding='bytes')
    text_data = data['text']
    bbox_list = []
    cls_list = []
    for box_info in text_data:
        bbox_list.append(box_info['bbox'])
        cls_list.append(0)
    return bbox_list, cls_list

def clip_img(img, bboxes, classes):

    def find_clip_dot(line):
        dots_index = np.where(line==0)
        if len(dots_index[0]) == 0:
            return [0]

        midle = int(len(line)/2)
        dots_index = dots_index[0]
        distance = abs(dots_index-midle)

        dot = dots_index[np.where(distance == min(distance))]
        return dot

    def clip_img_labels(img, bboxes, point, type):
        if type == 1:
            cut_img = img[0:point[0]+1, 0:point[1]+1,:]
            res_bboxes = bboxes
        elif type == 2:
            cut_img = img[0:point[0], point[1]:, :]
            res_bboxes = []
            for bbox in bboxes:
                if len(bbox) == 4:
                    bbox[0] -= point[1]
                    bbox[2] -= point[1]
                else:
                    bbox[0] -= point[1]
                    bbox[2] -= point[1]
                    bbox[4] -= point[1]
                    bbox[6] -= point[1]
                res_bboxes.append(bbox)
        elif type == 3:
            cut_img = img[point[0]:, 0:point[1]+1, :]
            res_bboxes = []
            for bbox in bboxes:
                if len(bbox) == 4:
                    bbox[1] -= point[0]
                    bbox[3] -= point[0]
                else:
                    bbox[1] -= point[0]
                    bbox[3] -= point[0]
                    bbox[5] -= point[0]
                    bbox[7] -= point[0]
                res_bboxes.append(bbox)
        elif type == 4:
            cut_img = img[point[0]:, point[1]:, :]
            res_bboxes = []
            for bbox in bboxes:
                if len(bbox) == 4:
                    bbox[1] -= point[0]
                    bbox[3] -= point[0]
                    bbox[0] -= point[1]
                    bbox[2] -= point[1]
                else:
                    bbox[1] -= point[0]
                    bbox[3] -= point[0]
                    bbox[5] -= point[0]
                    bbox[7] -= point[0]
                    bbox[0] -= point[1]
                    bbox[2] -= point[1]
                    bbox[4] -= point[1]
                    bbox[6] -= point[1]
                res_bboxes.append(bbox)

        return cut_img, res_bboxes
    # print(img_path)
    # img = cv2.imread(img_path)
    #
    # classes, bboxes = get_boxes_info(label_path)

    h, w, _ = img.shape

    h_flag = np.zeros([h])
    w_flag = np.zeros([w])

    for bbox in bboxes:
        if len(bbox) == 4:
            h_flag[bbox[1]:bbox[3]+1] = 1
            w_flag[bbox[0]:bbox[2]+1] = 1
        elif len(bbox) == 8:
            h_flag[min(bbox[1], bbox[3], bbox[5], bbox[7]):max(bbox[1], bbox[3], bbox[5], bbox[7])+1] = 1
            w_flag[min(bbox[0], bbox[2], bbox[4], bbox[6]):max(bbox[0], bbox[2], bbox[4], bbox[6])+1] = 1

    cut_h_index = find_clip_dot(h_flag)[0]
    cut_w_index = find_clip_dot(w_flag)[0]

    bboxes_top_l = []
    bboxes_top_r = []
    bboxes_bot_l = []
    bboxes_bot_r = []

    classes_top_l = []
    classes_top_r = []
    classes_bot_l = []
    classes_bot_r = []

    for i, bbox in enumerate(bboxes):
        # 先分为上下两部分
        if bbox[1] < cut_h_index:
            if bbox[0] < cut_w_index:
                bboxes_top_l.append(bbox)
                classes_top_l.append(classes[i])
            else:
                bboxes_top_r.append(bbox)
                classes_top_r.append(classes[i])
        else:
            if bbox[0] < cut_w_index:
                bboxes_bot_l.append(bbox)
                classes_bot_l.append(classes[i])
            else:
                bboxes_bot_r.append(bbox)
                classes_bot_r.append(classes[i])

    point = [cut_h_index,cut_w_index]
    cut_dataset_list = []

    num = 3
    if len(bboxes_top_l)>num:
        data_dict = {}
        cut_img, res_bboxes = clip_img_labels(img.copy(), bboxes_top_l, point, 1)
        data_dict["img"] = cut_img
        data_dict["class"] = classes_top_l
        data_dict["bbox"] = res_bboxes
        cut_dataset_list.append(data_dict)

    if len(bboxes_top_r)>num:
        data_dict = {}
        cut_img, res_bboxes = clip_img_labels(img.copy(), bboxes_top_r, point, 2)
        # draw_bbox(cut_img, classes_top_r, res_bboxes, "_2.jpg")
        data_dict["img"] = cut_img
        data_dict["class"] = classes_top_r
        data_dict["bbox"] = res_bboxes
        cut_dataset_list.append(data_dict)

    if len(bboxes_bot_l)>num:
        data_dict = {}
        cut_img, res_bboxes = clip_img_labels(img.copy(), bboxes_bot_l, point, 3)
        # draw_bbox(cut_img, classes_bot_l, res_bboxes, "_3.jpg")
        data_dict["img"] = cut_img
        data_dict["class"] = classes_bot_l
        data_dict["bbox"] = res_bboxes
        cut_dataset_list.append(data_dict)

    if len(bboxes_bot_r)>num:
        data_dict = {}
        cut_img, res_bboxes = clip_img_labels(img.copy(), bboxes_bot_r, point, 4)
        # draw_bbox(cut_img, classes_bot_r, res_bboxes, "_4.jpg")
        data_dict["img"] = cut_img
        data_dict["class"] = classes_bot_r
        data_dict["bbox"] = res_bboxes
        cut_dataset_list.append(data_dict)

    return cut_dataset_list

def create_inversion_data(cut_dataset_list):
    """
    创建颠倒数据
    :param cut_dataset_list:
    :return:
    """
    clip_num = len(cut_dataset_list)
    if clip_num <2:
        return None
    np.random.shuffle(cut_dataset_list)

    rand = random.random()
    # rand = 0.3

    img1 = cut_dataset_list[0]['img'].copy()
    img2 = cut_dataset_list[1]['img'].copy()
    img1_shape = img1.shape
    img2_shape = img2.shape

    # 颠倒在左，正图在右
    if rand < 0.25:
        # 新图shape = [最大h，两图宽和]
        new_img = np.zeros([max(img1_shape[0], img2_shape[0]), img1_shape[1]+img2_shape[1], 3])
        img1 = np.rot90(img1)
        img1 = np.rot90(img1)
        new_img[0:img1_shape[0], 0:img1_shape[1], :] = img1
        new_img[0:img2_shape[0], img1_shape[1]:img1_shape[1]+img2_shape[1], :] = img2
        boxes = np.array(cut_dataset_list[1]['bbox']).reshape([-1, 4, 2])
        boxes[:, :, 0] += img1_shape[1]
        return (new_img, boxes, cut_dataset_list[1]['class'])
    # 颠倒在右，正图在左
    elif rand < 0.50 and rand > 0.25:
        # 新图shape = [最大h，两图宽和]
        new_img = np.zeros([max(img1_shape[0], img2_shape[0]), img1_shape[1] + img2_shape[1], 3])
        img2 = np.rot90(img2)
        img2 = np.rot90(img2)
        new_img[0:img1_shape[0], 0:img1_shape[1], :] = img1
        new_img[0:img2_shape[0], img1_shape[1]:img1_shape[1]+img2_shape[1], :] = img2
        boxes = np.array(cut_dataset_list[0]['bbox']).reshape([-1, 4, 2])
        # boxes[:, :, 0] += img1_shape[1]
        return (new_img, boxes, cut_dataset_list[0]['class'])
    # 颠倒在上，正图在下
    elif rand < 0.75 and rand > 0.50:
        # 新图shape = [h之和，两最大ｗ]
        new_img = np.zeros([img1_shape[0] + img2_shape[0], max(img1_shape[1], img2_shape[1]), 3])
        img1 = np.rot90(img1)
        img1 = np.rot90(img1)
        new_img[0:img1_shape[0], 0:img1_shape[1], :] = img1
        new_img[img1_shape[0]:img1_shape[0]+img2_shape[0], 0:img2_shape[1], :] = img2
        boxes = np.array(cut_dataset_list[1]['bbox']).reshape([-1, 4, 2])
        boxes[:, :, 1] += img1_shape[0]
        return (new_img, boxes, cut_dataset_list[1]['class'])
    # 颠倒在下，正图在上
    else:
        # 新图shape = [h之和，最大ｗ]
        new_img = np.zeros([img1_shape[0] + img2_shape[0], max(img1_shape[1], img2_shape[1]), 3])
        img2 = np.rot90(img2)
        img2 = np.rot90(img2)
        new_img[0:img1_shape[0], 0:img1_shape[1], :] = img1
        new_img[img1_shape[0]:img1_shape[0]+img2_shape[0], 0:img2_shape[1], :] = img2
        boxes = np.array(cut_dataset_list[0]['bbox']).reshape([-1, 4, 2])
        # boxes[:, :, 0] += img1_shape[0]
        return (new_img, boxes, cut_dataset_list[0]['class'])

def create_long_text(cls_list):
    """
    创建长文本
    :param cls_list:
    :return:
    """
    np.random.shuffle(cls_list)
    return cls_list[0]['img'], cls_list[0]['bbox'], cls_list[0]['class']

def draw_img(img, boxes, cls, name):
    color_list = [(0, 255, 0), (255, 0, 0)]
    for i, box in enumerate(boxes):
        # print(cls[i])
        cv2.polylines(img, [np.array(box).reshape((-1, 1, 2))], True, color_list[cls[i]])
    cv2.imwrite(os.path.join('/home/tony/ikkyyu/test/test/', name), img)

if __name__ == '__main__':
    img_dir = '/share/zzh/east_data/test_data_new/org/raw_test_data/imgs'
    label_dir = '/share/zzh/east_data/test_data_new/org/raw_test_data/jsons'

    save_img_dir = '/share/zzh/east_data/test_data_new/inversion/imgs'
    save_label_dir = '/share/zzh/east_data/test_data_new/inversion/jsons'
    save_show_dir = '/share/zzh/east_data/test_data_new/inversion/show'

    img_path_list = os.listdir(img_dir)
    np.random.shuffle(img_path_list)

    need = img_path_list[0:6000]

    for img_name in tqdm.tqdm(need):
        base_name = img_name.split('.')[0]

        label_path = os.path.join(label_dir, base_name + '.json')
        img = cv2.imread(os.path.join(img_dir, img_name))[:, :, ::-1]

        data = get_boxes_info(label_path)
        if data is not None:
            boxes, cls = data
        else:
            continue
        # draw_img(img.copy(), boxes, cls, 'show.jpg')

        cut_dataset_list = clip_img(img, np.array(boxes).reshape((-1, 8)), cls)

        # 长文本
        # img, bbox, cls = create_long_text(cut_dataset_list)
        # draw_img(img.copy(), bbox, cls, 'long.jpg')
        # 颠倒
        data = create_inversion_data(cut_dataset_list)
        if data!=None:
            img, bboxes, cls = data

            cv2.imwrite(os.path.join(save_img_dir, base_name + '.jpg'), img)
            img_data_dict = {}
            text_data_list = []
            for box in bboxes:
                box_dict = {}

                box_dict['bbox'] = np.array(box, np.int32).tolist()
                box_dict['label'] = 'text'
                box_dict['context'] = 'None'
                text_data_list.append(box_dict)
                cv2.polylines(img, [np.array(box).reshape((-1, 1, 2))], True, (0, 255, 0))
            cv2.imwrite(os.path.join(save_show_dir, base_name + '.jpg'), img)
            img_data_dict['text'] = text_data_list
            # print(img_data_dict)
            with open(os.path.join(save_label_dir, base_name + '.json'), 'w') as f:
                f.writelines(json.dumps(img_data_dict, indent=4, ensure_ascii=False))



            # base_name = 'inversion_' + base_name
            # cv2.imwrite(os.path.join(save_img_dir, base_name + '.jpg'), img)
            # show_img = img.copy()
            # with open(os.path.join(save_label_dir, base_name + '.txt'), 'w') as f:
            #     for box in boxes:
            #         box_np = box.astype(np.int32)
            #
            #         cv2.polylines(show_img,
            #                       [box_np.reshape((-1, 1, 2))],
            #                       True,
            #                       (0, 255, 0))
            #         line = '{},{},{},{},{},{},{},{},###\n'.format(box_np[0][0], box_np[0][1],
            #                                                       box_np[1][0], box_np[1][1],
            #                                                       box_np[2][0], box_np[2][1],
            #                                                       box_np[3][0], box_np[3][1])
            #         f.writelines(line)
            #
            # cv2.imwrite(os.path.join(save_show_dir, base_name + '.jpg'), show_img)