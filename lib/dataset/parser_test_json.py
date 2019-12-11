# coding:utf-8
import os
import shutil
import cv2
import json
import re
from urllib import request
import numpy as np
from PIL import Image
from tqdm import tqdm
import shapely
from shapely.geometry import Polygon, MultiPoint  # 多边形

def  make_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

#
# # RAW_JSONS_DIR = '/share/zzh/raw_jsons/0723-badcase2'
# # SAVE_DIR = '/share/zzh/raw_train_data/0723-badcase2'
# RAW_JSONS_DIR = '/home/tony/data/text'
#
# SAVE_DIR = '/home/tony/data/text_parser'
# #
# IMG_DIR = os.path.join(SAVE_DIR, 'imgs')
# JSON_DIR = os.path.join(SAVE_DIR, 'jsons')
# SHOW_DIR = os.path.join(SAVE_DIR, 'show')



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


def label_is_text_box(make_data):
    """
    根据标签解析检测数据，判断标签类型
    :param make_data:
    :return:
    """
    label_dict = make_data['label']

    # 解析文本行和单字符
    # if '字符' in label:
    #     return True, 'character'

    # 解析文本
    # text = make_data['marked_text']
    # if '对' in label or '错' in label:
    #     return True, 'text'
    # else:
    #     return None, None

    for label in label_dict:
        if '对' in label['children']:
            return True, 'text'
        elif '错' in label['children']:
            return True, 'text'
        else:
            continue
    return None, None
    # 解析字符
    # label = make_data['label'][0]['children']
    # if '字符' in label:
    #     return True
    # else:
    #     return False

    #　解析大框
    # label = make_data['label'][0]['children']
    # text = make_data['marked_text']
    # #
    # if '脱式' in label or '解方程' in label or '竖式' in label:
    #     return True
    # else:
    #     return False

    # 解析文本行
    # if '文本' in label or '横式' in label or '解题过程' in label:
    #     if '脱式' in label or '解方程' in label or '竖式' in label:
    #         if bool(re.search(r'\d', text)):
    #             print(text)
    #             return True
    #         else:
    #             return False
    #     else:
    #         return True
    # else:
    #     return False


def get_rect(bbox):
    x1 = float(bbox[0][1:])
    y1 = float(bbox[1])
    x2 = float(bbox[2][1:])
    y2 = float(bbox[3])
    x3 = float(bbox[4][1:])
    y3 = float(bbox[5])
    x4 = float(bbox[6][1:])
    y4 = float(bbox[7])
    xy = np.array([x1, y1, x2, y2, x3, y3, x4, y4])
    return xy


def get_json_by_img_data(json_dir):
    """
    获取对应image_datas字段的json的信息
    :param json_dir:
    :return:
    """
    json_names = os.listdir(json_dir)
    images_datas = []
    for json_name in json_names:

        json_path = os.path.join(json_dir, json_name)
        json_file = json.loads(open(json_path).read())
        try:
            images_datas_temp = json_file['image_datas']  # ['true_message']
        except:
            try:
                images_datas_temp = json_file['true_message']['image_datas']
            except:
                images_datas_temp = json_file
        images_datas = images_datas + images_datas_temp
    return images_datas

def clean_data(img_data_dict):
    """
    清洗数据，清除掉无用横线框
    :param data_dict:
    :return:
    """
    if len(img_data_dict['shushi']) == 0:
        return

    new_text_dict_list = [] #img_data_dict['text'].copy()
    # tuoshi_dict_list = img_data_dict['tuoshi']
    # shushi_dict_list = img_data_dict['shushi']
    # jiefangcheng_dict_list = img_data_dict['jiefangcheng']

    for text_dict in img_data_dict['text']:
        is_dirty = False
        for shushi_dict in img_data_dict['shushi']:
            if polygon_riou(shushi_dict['bbox'], text_dict['bbox']) > 0.5:
                if bool(re.search(r'\d', text_dict['context'])) is False:
                    is_dirty = True
                    continue

        if is_dirty is False:
            new_text_dict_list.append(text_dict)

    img_data_dict['text'] = new_text_dict_list


def parser_jsons(jsons_dir):

    def handle_data(true_box, show_img, text_dict_list, tuoshi_dict_list, shushi_dict_list, jiefangcheng_dict_list):
        true_box = true_box.astype(np.int32).reshape([8, ])

        data_dict = {}
        data_dict['bbox'] = true_box.tolist()
        data_dict['label'] = state
        data_dict['context'] = mark_data['marked_text']

        if state == 'text':
            text_dict_list.append(data_dict)
            # cv2.polylines(show_img,
            #               [true_box.reshape((-1, 1, 2))],
            #               True,
            #               (0, 255, 0))
        elif state == 'shushi':
            shushi_dict_list.append(data_dict)
            cv2.polylines(show_img,
                          [true_box.reshape((-1, 1, 2))],
                          True,
                          (255, 255, 0))
        elif state == 'tuoshi':
            tuoshi_dict_list.append(data_dict)
            cv2.polylines(show_img,
                          [true_box.reshape((-1, 1, 2))],
                          True,
                          (0, 255, 255))
        elif state == 'jiefangcheng':
            jiefangcheng_dict_list.append(data_dict)
            cv2.polylines(show_img,
                          [true_box.reshape((-1, 1, 2))],
                          True,
                          (255, 0, 255))

    json_files_path = os.listdir(jsons_dir)
    jsons_dir_name = jsons_dir.split('/')[-1]
    images_data = []
    for json_file_name in json_files_path:
        if 'json' in json_file_name:
            json_path = os.path.join(RAW_JSONS_DIR, json_file_name)
            json_data = json.loads(open(json_path, encoding='utf-8').read(),
                                   encoding='bytes')
            if 'true_message' in json_data:
                images_data_temp = json_data['true_message']['image_datas']
            else:
                images_data_temp = json_data['image_datas']
            images_data += images_data_temp
    index_name = 0
    for img_data in tqdm(images_data):
        index_name += 1
        mark_datas = img_data['mark_datas']
        if len(mark_datas) == 1:
            continue
        url = img_data['pic_url']
        img_name = img_data['pic_name'].split('/')[-1]#jsons_dir_name + '_' + str(index_name) + '.jpg'
        # print(img_name)
        try:
            img_save_path = os.path.join(IMG_DIR, img_name)
            request.urlretrieve(url, img_save_path)
            try:
                angle = int(img_data['rotate_angle'])
            except:
                angle = 0
            img_data_dict = {}
            text_dict_list = []
            tuoshi_dict_list = []
            shushi_dict_list = []
            jiefangcheng_dict_list = []

            if True:
                if angle == 180:
                    origin = Image.open(img_save_path)
                    rotated = origin.rotate(180, expand=1)
                    rotated.save(os.path.join(img_save_path))
                    width, height = rotated.size
                    show_img = cv2.imread(img_save_path)
                    for iter, mark_data in enumerate(mark_datas):
                        flag, state= label_is_text_box(mark_data)
                        if flag is None:
                            continue
                        box = get_rect(mark_data['marked_path'].split())
                        true_box = np.array([1 - box[4], 1 - box[5], 1 - box[6], 1 - box[7],
                                             1 - box[0], 1 - box[1], 1 - box[2], 1 - box[3]]).reshape([4, 2])
                        true_box[:, 0] = true_box[:, 0] * width
                        true_box[:, 1] = true_box[:, 1] * height
                        handle_data(true_box, show_img, text_dict_list, tuoshi_dict_list, shushi_dict_list, jiefangcheng_dict_list)

                    # cv2.imwrite(os.path.join(SHOW_DIR, img_name), show_img)
                elif angle == 90:
                    origin = Image.open(img_save_path)
                    rotated = origin.rotate(270, expand=1)
                    rotated.save(os.path.join(img_save_path))
                    width, height = rotated.size
                    show_img = cv2.imread(img_save_path)

                    for iter, mark_data in enumerate(mark_datas):
                        flag, state = label_is_text_box(mark_data)
                        if flag is None:
                            continue
                        box = get_rect(mark_data['marked_path'].split())
                        true_box = np.array([1 - box[7], box[6], 1 - box[1], box[0],
                                             1 - box[3], box[2], 1 - box[5], box[4]]).reshape([4, 2])
                        true_box[:, 0] = true_box[:, 0] * width
                        true_box[:, 1] = true_box[:, 1] * height
                        handle_data(true_box, show_img, text_dict_list, tuoshi_dict_list, shushi_dict_list,
                                    jiefangcheng_dict_list)

                    # cv2.imwrite(os.path.join(SHOW_DIR, img_name), show_img)
                elif angle == 270:
                    origin = Image.open(img_save_path)
                    rotated = origin.rotate(90, expand=1)
                    rotated.save(os.path.join(img_save_path))
                    width, height = rotated.size
                    show_img = cv2.imread(img_save_path)

                    for iter, mark_data in enumerate(mark_datas):
                        flag, state = label_is_text_box(mark_data)
                        if flag is None:
                            continue
                        box = get_rect(mark_data['marked_path'].split())
                        true_box = np.array([box[3], 1 - box[2], box[5], 1 - box[4],
                                             box[7], 1 - box[6], box[1], 1 - box[0]]).reshape([4, 2])
                        true_box[:, 0] = true_box[:, 0] * width
                        true_box[:, 1] = true_box[:, 1] * height
                        handle_data(true_box, show_img, text_dict_list, tuoshi_dict_list, shushi_dict_list,
                                    jiefangcheng_dict_list)

                    # cv2.imwrite(os.path.join(SHOW_DIR, img_name), show_img)
                elif angle == 0:
                    img = cv2.imread(img_save_path)
                    height, width, _ = img.shape
                    show_img = cv2.imread(img_save_path)

                    for iter, mark_data in enumerate(mark_datas):
                        flag, state = label_is_text_box(mark_data)
                        if flag is None:
                            continue
                        box = get_rect(mark_data['marked_path'].split())
                        true_box = np.array([box[0], box[1], box[2], box[3],
                                             box[4], box[5], box[6], box[7]]).reshape([4, 2])
                        true_box[:, 0] = true_box[:, 0] * width
                        true_box[:, 1] = true_box[:, 1] * height
                        handle_data(true_box, show_img, text_dict_list, tuoshi_dict_list, shushi_dict_list,
                                    jiefangcheng_dict_list)

                    # cv2.imwrite(os.path.join(SHOW_DIR, img_name), show_img)
                else:
                    print(img_name, 'has error angle')
                    continue
                img_data_dict['text'] = text_dict_list
                img_data_dict['tuoshi'] = tuoshi_dict_list
                img_data_dict['shushi'] = shushi_dict_list
                img_data_dict['jiefangcheng'] = jiefangcheng_dict_list

                clean_data(img_data_dict)

                for text_dict in img_data_dict['text']:
                    cv2.polylines(show_img,
                                  [np.array(text_dict['bbox']).reshape((-1, 1, 2))],
                                  True,
                                  (0, 255, 0))
                cv2.imwrite(os.path.join(SHOW_DIR, img_name), show_img)

                base_name = img_name.split('.')[0]
                with open(os.path.join(JSON_DIR, base_name + '.json'), 'w') as f:
                    f.writelines(json.dumps(img_data_dict, indent=4, ensure_ascii=False))

        except Exception as e:
            import traceback
            traceback.print_exc()


def check_label(jsons_dir):
    json_files_path = os.listdir(jsons_dir)

    images_data = []
    for json_file_name in json_files_path:
        if 'json' in json_file_name:
            print(json_file_name)
            json_path = os.path.join(jsons_dir, json_file_name)
            print(json_path)
            json_data = json.loads(open(json_path, encoding='utf-8').read(),
                                   encoding='bytes')
            if 'true_message' in json_data:
                images_data_temp = json_data['true_message']['image_datas']
            else:
                images_data_temp = json_data['image_datas']
            images_data += images_data_temp
    labels = []
    text = []
    for img_data in tqdm(images_data):
        mark_datas = img_data['mark_datas']
        # print('mark_datas', mark_datas)
        if len(mark_datas) == 1:
            continue
        for mark_data in mark_datas:
            for label in mark_data['label']:
                if label['children'] not in labels:
                    labels.append(label['children'])

    print(labels)
    # print(text)


if __name__ == "__main__":
    RAW_JSONS_DIR_list = ['/share/pipeline_version/json_data/0710',
                          '/share/pipeline_version/json_data/0712',
                          '/share/pipeline_version/json_data/0724',
                          '/share/pipeline_version/json_data/0731',
                          '/share/pipeline_version/json_data/0807',
                          '/share/pipeline_version/json_data/0814',
                          '/share/pipeline_version/json_data/0821',
                          '/share/pipeline_version/json_data/0904',
                         ]
    # import glob
    # json_dir = '/share/zzh/q2_raw_data/*'
    # RAW_JSONS_DIR_list = glob.glob(os.path.join(json_dir))

    for RAW_JSONS_DIR in RAW_JSONS_DIR_list:
        name = RAW_JSONS_DIR.split('/')[-1]
        print('parser ', name)

        SAVE_DIR = os.path.join('/share/zzh/test_data', name)
        IMG_DIR = os.path.join(SAVE_DIR, 'imgs')
        JSON_DIR = os.path.join(SAVE_DIR, 'jsons')
        SHOW_DIR = os.path.join(SAVE_DIR, 'show')
        make_dir(IMG_DIR)
        make_dir(JSON_DIR)
        make_dir(SHOW_DIR)
        parser_jsons(RAW_JSONS_DIR)

    # check_label('/home/tony/ikkyyu/data/high')

    # xingchen jincheng 脱式' in label or '解方程' in label or '脱式' in label or '竖式' in label

# [['横式'], ['竖式'], ['解题过程'], ['脱式'], ['横式', '解题过程'], ['解题过程', '文本'], ['脱式', '解题过程'], ['解方程', '解题过程'], ['竖式', '解题过程'], ['横式', '文本'], ['横式', '竖式'], ['文本'], ['解方程'], ['不确定']]
# [['横式'], ['解题过程'], ['竖式'], ['脱式'], ['横式', '文本'], ['文本'], ['解方程']]
# [['横式'], ['横式', '文本'], ['解题过程'], ['竖式'], ['脱式'], ['解题过程', '文本'], ['文本'], ['竖式', '文本']]
#

