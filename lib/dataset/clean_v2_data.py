import os
import shutil
import glob
import cv2
import tqdm
import json
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPoint  # 多边形

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


def mask_img(img, mask_data):
    for mask in mask_data:
        # print(mask)
        im = np.ones(img.shape[:2], dtype="uint8")
        mask = np.array(mask['bbox'], np.int32).reshape([1, 4, 2])
        cv2.polylines(im, mask, 1, 0)
        cv2.fillPoly(im, mask, 0)
        img[:, :, 0] = img[:, :, 0] * im
        img[:, :, 1] = img[:, :, 1] * im
        img[:, :, 2] = img[:, :, 2] * im
    return img


def clean_data(json_path):
    """
    清洗数据
    :param json_path:
    :return:
    """
    json_data = json.loads(open(json_path).read())

    text_dict_list = json_data['text']

    mask_list = []
    if 'shushi' in json_data.keys():
        shushi_boxes = json_data['shushi']

        for shushi_box in shushi_boxes:
            num = 0
            box_in_shushi_list = []
            for box in text_dict_list:
                if polygon_riou(shushi_box['bbox'], box['bbox']) > 0.3:
                    num += 1
                    box_in_shushi_list.append(box)

            if len(box_in_shushi_list) < 3:
                mask_list.append(shushi_box)
                for box in box_in_shushi_list:
                    # print('remove ', box)
                    text_dict_list.remove(box)

    json_data['mask_data'] = mask_list
    json_data['text'] = text_dict_list

    if len(text_dict_list) is 0:
        return None, True

    if len(mask_list) != 0:
        return json_data, True
    else:
        return None, False


if __name__ == "__main__":


    root_dir = '/hostpersistent/zzh/dataset/text/parser_data/q2_data'

    dir_list = glob.glob(os.path.join(root_dir, '*'))


    file_op = open('/hostpersistent/zzh/ikkyyu/clean_data2.txt', 'w')

    for data_dir in dir_list:

        img_dir = os.path.join(data_dir, 'imgs')
        json_dir = os.path.join(data_dir, 'jsons')
        show_dir = os.path.join(data_dir, 'show')

        img_list = os.listdir(img_dir)
        print(data_dir)
        for img_name in tqdm.tqdm(img_list):
            basename = os.path.splitext(img_name)[0]

            json_path = os.path.join(json_dir, basename + '.json')

            json_data, is_dirty = clean_data(json_path)

            if is_dirty:
                if json_data is None:
                    # 删除图片,删除json,删除show
                    os.remove(os.path.join(img_dir, img_name))
                    os.remove(json_path)
                    os.remove(os.path.join(show_dir, img_name))
                    file_op.writelines('remove,' + json_path + '\n')
                else:
                    with open(json_path, 'w') as f:
                        f.writelines(json.dumps(json_data, indent=4, ensure_ascii=False))
                    img = cv2.imread(os.path.join(img_dir, img_name))
                    img = mask_img(img, json_data['mask_data'])
                    show_img = img.copy()
                    for text in json_data['text']:
                        cv2.polylines(show_img, [np.array(text['bbox']).reshape((-1, 1, 2))], True, (0, 255, 0))

                    cv2.imwrite(os.path.join(img_dir, img_name), img)
                    cv2.imwrite(os.path.join(show_dir, img_name), show_img)

                    file_op.writelines('mask,' + json_path + '\n')

    file_op.close()