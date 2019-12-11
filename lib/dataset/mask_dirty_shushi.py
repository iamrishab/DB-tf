import os
import json
import numpy as np
import shapely
import tqdm
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


def mask_dirty_shushi(json_path):
    json_data = json.loads(open(json_path).read())

    text_dict_list = json_data['text']

    mask_list = []
    mask_text_dict_list = []
    if 'shushi' in json_data.keys():
        shushi_boxes = json_data['shushi']

        for shushi_box in shushi_boxes:
            num = 0
            box_in_shushi_list = []
            for box in text_dict_list:
                if polygon_riou(shushi_box['bbox'], box['bbox']) > 0.6:
                    num += 1
                    box_in_shushi_list.append(box)


            if len(box_in_shushi_list) < 3:
                mask_list.append(shushi_box)
                for box in box_in_shushi_list:
                    print('remove ', box)
                    text_dict_list.remove(box)



    json_data['mask_data'] = mask_list
    json_data['text'] = text_dict_list

    if len(mask_list) != 0:
        print(json_path)
        return json_data, True
    else:
        return json_data, False


if __name__ == "__main__":
    import glob
    # data_dirs = '/share/zzh/parser_data'

    data_dir_list = glob.glob(os.path.join('/hostpersistent/zzh/dataset/text/parser_data/q2_data/*'))
    # data_dir_list = ['/hostpersistent/zzh/dataset/text/parser_data/q1_data/']
    json_data = []
    num = 0
    for data_dir in data_dir_list:
        if not os.path.isdir(data_dir):
            continue
        print('handle ', data_dir)
        data_abs_dir = data_dir
        # img_dir = os.path.join(data_abs_dir, 'imgs')
        json_dir = os.path.join(data_abs_dir, 'jsons')

        json_data_list = os.listdir(json_dir)

        for json_name in tqdm.tqdm(json_data_list):
            json_data, is_mask = mask_dirty_shushi(os.path.join(json_dir, json_name))
            if is_mask:
                num += 1
            with open(os.path.join(json_dir, json_name), 'w') as f:
                f.writelines(json.dumps(json_data, indent=4, ensure_ascii=False))

    print('total num', num)

