# coding:utf-8
import os
import json
import cv2
import shutil
import tqdm
import numpy as np

from PIL import Image, ImageDraw

import xml.etree.ElementTree as ET

def parser_xml(xml_path):
    # 解析xml
    tree = ET.parse(xml_path)
    root = tree.getroot()
    zoom = float(root.attrib.get('zoom'))

    json_data = {}
    text_dict_list = []

    for child in root:
        img_data_dict = {}
        if 'area' in child.tag:
            # print('into point...')
            points_str = child.attrib.get('points')
            if not len(points_str) > 0:
                # print('point none.')
                continue
            points_str_list = points_str.split(';')
            bbox = []
            for e_point in points_str_list:
                if not len(e_point) > 0:
                    break
                xy = e_point.replace('(', '').replace(')', '').strip().split(',')
                try:
                    x = int(int(xy[0]) * zoom)
                    y = int(int(xy[1]) * zoom)
                except:
                    try:
                        x = int(float(xy[0]) * zoom)
                        y = int(float(xy[1]) * zoom)
                    except:
                        print('point error:', xy, '    :::', e_point)
                bbox.append(x)
                bbox.append(y)
            text = child.attrib.get('text')

        if 'rect' in child.tag:
            # print('into xyz...')
            x = int(float(child.attrib.get('x')) * zoom)
            y = int(float(child.attrib.get('y')) * zoom)
            w = int(float(child.attrib.get('w')) * zoom)
            h = int(float(child.attrib.get('h')) * zoom)
            x1 = x + w
            y1 = y + h
            text = child.attrib.get('text')
            bbox = [x, y, x1, y, x1, y1, x, y1]

        # xy_list = np.reshape(np.array(bbox), (4, 2))
        img_data_dict['bbox'] = bbox
        img_data_dict['context'] = text
        img_data_dict['label'] = 'text'
        text_dict_list.append(img_data_dict)
    json_data['text'] = text_dict_list
    return json_data

def resize_img(img, max_side_len=720):
    """
    将图像进行缩放,最大边大于2400,按照最大边进行resize,然后判断每个边是否能够被32整除,再进行一次resize
    :param im:
    :param max_side_len:
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

    return resized_img, (ratio_h, ratio_w)

if __name__ == '__main__':
    data_dir = '/home/tony/houmian/0808_3000'
    save_img_dir = '/home/tony/houmian/par_08080/imgs'
    save_json_dir = '/home/tony/houmian/par_08080/jsons'
    save_show_dir = '/home/tony/houmian/par_08080/show'

    file_list = os.listdir(data_dir)

    for file_name in tqdm.tqdm(file_list):
        file_type = file_name.split('.')[-1]

        if file_type not in ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']:
            continue

        xml_name = file_name.split('.')[0] + '.xml'
        ss = xml_name + ' not exists'
        assert os.path.exists(os.path.join(data_dir, xml_name)), ss
        json_name = file_name.split('.')[0] + '.json'
        json_data = parser_xml(os.path.join(data_dir, xml_name))



        img = cv2.imread(os.path.join(data_dir, file_name))[:,:,::-1]

        resized_img, (ratio_h, ratio_w) = resize_img(img)
        cv2.imwrite(os.path.join(save_img_dir, file_name), resized_img)
        showimg = resized_img.copy()
        # print(showimg.shape)
        for data in json_data['text']:
            data_np = np.array(data['bbox'][0:8]).reshape([4, 2]).astype(np.float32)
            data_np[:, 0] *= ratio_w
            data_np[:, 1] *= ratio_h
            data['bbox'] = data_np.reshape([8,]).astype(np.int).tolist()

            cv2.polylines(showimg, [np.array(data['bbox']).reshape((-1, 1, 2))], True, (0, 255, 0))
        # cv2.rectangle(showimg, (0, 100), (200,200), (255, 0, 0), 5)
        cv2.imwrite(os.path.join(save_show_dir, file_name), showimg)

        # shutil.copyfile(os.path.join(data_dir, file_name), os.path.join(save_img_dir, file_name))
        with open(os.path.join(save_json_dir, json_name), 'w') as f:
            f.write(json.dumps(json_data, indent=4, ensure_ascii=False))
