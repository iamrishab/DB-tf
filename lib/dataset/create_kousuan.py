# coding:utf-8
import os
import cv2
import numpy as np
import json
import shutil
import random
import tqdm


def load_annoataion(label_file, ratio_h, ratio_w):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    try:
        boxes_dict = json.loads(open(os.path.join(label_file), encoding='utf-8').read(), encoding='bytes')
        text_boxes_list = boxes_dict['text']

        boxes_list = []
        boxes_context_list = []

        for box_dict in text_boxes_list:
            box = np.array(box_dict['bbox']).reshape(-1, 2)
            # 转为最小包围矩形
            rect = cv2.minAreaRect(box)
            box = cv2.boxPoints(rect)
            box[:,0] = box[:,0] * ratio_w
            box[:,1] = box[:,1] * ratio_h

            boxes_list.append(box)
            boxes_context_list.append(box_dict['context'])

        return np.array(boxes_list, dtype=np.int).reshape((-1, 4, 2)), boxes_context_list
    except :
        print('errror label ', label_file)
        return None


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

def pinjie(img1_path, img2_path, label1_path, label2_path):
    """
    将两张图进行拼接
    :param img1_path:
    :param img2_path:
    :param label1_path:
    :param label2_path:
    :return:
    """
    img1 = cv2.imread(img1_path)[:, :, ::-1]
    img1, (ratio_h1, ratio_w1) = resize_img(img1)

    img2 = cv2.imread(img2_path)[:, :, ::-1]
    img2, (ratio_h2, ratio_w2) = resize_img(img2)

    data1 = load_annoataion(label1_path, ratio_h1, ratio_w1)
    data2 = load_annoataion(label2_path, ratio_h2, ratio_w2)



    if data1 is None or data2 is None:
        return None

    boxes1, contexts1 = data1
    boxes2, contexts2 = data2

    img1_shape = img1.shape
    img2_shape = img2.shape

    new_h = max(img1_shape[0], img2_shape[0])
    new_img = np.zeros((new_h, img1_shape[1] + img2_shape[0], 3))
    new_img[0:img1_shape[0], 0:img1_shape[1], :] = img1
    new_img[0:img2_shape[0], img1_shape[1]:img1_shape[1]+img2_shape[1], :] = img2
    # print(boxes2)
    boxes2[:, :, 0] += img1_shape[1]

    return (new_img, (boxes1, contexts1, boxes2, contexts2))


def padding(img_path, label_path):
    img = cv2.imread(img_path)[:, :, ::-1]
    img, (ratio_h, ratio_w) = resize_img(img)

    data = load_annoataion(label_path, ratio_h, ratio_w)

    if data is None:
        return None

    boxes, contexts = data
    img_h, img_w, _ = img.shape
    new_h = int(img_h * (1 + random.random()*0.5))
    new_w = int(img_w * (1 + random.random()*0.5))
    new_img = np.zeros([new_h, new_w, 3])

    rand_w = int(random.random() * (new_w - img_w))

    rand_h = int(random.random() * (new_h - img_h))

    new_img[rand_h:rand_h+img_h, rand_w:rand_w+img_w, :] = img

    boxes[:, :, 0] += rand_w
    boxes[:, :, 1] += rand_h

    return (new_img, boxes, contexts)

def make_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


if __name__ == "__main__":
    img_dir = '/hostpersistent/zzh/dataset/text/parser_data/1017_manual/tihao/imgs'
    label_dir = '/hostpersistent/zzh/dataset/text/parser_data/1017_manual/tihao/jsons'

    # data_list = json.loads(open(os.path.join('/share/zzh/train_json/1017_data.json'), encoding='utf-8').read(), encoding='bytes')
    # hand_list = json.loads(open(os.path.join('/hostpersistent/zzh/dataset/text/parser_data/0808_kousuan/hand.json'), encoding='utf-8').read(), encoding='bytes')
    # none_hand_list = json.loads(open(os.path.join('/hostpersistent/zzh/dataset/text/parser_data/0808_kousuan/nohand.json'), encoding='utf-8').read(), encoding='bytes')
    #
    # np.random.shuffle(hand_list)
    # np.random.shuffle(none_hand_list)

    save_img_dir = '/hostpersistent/zzh/dataset/text/aug_data/tihao_2k/imgs'
    save_label_dir = '/hostpersistent/zzh/dataset/text/aug_data/tihao_2k/jsons'
    save_show_dir = '/hostpersistent/zzh/dataset/text/aug_data/tihao_2k/show'

    make_dir(save_img_dir)
    make_dir(save_show_dir)
    make_dir(save_label_dir)

    # none_hand_list.extend(hand_list[0:300])
    #
    use_list = os.listdir(img_dir)

    # img_list = os.listdir(img_dir)
    #
    # use_list = img_list
    np.random.shuffle(use_list)
    index = 0
    for i in tqdm.tqdm(range(2000)):
        index += 1
        # 拼接
        if random.random()>0.5:
            img1_name = use_list[random.randint(0, len(use_list)-1)]
            img2_name = use_list[random.randint(0, len(use_list)-1)]
            label1_name = img1_name.split('.')[0] + '.json'
            label2_name = img2_name.split('.')[0] + '.json'
            if not os.path.exists(os.path.join(img_dir, img1_name)) or not os.path.exists(os.path.join(img_dir, img2_name)):
                continue
            data = pinjie(os.path.join(img_dir, img1_name),
                          os.path.join(img_dir, img2_name),
                          os.path.join(label_dir, label1_name),
                          os.path.join(label_dir, label2_name))
            if data is not None:
                img, labels = data
                base_name = 'pinjie_' + str(i)#'pinjie_' + img1_name.split('.')[0] + '_' + img2_name.split('.')[0] + '_' + str(i)
                cv2.imwrite(os.path.join(save_img_dir, base_name + '.jpg'), img)
                show_img = img.copy()
                boxes1, contexts1, boxes2, contexts2 = labels

                img_data_dict = {}
                text_dict_list = []
                for i, boxes in enumerate(boxes1):
                    data_dict = {}
                    data_dict['bbox'] = boxes.reshape((8,)).tolist()
                    data_dict['label'] = 'text'
                    data_dict['context'] = contexts1[i]
                    text_dict_list.append(data_dict)
                    # print(boxes)
                    cv2.polylines(show_img,
                                  [boxes.reshape((-1, 1, 2))],
                                  True,
                                  (0, 255, 0))
                for i, boxes in enumerate(boxes2):
                    data_dict = {}
                    data_dict['bbox'] = boxes.reshape((8,)).tolist()
                    data_dict['label'] = 'text'
                    data_dict['context'] = contexts2[i]
                    text_dict_list.append(data_dict)
                    cv2.polylines(show_img,
                                  [boxes.reshape((-1, 1, 2))],
                                  True,
                                  (0, 255, 0))
                img_data_dict['text'] = text_dict_list
                with open(os.path.join(save_label_dir, base_name + '.json'), 'w') as f:
                    f.writelines(json.dumps(img_data_dict, indent=4, ensure_ascii=False))
                cv2.imwrite(os.path.join(save_show_dir, base_name + '.jpg'), show_img)
        else:
            # padding
            img_name = use_list[random.randint(0, len(use_list))-1]
            label_name = img_name.split('.')[0] + '.json'
            #
            if not os.path.exists(os.path.join(img_dir, img_name)):
                continue

            data = padding(os.path.join(img_dir, img_name),
                           os.path.join(label_dir, label_name))

            if data is not None:
                img, boxes, contexts = data
                base_name = 'padding_' + str(i)#+ img_name.split('.')[0] + '_' + str(index)
                cv2.imwrite(os.path.join(save_img_dir, base_name + '.jpg'), img)
                show_img = img.copy()

                # boxes, contexts = label

                img_data_dict = {}
                text_dict_list = []
                for i, boxes in enumerate(boxes):
                    data_dict = {}
                    data_dict['bbox'] = boxes.reshape((8,)).tolist()
                    data_dict['label'] = 'text'
                    data_dict['context'] = contexts[i]
                    text_dict_list.append(data_dict)
                    cv2.polylines(show_img,
                                  [boxes.reshape((-1, 1, 2))],
                                  True,
                                  (0, 255, 0))

                img_data_dict['text'] = text_dict_list
                with open(os.path.join(save_label_dir, base_name + '.json'), 'w') as f:
                    f.writelines(json.dumps(img_data_dict, indent=4, ensure_ascii=False))
                cv2.imwrite(os.path.join(save_show_dir, base_name + '.jpg'), show_img)






