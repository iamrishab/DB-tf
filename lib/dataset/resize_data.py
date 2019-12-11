import os
import numpy as np
import cv2
import json
import tqdm


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
    data_dir = '/hostpersistent/zzh/dataset/text/parser_data/pdf_0927/'

    save_img_dir = '/hostpersistent/zzh/dataset/text/parser_data/pdf_data/imgs'
    save_json_dir = '/hostpersistent/zzh/dataset/text/parser_data/pdf_data/jsons'
    save_show_dir = '/hostpersistent/zzh/dataset/text/parser_data/pdf_data/show'

    img_list = os.listdir(os.path.join(data_dir, 'imgs'))

    for img_name in tqdm.tqdm(img_list):

        json_name = os.path.splitext(img_name)[0] + '.json'

        img = cv2.imread(os.path.join(data_dir, 'imgs', img_name))[:,:,::-1]

        resized_img, (ratio_h, ratio_w) = resize_img(img)
        cv2.imwrite(os.path.join(save_img_dir, img_name), resized_img)
        showimg = resized_img.copy()


        json_data = json.loads(open(os.path.join(data_dir, 'jsons', json_name), encoding='utf-8').read(), encoding='bytes')
        # print(showimg.shape)
        for data in json_data['text']:
            data_np = np.array(data['bbox'][0:8]).reshape([4, 2]).astype(np.float32)
            data_np[:, 0] *= ratio_w
            data_np[:, 1] *= ratio_h
            data['bbox'] = data_np.reshape([8,]).astype(np.int).tolist()

            cv2.polylines(showimg, [np.array(data['bbox']).reshape((-1, 1, 2))], True, (0, 255, 0))
        # cv2.rectangle(showimg, (0, 100), (200,200), (255, 0, 0), 5)
        cv2.imwrite(os.path.join(save_show_dir, img_name), showimg)

        # shutil.copyfile(os.path.join(data_dir, file_name), os.path.join(save_img_dir, file_name))
        with open(os.path.join(save_json_dir, json_name), 'w') as f:
            f.write(json.dumps(json_data, indent=4, ensure_ascii=False))
