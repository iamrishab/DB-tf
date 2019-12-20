import os
import cv2
import tqdm
import time
import random
import numpy as np

from db_config import cfg
from lib.dataset.label_maker import make_border_map, make_score_map
from lib.dataset.generator_enqueuer import GeneratorEnqueuer
from lib.dataset.img_aug import crop_area, det_aug


def load_labels(gt_path):
    """
    load pts
    :param gt_path:
    :return: polys shape [N, 14, 2]
    """
    assert os.path.exists(gt_path), '{} is not exits'.format(gt_path)
    polys = []
    tags = []
    with open(gt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(',')
            x = float(parts[0])
            y = float(parts[1])
            pts = [float(i) for i in parts[4:32]]
            poly = np.array(pts) + [x, y] * 14
            polys.append(poly.reshape([-1, 2]))
            tags.append(False)
    return np.array(polys, np.float), tags


def resize_img(img, max_size=736):
    h, w, _ = img.shape

    if max(h, w) > max_size:
        ratio = float(max_size) / h if h > w else float(max_size) / w
    else:
        ratio = 1.

    resize_h = int(ratio * h)
    resize_w = int(ratio * w)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resized_img = cv2.resize(img, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return resized_img, (ratio_h, ratio_w)


def make_train_labels(polys, tags, h, w):
    """

    :param polys: numpy [N, 2]
    :param tags:
    :param h:
    :param w:
    :return:
    """

    threshold_map, thresh_mask = make_border_map(polys, tags, h, w)
    score_map, score_mask = make_score_map(polys, tags, h, w)

    return score_map, score_mask, threshold_map, thresh_mask

def generator(batchsize, random_scale=np.array(cfg.TRAIN.IMG_SCALE)):

    img_list = os.listdir(cfg.TRAIN.IMG_DIR)

    while True:
        train_imgs = []
        train_score_maps = []
        train_socre_masks = []
        train_thresh_maps = []
        train_thresh_masks = []

        np.random.shuffle(img_list)

        for img_name in img_list:
            try:
                img_path = os.path.join(cfg.TRAIN.IMG_DIR, img_name)
                label_path = os.path.join(cfg.TRAIN.LABEL_DIR, os.path.splitext(img_name)[0] + '.txt')

                img_input = np.zeros([cfg.TRAIN.IMG_SIZE, cfg.TRAIN.IMG_SIZE, 3], dtype=np.float32)

                img = cv2.imread(img_path)[:,:, ::-1]
                img, (ratio_h, ratio_w) = resize_img(img, cfg.TRAIN.IMG_SIZE)

                if random.random() < cfg.TRAIN.DATA_AUG_PROB:
                    img = det_aug(img)

                polys, tags = load_labels(label_path)
                polys[:, :, 0] *= ratio_w
                polys[:, :, 1] *= ratio_h

                if random.random() < cfg.TRAIN.CROP_PROB:
                    img, polys, tags = crop_area(img, polys, tags)
                    img, (ratio_h, ratio_w) = resize_img(img, cfg.TRAIN.IMG_SIZE)
                    polys[:, :, 0] *= ratio_w
                    polys[:, :, 1] *= ratio_h

                h, w, _ = img.shape
                img_input[:h, :w, :] = img
                h, w, _ = img_input.shape

                score_map, score_mask, threshold_map, thresh_mask = make_train_labels(polys, tags, h, w)

                train_imgs.append(img_input)
                train_score_maps.append(score_map[:, :, np.newaxis])
                train_socre_masks.append(score_mask[:, :, np.newaxis])
                train_thresh_maps.append(threshold_map[:, :, np.newaxis])
                train_thresh_masks.append(thresh_mask[:, :, np.newaxis])

                if len(train_imgs) == batchsize:
                    yield train_imgs, train_score_maps, train_socre_masks, train_thresh_maps, train_thresh_masks
                    train_imgs = []
                    train_score_maps = []
                    train_socre_masks = []
                    train_thresh_maps = []
                    train_thresh_masks = []

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(img_path)
                # print(polys[0])
                # img_input = img_input.astype(np.int)
                # for poly in polys:
                #     poly = np.array(poly, dtype=np.int)
                #     cv2.polylines(img_input, [poly.reshape((-1, 1, 2))], True, (0, 255, 0))
                # cv2.imwrite(img_name, img_input)
                continue


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        print('Generator use 10 batches for buffering, this may take a while, you can tune this yourself.')
        enqueuer.start(max_queue_size=10, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


if __name__ =='__main__':
    img_dir = '/Users/zhangzihao/AI/research/datasets/ctw1500/train/text_image'
    label_dir = '/Users/zhangzihao/AI/research/datasets/ctw1500/train/text_label_curve'


    img_list = os.listdir(img_dir)
    label_list = os.listdir(label_dir)
    # np.random.shuffle(img_list)
    print(img_list[0])
    img = cv2.imread(os.path.join(img_dir, img_list[0]))
    h, w, _ = img.shape
    polys, tags = load_labels(os.path.join(label_dir, os.path.splitext(img_list[0])[0] + '.txt'))
    threshold_map, thresh_mask = make_border_map(polys, tags, h, w)
    score_map, score_mask = make_score_map(polys, tags, h, w)

    #
    # for poly in polys:
    #     poly = np.array(poly, dtype=np.int)
    #     cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, (0, 255, 0))
    #
    #
    # threshold_map, thresh_mask = make_border_map(polys, tags, h, w)
    #
    # s = time.time()
    # score_map, score_mask = make_score_map(polys, tags, h, w)
    # print(time.time()-s)
    #
    # cv2.imwrite('s.jpg', score_map*255)
    # cv2.imwrite('t.jpg', threshold_map*255)
    # cv2.imwrite('sm.jpg', score_mask*255)
    #
    # cv2.imwrite('o.jpg', img)
