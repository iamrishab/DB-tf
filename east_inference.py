# -*- coding: utf-8 -*-
import cv2
import time
import os
import json
import shutil
import numpy as np
import pprint
import tensorflow as tf

import lib.network.model as model
from lib.dataset.data_utils import restore_rectangle

import lib.lanms.adaptor as lanms
from east_config import cfg
import tqdm

from lib.lanms.locality_aware_nms import nms_locality
from nms import bboxes_nms, bboxes_sort

def make_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

class East(object):

    def __init__(self, CKPT_PATH):
        """
        推理阶段
        :return:
        """
        self.is_lanms = True
        self.clock = False  # 是否输出各模块运行时间
        tf.reset_default_graph()
        self.input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        # 调用模型函数,返回分数和几何
        self.f_score, self.f_geometry = model.model(self.input_images, is_training=False)

        # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均的函数来获取平局值了。
        # 采用滑动平均的方法更新参数,衰减速率（decay），用于控制模型的更新速度,迭代的次数
        # 指数加权平均的求法，公式 total = a * total + (1 - a) * next
        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        # 保存和恢复都需要实例化一个Saver
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        # 添加对显存的限制 add by boby 20190603
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        gpu_config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options, allow_soft_placement=True)
        self.sess = tf.Session(config=gpu_config)

        # 重载模型的参数，继续训练或用于测试数据
        saver.restore(self.sess, CKPT_PATH)

    def __del__(self):
        self.sess.close()

    def test(self, img_path, reslut_txt_dir, reslut_img_dir, is_lanms=True, write_img=True):
        def post_boxes(boxes):
            scores = None
            if boxes is not None:
                scores = boxes[:, 8]
                boxes = boxes[:, :8].reshape((-1, 4, 2))
                boxes[:, :, 0] /= ratio_w
                boxes[:, :, 1] /= ratio_h

            result_list = []
            if boxes is not None:
                for box in boxes:
                    if cfg.EXP_BOX.is_exp:
                        box = self.__exp_box(box.reshape((8,))).reshape((4, 2), cfg.EXP_BOX.right_ratio,
                                                                        cfg.EXP_BOX.lrft_ratio)
                        box = self.__correct_box(box, h, w)
                    box.astype(np.int)
                    dic = [int(box[0, 0]), int(box[0, 1]), int(box[1, 0]), int(box[1, 1]),
                           int(box[2, 0]), int(box[2, 1]), int(box[3, 0]), int(box[3, 1])]
                    # dic = int(box.reshape((8,)).tolist())
                    result_list.append(dic)
            return result_list, scores

        def draw_boxes(boxes, color):
            if boxes is not None:
                for box in boxes:
                    # box = self.__sort_poly(box.astype(np.int32))
                    # if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    #     continue
                    cv2.polylines(img[:, :, ::-1], [np.array(box).astype(np.int32).reshape((-1, 1, 2))],
                                  True, color=color, thickness=2)
            else:
                pass

        self.is_lanms = is_lanms
        # 将cv读取的BGR->RGB
        if not os.path.exists(img_path):
            print('img is not exist:' + img_path)
            return
        img = cv2.imread(img_path)[:, :, ::-1]
        h, w, _ = img.shape
        resized_img, (ratio_h, ratio_w) = self.__resize_img(img)
        self.scale_width = w*ratio_w
        timer = {'net': 0, 'restore': 0, 'nms': 0}
        # 返回当前时间的时间戳
        start = time.time()

        score, geometry = self.sess.run([self.f_score, self.f_geometry], feed_dict={self.input_images: [resized_img]})

        timer['net'] = time.time() - start
        # 检测
        boxes, score_rgb_map, timer = self.__detect(score_map=score, geo_map=geometry, timer=timer)
        boxes, scores = post_boxes(boxes)

        if not os.path.exists(reslut_txt_dir):
            os.makedirs(reslut_txt_dir)

        json_data = {}
        json_data['text'] = []
        for text_box in boxes:
            dict_ = {'bbox': text_box}
            json_data['text'].append(dict_)

        file_basename = os.path.basename(img_path).split('.')[0]
        with open(os.path.join(reslut_txt_dir, file_basename + '.json'), 'w') as f:
            f.writelines(json.dumps(json_data, indent=4, ensure_ascii=False))

        if write_img:
            if reslut_img_dir is None:
                reslut_img_dir = reslut_txt_dir

            draw_boxes(boxes, (0, 255, 0))
            img_name = os.path.basename(img_path).split('.')[0]
            img_path = os.path.join(reslut_img_dir, os.path.basename(img_path))
            cv2.imwrite(img_path, img[:, :, ::-1])

            score_path = os.path.join(reslut_img_dir, img_name + '_score.jpg')

            if score_rgb_map is not None:
                h, w, c = score_rgb_map.shape
                # print(h,w)
                # print(img.shape)
                img = img[:, :, ::-1]
                score_rgb_map = cv2.resize(score_rgb_map, (int(w / ratio_w), int(h / ratio_h)))
                # print(score_rgb_map.shape)
                # print(img[np.where(score_rgb_map[:, :, 0] == 255)])
                img = cv2.resize(img, (int(w / ratio_w), int(h / ratio_h)))
                img[np.where(score_rgb_map[:, :, 0] == 255)] = img[np.where(score_rgb_map[:, :, 0] == 255)]*0.5 + [0, 122, 0]
                cv2.imwrite(score_path, img)

    def __detect(self, score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.01, lanms_thres=cfg.LANMS_THRESHOLD, nms_thres=cfg.NMS_THRESHOLD):
        """
        restore text boxes from score map and geo map
        :param score_map:
        :param geo_map:
        :param timer:
        :param score_map_thresh: threshhold for score map
        :param box_thresh: threshhold for boxes
        :param nms_thres: threshold for nms
        :return:
        """
        # input()
        if len(score_map.shape) == 4:
            score_map = score_map[0, :, :, 0]
            geo_map = geo_map[0, :, :, ]

        filer_pixel = np.where(score_map < score_map_thresh)
        need_pixel = np.where(score_map > score_map_thresh)
        score_map[filer_pixel] = 0
        score_map[need_pixel] = 1

        score_rgb_map = cv2.cvtColor(score_map * 255, cv2.COLOR_GRAY2RGB)
        # filter the score map
        xy_text = np.argwhere(score_map > score_map_thresh)
        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]

        if self.clock:
            start = time.time()

        # (xy_text[:, ::-1] * 4).shape [n,2] 乘以4原因是将图片缩放到512
        # geo_map[xy_text[:, 0], xy_text[:, 1], :] 得到过滤后的每个像素点的坐标和角度
        text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
        # text_box_restored shape = [N, 4,2]
        # print('{} text boxes before nms'.format(text_box_restored.shape[0]))

        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]

        if self.clock:
            timer['restore'] = time.time() - start

        if self.clock:
            start = time.time()
        # nms part
        # boxes = nms_locality(boxes.astype(np.float64), self.scale_width, lanms_thres)
        if self.is_lanms:
            boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), lanms_thres, nms_thres, self.scale_width, cfg.LONG_MERGE_THRESHOLD)
        else:
            boxes = bboxes_sort(boxes[:, 8], boxes)
            boxes = bboxes_nms(boxes)
        boxes = np.array(boxes)

        if self.clock:
            timer['nms'] = time.time() - start
            # print('{} text boxes after nms'.format(boxes.shape[0]))
        if boxes.shape[0] == 0:
            return None, None, timer

        # here we filter some low score boxes by the average score map, this is different from the orginal paper
        for i, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
            boxes[i, 8] = cv2.mean(score_map, mask)[0]
        boxes = boxes[boxes[:, 8] > box_thresh]

        return boxes, score_rgb_map, timer

    def __resize_img(self, img, max_side_len=cfg.INPUT_MAX_SIZE):
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

    def __sort_points(self, xy_np):
        """
        返回四个点顺序
        :param xy_list:
        :return:
        """
        xy_list = xy_np.reshape(4, 2).tolist()
        sort_x_list = sorted(xy_list, key=lambda x: x[0])
        left_point = sort_x_list[0:2]
        right_point = sort_x_list[2:4]

        sort_left_y_list = sorted(left_point, key=lambda x: x[1])
        sort_right_y_list = sorted(right_point, key=lambda x: x[1])

        left_top_p = sort_left_y_list[0]
        left_down_p = sort_left_y_list[1]

        right_top_p = sort_right_y_list[0]
        right_down_p = sort_right_y_list[1]

        new_point_list = [left_top_p, right_top_p, right_down_p, left_down_p]

        return np.array(new_point_list)

    def __correct_box(self, boxes_np, img_height, img_width):
        """
        修正box范围，防止box超出图片边界范围
        :param boxes_np: [-1, 4, 2]
        :param img_height:
        :param img_width:
        :return:
        """
        # 首先检查x,y是否小于0
        index_x_zero = np.where(boxes_np[:, 0] < 0)
        if len(index_x_zero[0]) != 0:
            boxes_np[index_x_zero[0], 0] = 0
        index_y_zero = np.where(boxes_np[:, 1] < 0)
        if len(index_y_zero[0]) != 0:
            boxes_np[index_y_zero[0], 1] = 0

        # 检查x,y是否超出图像边界框
        index_x_img = np.where(boxes_np[:, 0] > img_width)
        if len(index_x_img[0]) != 0:
            boxes_np[index_x_img[0], 0] = img_width
        index_y_img = np.where(boxes_np[:, 1] > img_height)
        if len(index_y_img[0]) != 0:
            boxes_np[index_y_img[0], 1] = img_height

        return boxes_np

    def __exp_box(self, box, right_threshold=0.02, left_threshold=0.01):
        # index = [2, 3, 0, 1, 6, 7, 4, 5]#[0, 1, 0, 1, 6, 7, 6, 7]
        #
        # vector = box - box[index]
        # new_box = threshold * vector + box
        # return new_box
        right_index = [0, 1, 0, 1, 6, 7, 6, 7]  #
        left_index = [2, 3, 2, 3, 4, 5, 4, 5]
        # right
        vector = box - box[right_index]
        new_box = right_threshold * vector + box

        # left
        vector = new_box - new_box[left_index]
        new_box = left_threshold * vector + new_box

        return new_box


    # 调用检测接口
    def east_process(self, img,  is_lanms=True):
        """

        :param img:
        :param is_lanms:
        :return: list shape [N, 8]
        """
        h, w, _ = img.shape
        self.is_lanms = is_lanms
        resized_img, (ratio_h, ratio_w) = self.__resize_img(img)

        self.scale_width = w*ratio_w

        timer = {'net': 0, 'restore': 0, 'nms': 0}
        if self.clock:
            start = time.time()

        score, geometry = self.sess.run([self.f_score, self.f_geometry], feed_dict={self.input_images: [resized_img]})

        if self.clock:
            end = time.time()
            timer['net'] = end - start
            print("net:{}".format(timer['net']))

        # 检测
        boxes, score_rgb_map, timer = self.__detect(score_map=score, geo_map=geometry, timer=timer)

        # 打印执行时间
        if self.clock:
            print('net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(timer['net'] * 1000,
                                                                        timer['restore'] * 1000,
                                                                        timer['nms'] * 1000))

        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h
        if self.clock:
            duration = time.time() - start
            print('[timing] {}'.format(duration))

        result_list = []
        if boxes is not None:
            for box in boxes:
                # to avoid submitting errors
                if cfg.EXP_BOX.is_exp:
                    box = self.__exp_box(box.reshape((8,))).reshape((4, 2), cfg.EXP_BOX.right_ratio, cfg.EXP_BOX.lrft_ratio)
                    box = self.__correct_box(box, h, w)
                dic = [int(box[0, 0]), int(box[0, 1]), int(box[1, 0]), int(box[1, 1]),
                       int(box[2, 0]), int(box[2, 1]),int(box[3, 0]), int(box[3, 1])]
                result_list.append(dic)
        return result_list


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    # CKPT_PATH = '/share/pipeline_version/V1.1.5/pipeline_for_monkey/Models/east/east_model.ckpt-36364'
    # CKPT_PATH = '/share/zzh/east_data/ckpt/online_models/V0.9.05.ckpt'
    # CKPT_PATH = '/home/tony/ikkyyu/east_models/0905/V0.9.05.ckpt'
    # CKPT_PATH = '/root/east_data/ckpt/east_model.ckpt-4042'
    CKPT_PATH = '/hostpersistent/zzh/ikkyyu/train_data/1107/ckpt/east_model.ckpt-442381'
    # CKPT_PATH = '/home/tony/ikkyyu/east_models/0929/V0.9.29'

    east = East(CKPT_PATH)
    # 将cv读取的BGR->RGB
    # temp_img = cv2.imread(img_path)[:, :, ::-1]
    # res = east.east_process(temp_img)
    # print(res)
    # 测试使用接口
    # img_dir = '/home/tony/ikkyyu/test/test/imgs'
    # txt_result_dir = '/home/tony/ikkyyu/test/test/res'
    # img_result_dir = '/home/tony/ikkyyu/test/test/res'

    # 测试文件夹文件夹中的图片
    # test_dir = '/share/test_data/org'
    # test_result_dir = '/share/test_data/east/test/0925-62621'
    # # test_result_dir = '/share/test_data/east/online/0905'
    # dir_list = os.listdir(test_dir)
    #
    # for dir in dir_list:
    #     img_dir = os.path.join(test_dir, dir)
    #
    #     img_name_list = os.listdir(img_dir)
    #     save_dir = os.path.join(test_result_dir, dir)
    #
    #     txt_result_dir = os.path.join(save_dir, 'txt')
    #     img_result_dir = os.path.join(save_dir, 'img')
    #
    #     make_dir(txt_result_dir)
    #     make_dir(img_result_dir)
    #     # print(img_result_dir)
    #     # print(txt_result_dir)
    #     print('test ', dir)
    #     for img_name in tqdm.tqdm(img_name_list):
    #         east.test(os.path.join(img_dir, img_name), txt_result_dir, img_result_dir)

    # 测试文件夹图片
    img_dir = '/hostpersistent/zzh/ikkyyu/test/imgs'
    img_result_dir = '/hostpersistent/zzh/ikkyyu/test/res'
    #
    # img_dir = '/home/tony/data/test-50/data/imgs'
    # img_result_dir = '/home/tony/data/test-50/data/0929_0'
    #
    # img_dir = '/home/tony/ikkyyu/test_data/test/imgs'
    # img_result_dir = '/home/tony/ikkyyu/test_data/test/res'
    #
    # img_dir = '/share/test_data/online_badcase/imgs'
    # img_result_dir = '/share/test_data/online_badcase/res'
    #
    # img_dir = '/home/tony/ikkyyu/test_data/temp/imgs'
    # img_result_dir = '/home/tony/ikkyyu/test_data/temp/0929-res-nonms'
    #
    # img_dir = '/home/tony/ikkyyu/test_data/edge/imgs'
    # img_result_dir = '/home/tony/ikkyyu/test_data/edge/0929_lanms'


    make_dir(img_result_dir)

    img_list = os.listdir(img_dir)
    for img_name in tqdm.tqdm(img_list):
        east.test(os.path.join(img_dir, img_name), img_result_dir, img_result_dir)
    pprint.pprint(cfg)

    # import glob
    # data_dirs = '/share/zzh/parser_data/1017_manual/'
    # save_dir = '/share/temp/1017_manual/'
    # data_dir_list = ['/share/zzh/east_data/test_data/v2/1017_test']#glob.glob(os.path.join(data_dirs, '*'))
    #
    # for data_dir in data_dir_list:
    #     print(data_dir)
    #
    #     dir_name = os.path.split(data_dir)[-1]
    #
    #     img_dir = os.path.join(data_dir, 'imgs')
    #
    #     img_result_dir = os.path.join(save_dir, dir_name)
    #
    #     make_dir(img_result_dir)
    #
    #     img_list = os.listdir(img_dir)
    #     for img_name in tqdm.tqdm(img_list):
    #         east.test(os.path.join(img_dir, img_name), img_result_dir, img_result_dir)
