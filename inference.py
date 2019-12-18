import os
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf
from db_config import cfg


from lib.postprocess.post_process import SegDetectorRepresenter
import lib.networks.model as model


def get_args():
    parser = argparse.ArgumentParser(description='DB-tf')
    parser.add_argument('--ckptpath', default='/hostpersistent/zzh/lab/DB-tf/ckpt_1217/1216_DB_model.ckpt-10101', type=str,
                        help='load model')
    parser.add_argument('--imgpath', default='/hostpersistent/zzh/dataset/open_data/ctw1500/test/text_image/1039.jpg',
                        type=str)
    parser.add_argument('--gpuid', default='0',
                        type=str)
    parser.add_argument('--ispoly', default=True,
                        type=bool)

    args = parser.parse_args()

    return args

class DB():

    def __init__(self, ckpt_path):
        tf.reset_default_graph()
        self._input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        self._binarize_map, self._threshold_map, self._thresh_binary = model.model(self._input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
        gpu_config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options, allow_soft_placement=True)
        self.sess = tf.Session(config=gpu_config)
        saver.restore(self.sess, ckpt_path)
        self.decoder = SegDetectorRepresenter()
        print('restore model from:', ckpt_path)

    def __del__(self):
        self.sess.close()

    def detect_img(self, img_path, ispoly=True, show_res=True):
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        resized_img, (ratio_h, ratio_w) = self._resize_img(img)

        s = time.time()
        binarize_map, threshold_map, thresh_binary = self.sess.run([self._binarize_map, self._threshold_map, self._thresh_binary],
                                                                   feed_dict={self._input_images: [resized_img]})

        net_time = time.time()-s

        s = time.time()
        boxes, scores = self.decoder([img], binarize_map, ispoly)

        post_time = time.time()-s

        if show_res:
            img_name = os.path.splitext(os.path.split(img_path)[-1])[0]
            cv2.imwrite(img_name + '_binarize_map.jpg', binarize_map[0]*255)
            cv2.imwrite(img_name + '_threshold_map.jpg', threshold_map[0]*255)
            cv2.imwrite(img_name + '_thresh_binary.jpg', thresh_binary[0]*255)
            for box in boxes[0]:

                cv2.polylines(img, [box.astype(np.int).reshape([-1, 1, 2])], True, (0, 255, 0))
            cv2.imwrite(img_name + '_show.jpg', img)

        return net_time, post_time


    def detect_batch(self, batch):
        pass

    def _resize_img(self, img, max_size=320):
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

        input_img = np.zeros([max_size, max_size, 3])
        input_img[0:resize_h, 0:resize_w, :] = resized_img

        return input_img, (ratio_h, ratio_w)


if __name__ == "__main__":
    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    db = DB(args.ckptpath)

    db.detect_img(args.imgpath, args.ispoly)

    # img_list = os.listdir('/hostpersistent/zzh/dataset/open_data/ctw1500/test/text_image/')

    # net_all = 0
    # post_all = 0
    # pipe_all = 0
    #
    # for i in img_list[0:50]:
    #     net_time, post_time = db.detect_img(os.path.join('/hostpersistent/zzh/dataset/open_data/ctw1500/test/text_image/',i), args.ispoly, show_res=False)
    #     net_all += net_time
    #     post_all += post_time
    #     pipe_all += (net_time + post_time)
    #
    # print('net:', net_all/50)
    # print('post:', post_all/50)
    # print('pipe:', pipe_all/50)