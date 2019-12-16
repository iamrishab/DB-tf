import os
import cv2
import time
import numpy as np
import tensorflow as tf
from db_config import cfg

from lib.postprocess.post_process import SegDetectorRepresenter
import lib.networks.model as model

class DB():

    def __init__(self, ckpt_path):
        tf.reset_default_graph()
        self._input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        self._binarize_map, self._threshold_map, self._thresh_binary = model.model(self._input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        gpu_config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options, allow_soft_placement=True)
        self.sess = tf.Session(config=gpu_config)
        saver.restore(self.sess, ckpt_path)
        self.decoder = SegDetectorRepresenter()
        print('restore model from:', ckpt_path)

    def __del__(self):
        self.sess.close()

    def detect_img(self, img):
        h, w, _ = img.shape
        resized_img, (ratio_h, ratio_w) = self._resize_img(img)

        binarize_map, threshold_map, thresh_binary = self.sess.run([self._binarize_map, self._threshold_map, self._thresh_binary],
                                                                   feed_dict={self._input_images: [resized_img]})

        cv2.imwrite('binarize_map.jpg', binarize_map[0]*255)
        cv2.imwrite('threshold_map.jpg', threshold_map[0]*255)
        cv2.imwrite('thresh_binary.jpg', thresh_binary[0]*255)

        boxes, scores = self.decoder([img], binarize_map, threshold_map, thresh_binary, True)
        # print(info)
        print(boxes)
        for box in boxes:
            cv2.polylines(img, [np.array(box).astype(np.int).reshape([-1, 1, 2])], True, (0, 255, 0))
        cv2.imwrite('show.jpg', img)


    def detect_batch(self, batch):
        pass

    def _resize_img(self, img, max_size=736):
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


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    db = DB('/hostpersistent/zzh/lab/DB-tf/ckpt/1216_DB_model.ckpt-10101')

    img = cv2.imread('/hostpersistent/zzh/dataset/open_data/ctw1500/test/text_image/1039.jpg')
    db.detect_img(img)

    # s = time.time()
    # for i in range(50):
    #     db.detect_img(img)
    # print((time.time()-s)/50)