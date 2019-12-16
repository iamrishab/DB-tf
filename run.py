import os
import cv2
import numpy as np
from lib.dataset.dataloader import load_labels, make_score_map, make_border_map


if __name__ =='__main__':
    img_dir = '/hostpersistent/zzh/dataset/open_data/ctw1500/train/text_image'
    label_dir = '/hostpersistent/zzh/dataset/open_data/ctw1500/train/text_label_curve'


    img_list = os.listdir(img_dir)
    label_list = os.listdir(label_dir)
    # np.random.shuffle(img_list)
    print(img_list[0])
    img = cv2.imread(os.path.join(img_dir, img_list[0]))
    h, w, _ = img.shape

    polys, tags = load_labels(os.path.join(label_dir, os.path.splitext(img_list[0])[0] + '.txt'))


    for poly in polys:
        poly = np.array(poly, dtype=np.int)
        cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, (0, 255, 0))

    threshold_map, thresh_mask = make_border_map(polys, tags, h, w)
    score_map, score_mask = make_score_map(polys, tags, h, w)

    cv2.imwrite('s.jpg', score_map*255)
    cv2.imwrite('t.jpg', threshold_map*255)
    cv2.imwrite('sm.jpg', score_mask*255)

    cv2.imwrite('o.jpg', img)