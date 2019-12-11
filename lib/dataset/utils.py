import os
import shutil
import cv2
import json
import tqdm
import random
import numpy as np
import glob
#
#
# def find_none_hand(img_path, json_dir):
#     img_name = img_path.split('/')[-1]
#     json_path = os.path.join(json_dir, img_name.split('.')[0] + '.json')
#     assert os.path.exists(json_path), 'file not exits'.format(json_path)
#     data = json.loads(open(json_path, encoding='utf-8').read(), encoding='bytes')
#
#     text_dict_list = data['text']
#     none_hand = 0
#     hand = 0
#     for boxes_info in text_dict_list:
#         if '$' not in boxes_info['context']:
#             none_hand += 1
#         else:
#             hand += 1
#
#         if none_hand ==10 or hand ==10:
#             break
#
#     if none_hand >=10:
#         return True
#     else:
#         return False
#
#
# img_dir = '/share/zzh/parser_data/0808_3000/imgs/'
# json_dir = '/share/zzh/parser_data/0808_3000/jsons'
# img_name_list = os.listdir(img_dir)
#
# none_hand_list = []
# hand_list = []
# for img_name in tqdm.tqdm(img_name_list):
#     if find_none_hand(os.path.join(img_dir, img_name), json_dir):
#         none_hand_list.append(img_name)
#     else:
#         hand_list.append(img_name)
# # print(none_hand_list)
# print(len(none_hand_list))
#
# with open('/share/zzh/0808_none_hand.json', 'w') as f:
#     f.writelines(json.dumps(none_hand_list, indent=4, ensure_ascii=False))
# with open('/share/zzh/0808_hand.json', 'w') as f:
#     f.writelines(json.dumps(hand_list, indent=4, ensure_ascii=False))
#

# org_dir = '/share/test_data/org'
#
# move_dir = '/share/test_data/online_badcase/imgs'
#
# with open('/share/fuck.txt', 'r') as f:
#     lines = f.readlines()
#
#     for line in lines:
#         try:
#
#             info = line.split('/')
#             ver = info[6]
#             img_name = info[-1].replace('\n', '')
#             print(img_name)
#             shutil.copy(os.path.join(org_dir, ver , img_name), os.path.join(move_dir, img_name))
#         except Exception as e:
#             import traceback
#             traceback.print_exc()
#
#             print('error')
#             continue

# org_dir = '/share/test_data/online_badcase/res'
#
# remove_dir = '/share/test_data/east/online/0905'
#
# move_dir = '/share/test_data/online_badcase/0905_res'
#
# img_list = os.listdir(org_dir)
#
# dir_list = ['0710', '0712', '0724', '0731', '0807', '0814', '0821']
#
#
# def find_dir(dir_list, img_name, org_dir):
#     for dir in dir_list:
#         if os.path.exists(os.path.join(org_dir, dir, 'img', img_name)):
#             return os.path.join(org_dir, dir, 'img', img_name)
#     return None
#
#
# for img_name in img_list:
#     path = find_dir(dir_list, img_name, remove_dir)
#
#     if path is not None:
#         print('move', img_name)
#         shutil.copy(path, os.path.join(move_dir, img_name))


# data_dir = '/share/zzh/parser_data'
#
# dirs_list = os.listdir(data_dir)
#
# error_dir = []
# error_data = []
#
# i = 0
# for dirr in dirs_list:
#     img_dir = os.path.join(data_dir, dirr, 'imgs')
#     img_list = os.listdir(img_dir)
#
#     json_dir = os.path.join(data_dir, dirr, 'jsons')
#     json_list = os.listdir(json_dir)
#
#     for img_name in tqdm.tqdm(img_list):
#         basename = os.path.splitext(img_name)[0]
#         json_name = basename + '.json'
#         if not os.path.exists(os.path.join(json_dir, json_name)):
#             i += 1
#             dict = {}
#             dict['img_path'] = os.path.join(img_dir, img_name)
#             dict['label_path'] = os.path.join(json_dir, json_name)
#             error_data.append(dict)
#             # print('json name:', json_dir, '/',json_name)
#         if len(img_name.split('.'))>2:
#             error_dir.append(dirr)
# print(error_dir)
# print(i)
#
# with open(os.path.join('/share/zzh/error_label.json'), 'w') as f:
#     f.writelines(json.dumps(error_data, indent=4, ensure_ascii=False))


# data_dir = '/share/zzh/parser_data'
# dirs_list = os.listdir(data_dir)
#
# aug_img_dir = '/share/zzh/aug_data/net/imgs'
# aug_json_dir = '/share/zzh/aug_data/net/jsons'
#
#
# for dirr in dirs_list:
#     img_dir = os.path.join(data_dir, dirr, 'imgs')
#     json_dir = os.path.join(data_dir, dirr, 'jsons')
#
#     img_list = os.listdir(img_dir)
#     random.shuffle(img_list)
#     num = int(0.1 * len(img_list))
#
#     for img_name in tqdm.tqdm(img_list[0:num]):
#         shutil.copyfile(os.path.join(img_dir, img_name),
#                         os.path.join(aug_img_dir, img_name))
#
#         json_name = os.path.splitext(img_name)[0] + '.json'
#
#         shutil.copyfile(os.path.join(json_dir, json_name),
#                         os.path.join(aug_json_dir, json_name))

# 检查图片中是否有大框

data_dir = '/share/zzh/parser_data/new_1017_manual/*'

all_dirs = ['/hostpersistent/zzh/dataset/text/parser_data/1114']#glob.glob(os.path.join(data_dir))

error_data = {}
error_num = 0
i = 0
for dir in all_dirs:
    json_list = os.listdir(os.path.join(dir, 'jsons'))

    for json_file in tqdm.tqdm(json_list):
        json_data = json.loads(open(os.path.join(dir, 'jsons', json_file), encoding='utf-8').read(),
                                   encoding='bytes')
        i += 1
        text_text = json_data['text']
        tuoshi = json_data['tuoshi']
        shushi = json_data['shushi']
        jiefangcheng = json_data['jiefangcheng']
        handwritten = json_data['handwritten']

        if len(tuoshi) != 0 or len(shushi) != 0 or len(jiefangcheng) != 0 or len(handwritten) != 0:
            text_text.extend(tuoshi)
            text_text.extend(shushi)
            text_text.extend(jiefangcheng)
            text_text.extend(handwritten)
            if dir in error_data.keys():
                error_data[dir] += 1
            else:
                error_data[dir] = 1
            error_num += 1
            print(os.path.join(dir, 'jsons', json_file))
            with open(os.path.join(dir, 'jsons', json_file), 'w') as f:
                f.writelines(json.dumps(json_data, indent=4, ensure_ascii=False))
print(i)
print(error_data)
print(error_num)
#
# # 统计大图数目
#
# data_dir = '/share/zzh/parser_data/1017_data'
#
# dir_list = os.listdir(data_dir)
# # dir_list.remove('1017_data')
# # dir_list.remove('1017_manual')
# # dir_list.remove('q1_data')
#
#
# all_img = 0
# all_text = 0
#
# for dir_name in tqdm.tqdm(dir_list):
#     json_list = os.listdir(os.path.join(data_dir, dir_name, 'jsons'))
#
#     all_img += len(json_list)
#
#     for json_name in json_list:
#         json_data = json.loads(open(os.path.join(data_dir, dir_name, 'jsons', json_name), encoding='utf-8').read(),
#                                            encoding='bytes')
#
#         text_num = len(json_data['text'])
#         all_text += text_num
#
# print(all_img)
#
# print(all_text)

# 根据json画框
# data_dir = '/share/zzh/aug_data/badcase'
#
# img_list = os.listdir(os.path.join(data_dir, 'imgs'))
#
# for img_name in tqdm.tqdm(img_list):
#     img = cv2.imread(os.path.join(data_dir, 'imgs', img_name))
#
#     json_name = os.path.splitext(img_name)[0] + '.json'
#
#     json_data = json.loads(open(os.path.join(data_dir, 'jsons', json_name), encoding='utf-8').read(),
#                            encoding='bytes')
#
#     for text in json_data['text']:
#         # print(text)
#         cv2.polylines(img, [np.array(text['bbox']).reshape((-1, 1, 2))], True,
#                                   (0, 255, 0))
#
#     cv2.imwrite(os.path.join(data_dir, 'show', img_name), img)