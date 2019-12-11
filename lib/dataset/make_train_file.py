# coding:utf-8
import os
import json
import numpy as np
import tqdm


if __name__ == "__main__":

    # data_dirs = '/share/zzh/parser_data/new_1017_manual'
    # data_dirs = '/hostpersistent/zzh/dataset/text/parser_data/1017_manual'

    # data_dirs = '/share/zzh/parser_data/new_1017_manual'
    # #
    # data_dir_list = os.listdir(data_dirs)
    # print(data_dir_list)
    # if '2big_600' in data_dir_list:
    #     print('remove 2big_600')
    #     data_dir_list.remove('2big_600')
    # if '10timg_pot_75' in data_dir_list:
    #     print('remove 10timg_pot_75')
    #     data_dir_list.remove('10timg_pot_75')
    # if '11timg_pot_space_75' in data_dir_list:
    #     print('remove 11timg_pot_space_75')
    #     data_dir_list.remove('11timg_pot_space_75')

    #
    # data_dir_list = [
    #     #'0723-badcase1', '0723-badcase2', '0723-badcase3', '0723-badcase4',
    #     #             '0723-badcase5', '0723-badcase6', '0726_jincheng', '0726_jincheng2',
    #     #             '0726_lingbo', '0726_qirui', '0808_3000', 'danwei1', 'danwei2', 'jincheng1',
    #     #              'xingchen1',
    #                  'badcase_aug', 'badcase',
    #     # 'long_long',
    #     # '1017_shushi', '1017_tihao',
    #     #'0909_12156',
    #     # 'q1_data',
    #     # '1017_wangge',
    #     # '1017_aug',
    #     # 'long_8k',
    #     # 'unit_convert',
    #     # 'goux',
    # ]
    import glob
    # data_dir_list = glob.glob(os.path.join('/hostpersistent/zzh/dataset/text/parser_data/q2_data', '*'))
    # data_dir_list = glob.glob(os.path.join('/hostpersistent/zzh/dataset/text/parser_data/1027_15', '*'))
    data_dir_list = ['/hostpersistent/zzh/dataset/text/parser_data/1114']
    data_dirs = ''

    json_data = []
    for data_dir in data_dir_list:
        # json_data = []

        if not os.path.isdir(os.path.join(data_dirs, data_dir)):
            continue
        print('handle ', os.path.join(data_dirs, data_dir))
        data_abs_dir = os.path.join(data_dirs, data_dir)
        img_dir = os.path.join(data_abs_dir, 'imgs')
        json_dir = os.path.join(data_abs_dir, 'jsons')

        img_list = os.listdir(img_dir)
        np.random.shuffle(img_list)
        half = int(len(img_list) / 3)
        use_list = img_list#[0:4000]

        for img_name in tqdm.tqdm(use_list):
            img_path = os.path.join(img_dir, img_name)
            label_name = os.path.splitext(img_name)[0] + '.json'
            label_path = os.path.join(json_dir, label_name)

            if os.path.exists(img_path) and os.path.exists(label_path):
                dict = {}
                dict['img_path'] = img_path
                dict['label_path'] = label_path
                json_data.append(dict)
    name = '1114_24k'#'1027_15_' + os.path.split(data_dir)[-1]
    with open(os.path.join('/hostpersistent/zzh/dataset/text/train_file/', name + '.json'), 'w') as f:
        f.writelines(json.dumps(json_data, indent=4, ensure_ascii=False))
'''
    img_dir = '/share/zzh/parser_data/v3_0916_12k/imgs'
    label_dir = '/share/zzh/parser_data/v3_0916_12k/jsons'

    img_list = os.listdir(img_dir)
    np.random.shuffle(img_list)
    json_data = []
    use_list = img_list
    for img_name in tqdm.tqdm(use_list):

        img_path = os.path.join(img_dir, img_name)
        label_name = img_name.split('.')[0] + '.json'
        label_path = os.path.join(label_dir, label_name)

        if os.path.exists(img_path) and os.path.exists(label_path):
            dict = {}
            dict['img_path'] = img_path
            dict['label_path'] = label_path
            json_data.append(dict)

    with open('/share/zzh/train_json/0916/0916_12k_train.json', 'w') as f:
        f.writelines(json.dumps(json_data, indent=4, ensure_ascii=False))
'''