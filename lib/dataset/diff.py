import os
import shutil
import tqdm

#RAW_JSONS_DIR_list = ['/share/zzh/aug_data/pdf/']
'''['/share/zzh/parser_data/q1_data',
                          '/share/zzh/parser_data/0723-badcase1',
                          '/share/zzh/parser_data/0723-badcase2',
                          '/share/zzh/parser_data/0723-badcase3',
                          '/share/zzh/parser_data/0723-badcase4',
                          '/share/zzh/parser_data/0723-badcase5',
                          '/share/zzh/parser_data/0723-badcase6',
                          '/share/zzh/parser_data/0726_jincheng',
                        '/share/zzh/parser_data/0726_jincheng2',
                          '/share/zzh/parser_data/0726_lingbo',
                          '/share/zzh/parser_data/0726_qirui',
                          '/share/zzh/parser_data/danwei1',
                          '/share/zzh/parser_data/danwei2',
                          '/share/zzh/parser_data/jincheng1',
                          '/share/zzh/parser_data/xingchen1'
                         ]'''

import glob

dirs = '/hostpersistent/zzh/dataset/text/parser_data/q2_data/*'
RAW_JSONS_DIR_list = ['/home/tony/houmian/par_q2/0']#glob.glob(os.path.join(dirs))

for data_dir in RAW_JSONS_DIR_list:
    print('start to handle ', data_dir)
    img_dir = os.path.join(data_dir, 'imgs')
    json_dir = os.path.join(data_dir, 'jsons')

    img_name_list = os.listdir(img_dir)
    json_name_list = os.listdir(json_dir)

    if len(img_name_list) != len(json_dir):
        for img_name in tqdm.tqdm(img_name_list):
            base_name = img_name.split('.')[0]
            json_name = base_name + '.json'
            if json_name not in json_name_list:
                print('remove ', os.path.join(img_dir, img_name))
                os.remove(os.path.join(img_dir, img_name))
    else:
        print('true')
