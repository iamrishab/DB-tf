import os
import json
import glob
import tqdm
import glob


all_data_dir_list = ['/hostpersistent/zzh/dataset/text/parser_data/q1_data',
                    '/hostpersistent/zzh/dataset/text/parser_data/v3_data/v3_0916',
                    '/hostpersistent/zzh/dataset/text/parser_data/v3_data/v3_0909',
                    '/hostpersistent/zzh/dataset/text/parser_data/q1_data',
                     ]
all_data_dir_list.extend(glob.glob(os.path.join('/hostpersistent/zzh/dataset/text/parser_data/q2_data', '*')))


shushi_dict = []
tuoshi_dict = []
jiefangcheng = []

for data_dir in all_data_dir_list:
    print('handle ', data_dir)

    json_dir = os.path.join(data_dir, 'jsons')
    img_dir = os.path.join(data_dir, 'imgs')

    json_list = os.listdir(json_dir)

    for json_name in tqdm.tqdm(json_list):
        json_data = json.loads(open(os.path.join(json_dir, json_name), encoding='utf-8').read(),
                               encoding='bytes')

        if 'tuoshi' in json_data.keys():
            if len(json_data['tuoshi']) != 0:
                info = {}
                basename = json_name.split('.')[0]
                info['img_path'] = glob.glob(os.path.join(img_dir, basename + '*'))[0]
                info['label_path'] = os.path.join(json_dir, json_name)
                tuoshi_dict.append(info)
                continue

        if 'shushi' in json_data.keys():
            if len(json_data['shushi']) != 0:
                info = {}
                basename = json_name.split('.')[0]
                info['img_path'] = glob.glob(os.path.join(img_dir, basename + '*'))[0]
                info['label_path'] = os.path.join(json_dir, json_name)
                shushi_dict.append(info)
                continue

        if 'jiefangcheng' in json_data.keys():
            if len(json_data['jiefangcheng']) != 0:
                info = {}
                basename = json_name.split('.')[0]
                info['img_path'] = glob.glob(os.path.join(img_dir, basename + '*'))[0]
                info['label_path'] = os.path.join(json_dir, json_name)
                jiefangcheng.append(info)
                continue


with open(os.path.join('/hostpersistent/zzh/dataset/text/train_file/', 'shushi_' + str(len(shushi_dict)) + '_.json'), 'w') as f:
    f.writelines(json.dumps(shushi_dict, indent=4, ensure_ascii=False))
with open(os.path.join('/hostpersistent/zzh/dataset/text/train_file/', 'tuoshi_' + str(len(tuoshi_dict)) + '_.json'), 'w') as f:
    f.writelines(json.dumps(tuoshi_dict, indent=4, ensure_ascii=False))
with open(os.path.join('/hostpersistent/zzh/dataset/text/train_file/', 'jiefangcheng_' + str(len(jiefangcheng)) + '_.json'), 'w') as f:
    f.writelines(json.dumps(jiefangcheng, indent=4, ensure_ascii=False))
    

