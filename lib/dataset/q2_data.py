import os
import json
import shutil
import tqdm

q2_dir = '/share/dataset/src_data/q2'

save_dir = '/share/zzh/q2_raw_data'

dir_list = os.listdir(q2_dir)
dir_list.remove('第二批')

for i, dir_name in enumerate(dir_list):
    print(dir_name)
    json_dir = os.path.join(save_dir, str(i))
    os.makedirs(json_dir)

    time_list = os.listdir(os.path.join(q2_dir, dir_name))

    for time_dir in tqdm.tqdm(time_list):
        json_list = os.listdir(os.path.join(q2_dir, dir_name, time_dir, 'fhjsons'))

        for json_name in json_list:
            shutil.copyfile(os.path.join(q2_dir, dir_name, time_dir, 'fhjsons', json_name),
                            os.path.join(json_dir, json_name))