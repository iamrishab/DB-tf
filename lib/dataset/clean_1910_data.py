import os
import glob
import tqdm
import json

# 检查图片中是否有大框

data_dir = '/hostpersistent/zzh/dataset/text/parser_data/1027_15/*'

all_dirs = glob.glob(os.path.join(data_dir))

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