import os
import shutil
import random
import tqdm

# 创建测试数据
#
# data_dir = '/share/zzh/parser_data/1017_data/1017_nets'
#
# test_dir = '/share/zzh/east_data/test_data/v2/1017_test'
#
# img_list = os.listdir(os.path.join(data_dir, 'imgs'))
#
# random.shuffle(img_list)
#
# for img_name in tqdm.tqdm(img_list[:50]):
#     shutil.move(os.path.join(data_dir, 'imgs', img_name),
#                 os.path.join(test_dir, 'imgs', img_name))
#
#     json_name = img_name.split('.')[0] + '.json'
#     shutil.move(os.path.join(data_dir, 'jsons', json_name),
#                 os.path.join(test_dir, 'jsons', json_name))
#
#     shutil.move(os.path.join(data_dir, 'show', img_name),
#                 os.path.join(test_dir, 'show', img_name))

# 移动文件夹中数据
# all_data_dir = '/share/zzh/parser_data/new_1017_manual/'
#
# data_dir_list = ['goux', 'unit_convert']#os.listdir(all_data_dir)
#
# test_dir = '/share/zzh/east_data/test_data/v2/1017_manual'
#
# for data_dir in data_dir_list:
#
#     data_dir = os.path.join(all_data_dir, data_dir)
#
#     img_list = os.listdir(os.path.join(data_dir, 'imgs'))
#
#     random.shuffle(img_list)
#
#     # need_num = int(len(img_list) * 0.083)
#
#
#     for img_name in tqdm.tqdm(img_list[:10]):
#         shutil.move(os.path.join(data_dir, 'imgs', img_name),
#                     os.path.join(test_dir, 'imgs', img_name))
#
#         json_name = img_name.split('.')[0] + '.json'
#         shutil.move(os.path.join(data_dir, 'jsons', json_name),
#                     os.path.join(test_dir, 'jsons', json_name))
#
#         shutil.move(os.path.join(data_dir, 'show', img_name),
#                     os.path.join(test_dir, 'show', img_name))



#
# # 数据回传
#
# test_dir = '/share/zzh/east_data/test_data/v2/1017_manual'
# all_data_dir = '/share/zzh/parser_data/1017_manual/'
#
#
# img_list = os.listdir(os.path.join(test_dir, 'imgs'))
#
# for img_name in tqdm.tqdm(img_list):
#     basename = os.path.splitext(img_name)[0]
#     json_name = basename + '.json'
#
#     data_dir = json_name.replace('_' + basename.split('_')[-1] + '.json', '')
#
#     shutil.move(os.path.join(test_dir, 'imgs', img_name),
#                 os.path.join(all_data_dir, data_dir, 'imgs', img_name))
#
#     # json_name = img_name.split('.')[0] + '.json'
#     shutil.move(os.path.join(test_dir, 'jsons', json_name),
#                 os.path.join(all_data_dir, data_dir, 'jsons', json_name))
#
#     shutil.move(os.path.join(test_dir, 'show', img_name),
#                 os.path.join(all_data_dir, data_dir, 'show', img_name))


# 移动测试数据

save_dir = '/share/zzh/aug_data/badcase'

org_dir = '/share/zzh/test_data'
date = '0731'

img_name_list = ['6c31e4ac-a9c3-11e9-81aa-00163e0a34a0.jpg']
    # ['7f47e102-a547-11e9-9bc4-00163e3060a3.jpg',
    #              'f8f80032-a94c-11e9-8b8c-00163e2ec503.jpg',
    #              ]
    # ['57a7f4d4-af8d-11e9-9c03-00163e3060a3.jpg',
    #              '911e9b96-b032-11e9-910b-00163e0a7aba.jpg',
    #              'adaf8502-b278-11e9-9c10-00163e3060a3.jpg',
    #              'b853534a-af61-11e9-b7e6-00163e30dd5d.jpg']
    # ['0904_01_3_15_.jpg', '0904_02_2_15_.jpg', '0904_02_5_12_.jpg',
    #              '0904_03_2_11_.jpg', '0904_03_2_14_.jpg', '0904_03_5_12_.jpg',
    #              '0904_02_3_8_.jpg']
#['0821_02_1_13_.jpg', '0821_02_1_2_.jpg', '0821_02_4_11_.jpg',
#                 '0821_02_3_7_.jpg', '0821_02_5_2_.jpg']

for img_name in img_name_list:

    shutil.copyfile(os.path.join(org_dir, date, 'imgs', img_name),
                    os.path.join(save_dir, 'imgs', img_name))

    json_name = os.path.splitext(img_name)[0] + '.json'
    shutil.copyfile(os.path.join(org_dir, date, 'jsons', json_name),
                os.path.join(save_dir, 'jsons', json_name))

    shutil.copyfile(os.path.join(org_dir, date, 'show', img_name),
                os.path.join(save_dir, 'show', img_name))

'''
0821_02_1_2_.jpg_2recognition.jpg
0821_02_4_11_.jpg_gt.jpg
0821_02_3_7_.jpg
0821_02_5_2_.jpg
0904_2/error/0904_02_3_8_.jpg_pred.jpg


/0724/error/7f47e102-a547-11e9-9bc4-00163e3060a3.jpg_2recognition.jpg
/0724/error/f8f80032-a94c-11e9-8b8c-00163e2ec503.jpg_pred.jpg
0731/error/6c31e4ac-a9c3-11e9-81aa-00163e0a34a0.jpg_2recognition.jpg
0807/error/57a7f4d4-af8d-11e9-9c03-00163e3060a3.jpg_gt.jpg
0807/error/911e9b96-b032-11e9-910b-00163e0a7aba.jpg_2recognition.jpg
0807/error/adaf8502-b278-11e9-9c10-00163e3060a3.jpg_2recognition.jpg
0807/error/b853534a-af61-11e9-b7e6-00163e30dd5d.jpg_2recognition.jpg
0821_2/error/0821_02_1_13_.jpg_pred.jpg
0904_1/error/0904_01_3_15_.jpg_pred.jpg
0904_2/error/0904_02_2_15_.jpg_2recognition.jpg
0904_2/error/0904_02_5_12_.jpg_2recognition.jpg
0904_3/error/0904_03_2_11_.jpg_pred.jpg
0904_3/error/0904_03_2_14_.jpg_pred.jpg
0904_3/error/0904_03_5_12_.jpg_2recognition.jpg


'''