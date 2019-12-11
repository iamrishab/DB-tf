import os, shutil
import numpy as np
from lib.prepare_training_data.parse_tal_xml import ParseXml
class_name = ['dontcare', 'handwritten', 'print']

val_img_dir = "/home/tony/ocr/ocr_dataset/ctpn/val_data/img"
val_xml_dir = "/home/tony/ocr/ocr_dataset/ctpn/val_data/xml"
save_dir = "data/mAP/ground_truth"

if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir)


xml_list = os.listdir(val_xml_dir)

for xml in xml_list:
    if "xml" in xml:

        xml_file = os.path.join(val_xml_dir, xml)
        txt_file = xml.split('.')[0] + ".txt"
        parser = ParseXml(xml_file)
        _, class_list, bbox_list = parser.get_bbox_class()
        with open(os.path.join(save_dir, txt_file), 'w') as f:
            print(os.path.join(save_dir, txt_file))
            for bbox_index in range(len(bbox_list)):
                if len(bbox_list[bbox_index]) == 8:
                    xmin = int(np.floor(
                        min(bbox_list[bbox_index][0], bbox_list[bbox_index][2], bbox_list[bbox_index][4],
                            bbox_list[bbox_index][6])))
                    ymin = int(np.floor(
                        min(bbox_list[bbox_index][1], bbox_list[bbox_index][3], bbox_list[bbox_index][5],
                            bbox_list[bbox_index][7])))
                    xmax = int(np.ceil(
                        max(bbox_list[bbox_index][0], bbox_list[bbox_index][2], bbox_list[bbox_index][4],
                            bbox_list[bbox_index][6])))
                    ymax = int(np.ceil(
                        max(bbox_list[bbox_index][1], bbox_list[bbox_index][3], bbox_list[bbox_index][5],
                            bbox_list[bbox_index][7])))
                elif len(bbox_list[bbox_index]) == 4:
                    xmin = int(np.floor(bbox_list[bbox_index][0]))
                    ymin = int(np.floor(bbox_list[bbox_index][1]))
                    xmax = int(np.ceil(bbox_list[bbox_index][2]))
                    ymax = int(np.ceil(bbox_list[bbox_index][3]))
                else:
                    print(xml_file)
                    assert 0, "{}bbox error".format(xml_file)

                f.writelines(class_name[class_list[bbox_index] + 1])
                f.writelines(" ")
                f.writelines(str(xmin))
                f.writelines(" ")
                f.writelines(str(ymin))
                f.writelines(" ")
                f.writelines(str(xmax))
                f.writelines(" ")
                f.writelines(str(ymax))
                f.writelines("\n")

