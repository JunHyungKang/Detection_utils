#######################################################################################################
## Original Source: https://github.com/dacon-ai/AerialDetection/blob/master/dacon_utils/geojson2coco.py
#######################################################################################################

import os
import json
from typing import List
import math
from glob import glob

from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image

rootfolder = 'D:/Dacon/Arirang_Dataset/512_256/'

image_root = os.path.join(rootfolder, 'images')
txt_path = os.path.join(rootfolder, 'labels')
dest_file = os.path.join(rootfolder, 'train.json')

task = 'train'
# task = 'test'
difficult = '-1'

NIA_CLASSES = ['배경', '소형 선박', '대형 선박', '민간 항공기', '군용 항공기', '소형 승용차', '버스', '트럭', '기차', '크레인', '다리',
               '정유탱크', '댐', '운동경기장', '헬리패드', '원형 교차로']
CLASS_NAMES_EN = ('background', 'small_ship', 'large_ship', 'civilian_aircraft', 'military_aircraft', 'small_car',
                  'bus', 'truck', 'train', 'crane', 'bridge', 'oil_tank', 'dam', 'athletic_field', 'helipad',
                  'roundabout')
CLASS_DICT = {'background':0, 'small_ship':1, 'large_ship':2, 'civilian_aircraft':3, 'military_aircraft':4, 'small_car':5,
              'bus':6, 'truck':7, 'train':8, 'crane':9, 'bridge':10, 'oil_tank':11, 'dam':12, 'athletic_field':13,
              'helipad':14, 'roundabout':15}

# dota to coco
if task == 'train':
    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(CLASS_NAMES_EN[1:]):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    with open(dest_file, 'w') as f_out:
        txts = os.listdir(txt_path)
        obj_coords = list()
        image_ids = list()
        class_indices = list()
        class_names = list()
        for txt in tqdm(txts, desc='loading txt files'):
            with open(os.path.join(txt_path, txt)) as f:
                labels = f.readlines()
                for label in labels:
                    label = label.split(' ')
                    if label[8] != 'etc':
                        obj_coords.append(label[0:8])
                        image_ids.append(os.path.join(txt_path, txt).replace('.txt', '.png').split(os.path.sep)[-1])
                        class_indices.append(CLASS_DICT[label[8]])
                        class_names.append(label[8])

        img_id_map = {img_file: i + 1 for i, img_file in enumerate(list(set(image_ids)))}
        image_ids = [img_id_map[img_file] for img_file in image_ids]

        # convert_labels_to_objects(coords, class_ids, class_names, image_ids, difficult=0, is_clockwise=False):
        objs = list()
        inst_count = 1

        for coords, cls_id, cls_name, img_id in tqdm(zip(obj_coords, class_indices, class_names, image_ids),
                                                       desc="converting labels to objects"):
            for i in range(len(coords)):
                coords[i] = float(coords[i])

            x_coords = coords[0::2]
            y_coords = coords[1::2]

            xmin = min(x_coords)
            ymin = min(y_coords)

            xmax = max(x_coords)
            ymax = max(y_coords)

            w = xmax - xmin
            h = ymax - ymin

            single_obj = {}
            single_obj['difficult'] = difficult
            single_obj['area'] = w * h

            if cls_name in CLASS_NAMES_EN:
                single_obj['category_id'] = CLASS_DICT[cls_name]
            else:
                continue

            single_obj['segmentation'] = [[int(p) for p in coords]]
            single_obj['iscrowd'] = 0
            single_obj['bbox'] = (xmin, ymin, w, h)
            single_obj['image_id'] = img_id
            single_obj['id'] = inst_count
            inst_count += 1
            objs.append(single_obj)

        data_dict['annotations'].extend(objs)

        for imgfile in tqdm(img_id_map, desc='saving img info'):
            imagepath = os.path.join(image_root, imgfile)
            img_id = img_id_map[imgfile]
            img = cv2.imread(imagepath)
            height, width, c = img.shape
            single_image = {}
            single_image['file_name'] = imgfile
            single_image['id'] = img_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

        json.dump(data_dict, f_out)

elif task == 'test':
    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    for idex, name in enumerate(CLASS_NAMES_EN[1:]):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    with open(dest_file, 'w') as f_out:
        image_ids = os.listdir(image_root)
        img_id_map = {img_file: i + 1 for i, img_file in enumerate(list(set(image_ids)))}
        image_ids = [img_id_map[img_file] for img_file in image_ids]

        for imgfile in tqdm(img_id_map, desc='saving img info'):
            imagepath = os.path.join(image_root, imgfile)
            img_id = img_id_map[imgfile]
            img = cv2.imread(imagepath)
            height, width, c = img.shape
            single_image = {}
            single_image['file_name'] = imgfile
            single_image['id'] = img_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

        json.dump(data_dict, f_out)
