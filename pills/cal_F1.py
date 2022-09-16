import os
import glob
import shutil
import numpy as np
import cv2
import argparse
import tqdm


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_results', type=str)
    parser.add_argument('--new_results', type=str)
    parser.add_argument('--ocr_results', type=str)
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--label_path', type=str)
    parser.add_argument('--ocr_class', type=str)
    parser.add_argument('--crop_pjt_name', type=str)
    parser.add_argument('--conf_thr', type=float)
    opt = parser.parse_args()
    return opt


def main(opt):
    if opt.ocr_results:
        replace_ocr(opt.new_results, opt.old_results, opt.ocr_class, opt.ocr_results, opt.crop_pjt_name, opt.img_path)
        cal_f1(opt.img_path, opt.label_path, opt.new_results, opt.conf_thr)
    elif not opt.ocr_results:
        cal_f1(opt.img_path, opt.label_path, opt.old_results, opt.conf_thr)


def replace_ocr(new_results, old_results, ocr_class, ocr_results, crop_pjt_name, img_path):
    try:
        shutil.rmtree(new_results)
    except:
        pass
    shutil.copytree(old_results, new_results)
    os.remove(os.path.join(new_results, f'Task1_{ocr_class}.txt'))

    files = sorted(glob.glob(os.path.join(old_results, 'Task1_*.txt')))
    results = []
    for file in files:
        cl_name = os.path.basename(file).split('.')[0].split('_')[-1]
        with open(file, 'r') as f:
            lines = f.readlines()
        lines = [x.strip().split(' ') for x in lines]
        lines = [x + [cl_name] for x in lines]
        results.extend(lines)

    with open(ocr_results, 'r') as f:
        ocr_lines = [x.strip().split('\t') for x in f.readlines()]

    ocr_dict = dict()
    for ocr_line in ocr_lines:
        ocr_line = [x.strip() for x in ocr_line]
        if ocr_line[0].split('/')[-2] == crop_pjt_name:
            img_name = os.path.basename(ocr_line[0])
            obj = ocr_line[1]
            ocr_conf = ocr_line[2]
            ocr_dict[img_name] = [obj, ocr_conf]

    imgs = sorted([os.path.basename(x).split('.')[0] for x in glob.glob(os.path.join(img_path, '*.jpg'))])
    for img in imgs:
        result = [x for x in results if x[0] == img]
        for idx, pred_obj in enumerate(result):
            conf = pred_obj[1]
            cl_name = pred_obj[-1]
            coord = pred_obj[2:-1]
            crop_name = f'{img}_{idx}_{round(float(conf), 1)}' + '.jpg'
            if cl_name == ocr_class:
                try:
                    new_cl = ocr_dict[crop_name][0]
                    if new_cl == '#':
                        new_cl = 'logo'
                    ocr_conf = ocr_dict[crop_name][1]
                    add_line = [img, float(conf) * float(ocr_conf)] + coord
                    add_line = list(map(str, add_line))
                    try:
                        with open(os.path.join(new_results, f'Task1_{new_cl}.txt'), 'a') as f:
                            f.write(' '.join(add_line) + '\n')
                    except:
                        with open(os.path.join(new_results, f'Task1_{new_cl}.txt'), 'w') as f:
                            f.write(' '.join(add_line) + '\n')
                except:  # ocr의 결과값이 없는 경우
                    pass


def cal_f1(img_path, label_path, new_results, conf_thr):
    gt_imgs = dict()
    for img in glob.glob(os.path.join(img_path, '*.jpg')):
        file_name = os.path.basename(img).split('.')[0]
        objs = []
        with open(os.path.join(label_path, file_name + '.txt')) as f:
            obj_list = f.readlines()
            for obj in obj_list:
                objs.append(obj.strip().split(' ')[-2])
        gt_imgs[file_name] = objs

    pred_dict = dict()

    for pred in sorted(glob.glob(os.path.join(new_results, 'Task1_*.txt'))):
        with open(pred, 'r') as f:
            lines = f.readlines()
        obj_name = os.path.basename(pred).split('_')[1].split('.')[0]
        lines = [x.strip() for x in lines]
        imgs = [x.split(' ')[0] for x in lines]
        confs = [x.split(' ')[1] for x in lines]
        bboxes = [x.split(' ')[2:] for x in lines]
        for img, conf, bbox in zip(imgs, confs, bboxes):
            if float(conf) >= conf_thr:
                try:
                    pred_dict[img + '.jpg'].append([obj_name, bbox, conf])
                except:
                    pred_dict[img + '.jpg'] = [[obj_name, bbox, conf]]

    pred_imgs = dict()
    for img in glob.glob(os.path.join(img_path, '*.jpg')):
        file_name = os.path.basename(img).split('.')[0]
        objs = []
        try:
            for obj in pred_dict[file_name + '.jpg']:
                objs.append(obj[0])
        except:
            pass
        pred_imgs[file_name] = objs

    imgs_name = [os.path.basename(x).split('.')[0] for x in glob.glob(os.path.join(img_path, '*.jpg'))]
    tp_imgs = cal_performance(imgs_name, gt_imgs, pred_imgs)

    tp, pred, gt = [], [], []
    for img in imgs_name:
        tp.extend(tp_imgs[img])
        pred.extend(pred_imgs[img])
        gt.extend(gt_imgs[img])

    micro_precision = len(tp) / len(pred)
    micro_recall = len(tp) / len(gt)
    print(f'TP: {len(tp)}, PRED: {len(pred)}, GT: {len(gt)}')
    print(f'micro_precision: {round(micro_precision, 2)}')
    print(f'micro_recall: {round(micro_recall, 2)}')
    print(f'F1: {round(micro_recall * micro_precision * 2 / (micro_precision + micro_recall), 2)}')


def cal_performance(imgs, gt_imgs, pred_imgs):
    tp_imgs = dict()
    for img in imgs:
        tp = []
        obj_pred = pred_imgs[img]
        obj_gt = gt_imgs[img].copy()
        for pred in obj_pred:
            if pred in obj_gt:
                tp.append(pred)
                obj_gt.remove(pred)
        tp_imgs[os.path.basename(img).split('.')[0]] = tp
    return tp_imgs


# def cal_f1(img_path, label_path, pred_path, pred_format):
#     gt_dict = generate_dota_dict(img_path, label_path)
#     if pred_format == 'Task1':
#         pred_dict = generate_task1_dict(img_path, pred_path)
#     img_names = [os.path.basename(x).split('.')[0] for x in glob.glob(os.path.join(img_path, '*.jpg'))]
#
#     tp_dict = dict()
#     for img in img_names:
#         tp = []
#         obj_pred = pred_dict[img]
#         obj_gt = gt_dict[img].copy()
#         for pred in obj_pred:
#             if pred in obj_gt:
#                 tp.append(pred)
#                 obj_gt.remove(pred)
#         tp_dict[os.path.basename(img).split('.')[0]] = tp
#
#     tp, pred, gt = [], [], []
#     for img in img_names:
#         tp.extend(tp_dict[img])
#         pred.extend(pred_dict[img])
#         gt.extend(gt_dict[img])
#
#     micro_precision = len(tp) / len(pred)
#     micro_recall = len(tp) / len(gt)
#     print(f'TP: {len(tp)}, PRED: {len(pred)}, GT: {len(gt)}')
#     print(f'micro_precision: {round(micro_precision, 2)}')
#     print(f'micro_recall: {round(micro_recall, 2)}')
#     print(f'F1 score: {round(micro_recall * micro_precision * 2 / (micro_precision + micro_recall), 2)}')
#
#
# def generate_dota_dict(img_path, label_path):
#     obj_dict = dict()
#     for img in glob.glob(os.path.join(img_path, '*.jpg')):
#         file_name = os.path.basename(img).split('.')[0]
#         objs = []
#         with open(os.path.join(label_path, file_name + '.txt')) as f:
#             obj_list = f.readlines()
#             for obj in obj_list:
#                 objs.append(obj.strip().split(' ')[-2])
#         obj_dict[file_name] = objs
#     return obj_dict
#
#
# def generate_task1_dict(img_path, pred_path):
#     pred_dict = dict()
#     for pred in sorted(glob.glob(os.path.join(pred_path, 'Task1_*.txt'))):
#         with open(pred, 'r') as f:
#             lines = f.readlines()
#         obj_name = os.path.basename(pred).split('_')[1].split('.')[0]
#         lines = [x.strip() for x in lines]
#         imgs = [x.split(' ')[0] for x in lines]
#         for img in imgs:
#             try:
#                 pred_dict[img].append(obj_name)
#             except:
#                 pred_dict[img] = [obj_name]
#     for img in glob.glob(os.path.join(img_path, '*.jpg')):
#         img = os.path.basename(img).split('.')[0]
#         if img not in pred_dict.keys():
#             pred_dict[img] = []
#     return pred_dict


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)