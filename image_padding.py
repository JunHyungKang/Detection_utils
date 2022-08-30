import os
import glob
import shutil
import numpy as np
import cv2
import tqdm
import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--label_path', type=str)
    parser.add_argument('--save_img_path', type=str)
    parser.add_argument('--save_label_path', type=str)
    parser.add_argument('--exist_ok', type=bool)
    parser.add_argument('--padding_color', type=str, help='options: {"gray", "black", "white"}')
    opt = parser.parse_args()
    return opt


def main(opt):
    imgs = glob.glob(os.path.join(opt.img_path, '**', '*.jpg'), recursive=True)
    labels = glob.glob(os.path.join(opt.label_path, '**', '*.txt'), recursive=True)
    assert len(imgs) == len(labels), 'imgs and labels are not matched!!'
    if opt.exist_ok:
        if os.path.isdir(opt.save_img_path):
            shutil.rmtree(opt.save_img_path)
        if os.path.isdir(opt.save_label_path):
            shutil.rmtree(opt.save_label_path)

    os.makedirs(opt.save_img_path)
    os.makedirs(opt.save_lasbel_path)

    for img in tqdm.tqdm(imgs):
        label = [x for x in labels if os.path.basename(x).split('.')[0] == os.path.basename(img).split('.')[0]]
        h, w, c = cv2.imread(img).shape

        padding_img(img, h, w, opt_save_img_path)
        padding_label(label, h, w, opt_save_label_path)


def padding_img(img, h, w, save_path):
    t, b, l, r = 0, 0, 0, 0
    img_array = cv2.imread(img)
    size_diff = h-w
    if size_diff < 0:
        t, b = size_diff/2, size_diff/2
        cv2.copyMakeBorder(img_arry, t, b, l, r, cv2.BORDER_CONSTANT, value='gray')
    elif size_diff > 0:
        l, r = size_diff / 2, size_diff / 2
        cv2.copyMakeBorder(img_arry, t, b, l, r, cv2.BORDER_CONSTANT, value='gray')
    cv2.imwrite(os.path.join(save_path, os.path.basename(img)), img_array)


def padding_label(label, h, w, save_path):
    with open(label, 'r') as f:
        lines = [x.strip() for x in f.readlines()]

    contents = []
    size_diff = h-w
    for line in lines:
        pred = line.split(' ')  # x1, y1, x2, y2, x3, y3, x4, y4, class, difficult
        if size_diff < 0:  # small h
            pred[0] += size_diff / 2
            pred[2] += size_diff / 2
            pred[4] += size_diff / 2
            pred[5] += size_diff / 2
        if size_diff < 0:  # small w
            pred[1] += size_diff / 2
            pred[3] += size_diff / 2
            pred[5] += size_diff / 2
            pred[7] += size_diff / 2
        contents.append(' '.join(pred))

    with open(os.path.join(save_path, os.path.basename(label)), 'w') as f:
        f.write('\n'.join(contens))










if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
