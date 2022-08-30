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
    parser.add_argument('--padding_color', type=str, help='gray, black')
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
    os.makedirs(opt.save_label_path)

    for img in tqdm.tqdm(imgs):
        label = [x for x in labels if os.path.basename(x).split('.')[0] == os.path.basename(img).split('.')[0]]
        assert len(label) == 1, f'{label} have same name!'
        label = label[0]
        h, w, c = cv2.imread(img).shape

        padding_img(img, h, w, opt.save_img_path, opt.padding_color)
        padding_label(label, h, w, opt.save_label_path)


def padding_img(img, h, w, save_path, color_class):
    t, b, l, r = 0, 0, 0, 0
    img_array = cv2.imread(img)
    size_diff = h-w
    color = (0, 0, 0)
    if color_class == 'gray':
        color = (128, 128, 128)
    elif color_class == 'black':
        color = (0, 0, 0)
    if size_diff < 0:
        t, b = abs(int(size_diff/2)), abs(int(size_diff/2))
        img_array = cv2.copyMakeBorder(img_array, t, b, l, r, cv2.BORDER_CONSTANT, value=color)
    elif size_diff > 0:
        l, r = abs(int(size_diff/2)), abs(int(size_diff/2))
        img_array = cv2.copyMakeBorder(img_array, t, b, l, r, cv2.BORDER_CONSTANT, value=color)
    cv2.imwrite(os.path.join(save_path, os.path.basename(img)), img_array)


def padding_label(label, h, w, save_path):
    with open(label, 'r') as f:
        lines = [x.strip() for x in f.readlines()]

    contents = []
    size_diff = h-w
    for line in lines:
        pred = line.split(' ')  # x1, y1, x2, y2, x3, y3, x4, y4, class, difficult
        if size_diff > 0:  # small h
            pred[0] = f'{(float(pred[0]) + abs(int(size_diff/2)))}'
            pred[2] = f'{(float(pred[2]) + abs(int(size_diff/2)))}'
            pred[4] = f'{(float(pred[4]) + abs(int(size_diff/2)))}'
            pred[6] = f'{(float(pred[6]) + abs(int(size_diff/2)))}'
        if size_diff < 0:  # small w
            pred[1] = f'{(float(pred[1]) + abs(int(size_diff/2)))}'
            pred[3] = f'{(float(pred[3]) + abs(int(size_diff/2)))}'
            pred[5] = f'{(float(pred[5]) + abs(int(size_diff/2)))}'
            pred[7] = f'{(float(pred[7]) + abs(int(size_diff/2)))}'
        contents.append(' '.join(pred))

    with open(os.path.join(save_path, os.path.basename(label)), 'w') as f:
        f.write('\n'.join(contents))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
