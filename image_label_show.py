import os
import glob
import shutil
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import argparse
import tqdm


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--label_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--exist_ok', type=bool)
    opt = parser.parse_args()
    return opt


def main(opt):
    results = extracting_results(opt)
    save_imgs(results, opt)


def extracting_results(opt):
    results_file = sorted(glob.glob(os.path.join(opt.label_path, '*.txt')))
    results = []
    for result_file in results_file:
        img_name = os.path.basename(result_file).split('.')[0]
        with open(result_file, 'r') as f:
            lines = f.readlines()
        lines = [x.strip().split(' ') for x in lines]
        lines = [x + [img_name] for x in lines]
        results.extend(lines)
    return results


def save_imgs(results, opt):
    if opt.exist_ok:
        if os.path.isdir(opt.save_path):
            shutil.rmtree(opt.save_path)
    os.makedirs(opt.save_path, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(opt.img_path, '*.jpg')))
    for img in tqdm.tqdm(imgs):
        result = [x for x in results if x[-1] == os.path.basename(img).split('.')[0]]
        img_array = Image.fromarray(draw_pred(img, result))
        img_array.save(os.path.join(opt.save_path, os.path.basename(img)))


def draw_pred(img_path, result):
    image = Image.open(img_path)
    draw = ImageDraw.Draw(image)
    img_w = np.asarray(image).shape[0]

    for pred in result:
        pts = []
        cl_name = pred[-3]
        coord = pred[:8]

        pts.append(list(map(float, coord[:2])))
        pts.append(list(map(float, coord[2:4])))
        pts.append(list(map(float, coord[4:6])))
        pts.append(list(map(float, coord[6:8])))

        pts = np.array(pts)
        rect = order_points(pts)
        rect = list(map(tuple, rect))
        draw.polygon([rect[0], rect[1], rect[2], rect[3]], width=3, outline='red')

        font = ImageFont.truetype('/usr/share/fonts/truetype/Gargi/Gargi.ttf', size=int(img_w/50))
        draw.text((rect[3][0] + int(img_w/100), rect[3][1] + int(img_w/100)),
                  f'{cl_name}', 'red', font=font)
    return np.asarray(image)
#
#
# def crop_imgs(results, opt):
#     imgs = sorted([os.path.basename(x).split('.')[0] for x in glob.glob(os.path.join(opt.img_path, '*.jpg'))])
#     if os.path.isdir(os.path.join(opt.save_path, 'crop_imgs')):
#         shutil.rmtree(os.path.join(opt.save_path, 'crop_imgs'))
#     for img in imgs:
#         result = [x for x in results if x[0] == img]
#         img_array = Image.open(os.path.join(opt.img_path, img + '.jpg'))
#         for idx, pred_obj in enumerate(result):
#             pts = []
#             conf = pred_obj[1]
#             if float(conf) >= opt.conf_thr:
#                 cl_name = pred_obj[-1]
#                 coord = pred_obj[2:-1]
#
#                 pts.append(list(map(float, coord[:2])))
#                 pts.append(list(map(float, coord[2:4])))
#                 pts.append(list(map(float, coord[4:6])))
#                 pts.append(list(map(float, coord[6:8])))
#
#                 pts = np.asarray(pts, dtype='float32')
#                 rect = order_points(cv2.boxPoints(cv2.minAreaRect(pts)))
#                 crop_img = four_point_transform(np.asarray(img_array), rect)
#                 crop_img = Image.fromarray(crop_img)
#                 os.makedirs(os.path.join(opt.save_path, 'crop_imgs', f'{cl_name}'), exist_ok=True)
#                 crop_img.save(os.path.join(opt.save_path, 'crop_imgs', f'{cl_name}',
#                                            f'{img}_{idx}_{round(float(conf), 1)}' + '.jpg'))
#
#
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect  # (tl, tr, br, bl)


def four_point_transform(image, rect):
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)