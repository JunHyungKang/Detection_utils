import os
import glob
import glob
import shutil
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--fps', type=int)
    opt = parser.parse_args()
    return opt


def main(opt):
    images = glob.glob(os.path.join(opt.img_path, '**', '*'), recursive=True)
    images = sorted([x for x in images if x.split('.')[-1] in ['jpg']])

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    video_name = opt.save_name
    codec = 'DIVX'
    fourcc = cv2.VideoWriter_fourcc(*codec)
    fps = 3

    os.makedirs(opt.save_path, exist_ok=True)
    video = cv2.VideoWriter(os.path.join(opt.save_path, video_name), fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(image))

    video.release()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)