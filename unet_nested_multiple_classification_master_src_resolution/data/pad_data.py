# -*- coding: utf-8 -*-
# @Time    : 2020/8/25 10:11
# @Author  : codingchaozhang
# @File    : cut_data.py
# @Desc    : 对大图片进行pad操作 使其能整除与32

# 导包
import numpy as np
from skimage import io
import cv2
import os
from PIL import Image
import argparse
from glob import glob

# 定义获取超参数的函数
def get_arguments():
    parser = argparse.ArgumentParser(description="crop")

    # 设置的图片保存路径
    parser.add_argument("--output_dir",type=str,default="./data/train")
    # 设置要裁剪的大图的image和label的路径
    parser.add_argument("--image_path",type=str,default="./data/src/ori_imgs")
    parser.add_argument("--label_path",type=str,default="./data/src/ori_masks")
    # 定义图片能整除
    parser.add_argument("--unit",type=int,default=32)

    return parser.parse_args()

# 获的参数
args = get_arguments()
# UNIT = args.unit
OUTPUT_DIR = args.output_dir
IMAGE_PATH = args.image_path
LABEL_PATH = args.label_path
# 定义图片整除的数
UNIT = args.unit
# 定義縮放的比例
SCALE = 0.25
# 定义读取图片的函数
def open_big_pic(path):
    # opencv读取高分辨图片时可能会触发警告，通过该语句来提高警告的阈值
    Image.MAX_IMAGE_PIXELS = 1000000000000000000
    print('open{}'.format(path))
    img = Image.open(path)
    img = np.asarray(img)
    print('img_shape:{}'.format(img.shape))
    return img

# 定义补图片的函数
def pad_img_label(img,label,output_dir,ori_name):
    # 如果输出目录不存在，则创建
    if not os.path.exists(os.path.join(output_dir,'imgs')):
        print("output img dir not exists make {} dir".format(output_dir))
        os.makedirs(os.path.join(output_dir,'imgs'))
    if not os.path.exists(os.path.join(output_dir,'masks')):
        print("output masks dir not exists make {} dir".format(output_dir))
        os.makedirs(os.path.join(output_dir,'masks'))

    # 将图片变为1024的整数倍，计算出整数倍与原先图片的pad的差距，之后调用np.pad
    new_img_height = (img.shape[0]) if (img.shape[0] * SCALE % UNIT == 0) else (img.shape[0] * SCALE // UNIT + 1) * (UNIT) / SCALE
    new_img_width =  (img.shape[1])  if (img.shape[1]  * SCALE % UNIT == 0) else (img.shape[1] * SCALE // UNIT + 1) * (UNIT) / SCALE

    new_img_height = int(new_img_height)
    new_img_width = int(new_img_width)

    h_pad = new_img_height - img.shape[0]
    w_pad = new_img_width - img.shape[1]
    print(new_img_height,new_img_width)
    print(h_pad,w_pad)
    # 对原图进行填充（填充完为1024的整数倍）
    # if h_pad == 0:
    #     img = np.pad(img, ((0, 0), (0, w_pad), (0, 0)), mode='constant', constant_values=0)
    #     label = np.pad(label, ((0, 0), (0, w_pad)), mode='constant', constant_values=0)
    # if w_pad == 0:
    #     img = np.pad(img, ((0, h_pad), (0, 0), (0, 0)), mode='constant', constant_values=0)
    #     label = np.pad(label, ((0, h_pad), (0, 0)), mode='constant', constant_values=0)
    # if h_pad!=0 & w_pad!=0:
    img = np.pad(img, ((0, h_pad), (0, w_pad), (0, 0)), mode='constant', constant_values=0)
    label = np.pad(label, ((0, h_pad), (0, w_pad)), mode='constant', constant_values=0)
    # 注：在单通道图像中，像素值0为黑色  像素值255为白色
    # 在本次实验中 检测物体 255是白色  背景0是黑色
    label = np.where(label>122,1,0)

    # 获取img和label的大小并验证
    img_h, img_w, _ = img.shape
    label_h, label_w = label.shape

    assert img_h == label_h
    assert img_w == label_w
    print(img_h,img_w)
    pad_img_path = os.path.join(output_dir,"imgs","{}.png".format(ori_name))
    pad_label_path = os.path.join(output_dir,"masks","{}.png".format(ori_name))
    io.imsave(pad_img_path, img)
    cv2.imwrite(pad_label_path, label)

# 主函数
if __name__ == '__main__':
    img_path = IMAGE_PATH
    label_path = LABEL_PATH
    # 获取image和label同名称的name
    img_name = [(name.split(os.sep)[-1]).split('.')[0] for name in glob(os.path.join(img_path,'*.png'))]
    # 对其name遍历
    for ori_name in img_name:
        img = open_big_pic(os.path.join(img_path,ori_name+".png"))
        label = open_big_pic(os.path.join(label_path,ori_name+".png"))
        pad_img_label(img,label,OUTPUT_DIR,ori_name=ori_name)


