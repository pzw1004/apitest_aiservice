# -*- coding: utf-8 -*-
# @Time    : 2020/8/25 10:11
# @Author  : codingchaozhang
# @File    : cut_data.py
# @Desc    : 对大图片进行滑窗操作slide window 即将大图片裁剪成1024*1024大小，滑动步长为1024

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
    # 裁剪的高和长 以及步长
    parser.add_argument("--cut_height",type=int,default=512)
    parser.add_argument("--cut_width",type=int,default=512)
    # parser.add_argument("--unit_h",type=int,default=512)
    # parser.add_argument("--unit_w",type=int,default=2048)

    # 设置滑窗裁剪后的图片保存路径
    parser.add_argument("--output_dir",type=str,default="./data/train")
    # 设置要裁剪的大图的image和label的路径
    parser.add_argument("--image_path",type=str,default="./data/src/ori_imgs")
    parser.add_argument("--label_path",type=str,default="./data/src/ori_masks")
    return parser.parse_args()

# 获的参数
args = get_arguments()
# UNIT = args.unit
OUTPUT_DIR = args.output_dir
IMAGE_PATH = args.image_path
LABEL_PATH = args.label_path
CUT_HEIGHT = args.cut_height
CUT_WIDTH = args.cut_width

# 定义读取图片的函数
def open_big_pic(path):
    # opencv读取高分辨图片时可能会触发警告，通过该语句来提高警告的阈值
    Image.MAX_IMAGE_PIXELS = 1000000000000000000
    print('open{}'.format(path))
    img = Image.open(path)
    img = np.asarray(img)
    print('img_shape:{}'.format(img.shape))
    return img

# 定义滑窗slide window裁剪图片的函数
def crop_img_label(img,label,output_dir,ori_name):
    # 如果输出目录不存在，则创建
    if not os.path.exists(os.path.join(output_dir,'imgs')):
        print("output img dir not exists make {} dir".format(output_dir))
        os.makedirs(os.path.join(output_dir,'imgs'))
    if not os.path.exists(os.path.join(output_dir,'masks')):
        print("output masks dir not exists make {} dir".format(output_dir))
        os.makedirs(os.path.join(output_dir,'masks'))

    # 将图片变为1024的整数倍，计算出整数倍与原先图片的pad的差距，之后调用np.pad
    new_img_height = (img.shape[0] // CUT_HEIGHT) * CUT_HEIGHT if (img.shape[0] // CUT_HEIGHT == 0) else (img.shape[0] // CUT_HEIGHT + 1) * CUT_HEIGHT
    new_img_width = (img.shape[1] // CUT_WIDTH) * CUT_WIDTH if (img.shape[1] // CUT_WIDTH == 0) else (img.shape[1] // CUT_WIDTH + 1) * CUT_WIDTH
    h_pad = new_img_height - img.shape[0]
    w_pad = new_img_width - img.shape[1]
    # 对原图进行填充（填充完为1024的整数倍）
    img = np.pad(img, ((0, h_pad), (0, w_pad), (0, 0)), mode='constant', constant_values=0)
    label = np.pad(label, ((0, h_pad), (0, w_pad)), mode='constant', constant_values=0)
    # 注：在单通道图像中，像素值0为黑色  像素值255为白色
    # 在本次实验中 检测物体 255是白色  背景0是黑色
    # label = np.where(label>122,255,0)

    # 获取img和label的大小并验证
    img_h, img_w, _ = img.shape
    label_h, label_w = label.shape

    assert img_h == label_h
    assert img_w == label_w
    print(img_h,img_w)
    # 记录高度开始，以及裁剪图片的索引k
    h_index = 0
    k = 0
    # 两个while循环 控制滑窗
    while h_index <= img_h - CUT_HEIGHT:
        # 记录宽度的开始
        w_index = 0
        while w_index <= img_w - CUT_WIDTH:
            img_unit    = img[h_index : h_index + CUT_HEIGHT, w_index : w_index + CUT_WIDTH, :]
            label_unit  = label[h_index : h_index + CUT_HEIGHT, w_index : w_index + CUT_WIDTH]
            # 对其进行判断，如果1即分割 设置为1 统计其个数 如果在1024*1024中包含了要分割的物体超过100个像素点才保存
            if np.sum(np.where(label_unit==1,1,0)) > 100:
                k = k + 1
                print("\rcrop {} unit image".format(k),end = '',flush=True)
                path_unit_img = os.path.join(output_dir,'imgs','{}_{}.png'.format(ori_name,k))
                path_unit_label = os.path.join(output_dir,'masks','{}_{}.png'.format(ori_name,k))
                io.imsave(path_unit_img,img_unit)
                cv2.imwrite(path_unit_label,label_unit)
            w_index = w_index + CUT_WIDTH

        h_index = h_index + CUT_HEIGHT

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
        crop_img_label(img,label,OUTPUT_DIR,ori_name=ori_name)


