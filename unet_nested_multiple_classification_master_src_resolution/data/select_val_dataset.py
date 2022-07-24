# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Project Name :  pytorch-code
   File Name    :  select_val_dataset
   Author       :  sjw
   Time         :  20-1-5 16:18
   Description  :  随机选择20%的训练集当做测试集
-------------------------------------------------
   Change Activity: 
                   20-1-5 16:18
-------------------------------------------------
"""
from glob import glob
import shutil
import os
import random
from tqdm import tqdm

# 设置随机种子
random.seed(1)
# 获取图片名称
file_name_list = [(name.split(os.sep)[-1]).split('.')[0] for name in glob(os.path.join('./data/complex', 'train', 'imgs', '*.png'))]
# 打乱其顺序
random.shuffle(file_name_list)
# 给出训练集目录
src = './data/complex/train'
# 给出验证集目录
dst = './data/complex/val'

# 若不存在，则创建
if not os.path.exists(os.path.join(dst,'imgs')):
    print("output img dir not exists make {} dir".format(dst))
    os.makedirs(os.path.join(dst,"imgs"))
if not os.path.exists(os.path.join(dst,"masks")):
    print("output mask dir not exists make {} dir".format(dst))
    os.makedirs(os.path.join(dst,"masks"))

# 从图片列表中拿出20%作为验证集
for file in tqdm(file_name_list[:len(file_name_list)//5]):
    # 移动imgs
    imgs_src = os.path.join(src, 'imgs', file+'.png')
    imgs_dst = os.path.join(dst, 'imgs', file + '.png')
    shutil.move(src=imgs_src, dst=imgs_dst)
    # 移动masks
    masks_src = os.path.join(src, 'masks', file + '.png')
    masks_dst = os.path.join(dst, 'masks', file + '.png')
    shutil.move(src=masks_src, dst=masks_dst)
