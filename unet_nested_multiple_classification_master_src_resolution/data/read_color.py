"""
# -*- coding: utf-8 -*-
# @Time : 2020/8/29 16:57 
# @Author  : Codingchaozhang
# @File    : read_color
"""
import cv2
import numpy as np

mask_path_1 = './data/src/ori_masks/0_1.png'
mask_path_2 = './data/src/ori_masks/1_1.png'

mask_img_1 = cv2.imread(mask_path_1)
mask_img_2 = cv2.imread(mask_path_2)

print(mask_img_1.shape)
print(np.unique(mask_img_1))

print(mask_img_2.shape)
print(np.unique(mask_img_2))