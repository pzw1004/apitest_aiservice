"""
@Time : 2020/7/22 22:13 
@Author : codingchaozhang
"""
# @Time : 2020/7/22 22:12
# @Author : codingchaozhang
import os
import os.path as osp
from tqdm import tqdm
import cv2
import numpy as np


num_classes = 3
mask_dir = "data/masks"
mask_names = os.listdir(mask_dir)

for mask_name in tqdm(mask_names):
    mask_path = osp.join(mask_dir, mask_name)
    mask = cv2.imread(mask_path, 0)
    h, w = mask.shape[:2]
    pix = []
    for i in range(0, num_classes):
        pix.append(len(np.where(mask==i)[0]))
    if sum(pix) != h*w:
        print("error: " + mask_name)