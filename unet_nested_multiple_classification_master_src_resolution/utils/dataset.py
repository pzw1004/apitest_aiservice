"""
# -*- coding: utf-8 -*-
# @Time : 2020/7/22 22:19
# @Author  : Codingchaozhang
# @File    : dataset.py
"""
import os
import os.path as osp
import logging

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# 定义数据读取函数
class BasicDataset(Dataset):
    # 初始化img和mask目录
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir   = imgs_dir
        self.masks_dir  = masks_dir
        self.scale      = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.img_names = os.listdir(imgs_dir)
        logging.info(f'Creating dataset with {len(self.img_names)} examples')

    def __len__(self):
        return len(self.img_names)

    # 定义一个预处理，若图像过大，可rescale缩放，不需缩放，则调整图像的维度
    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            # mask target image
            img_nd = np.expand_dims(img_nd, axis=2)
        else:
            # grayscale input image
            # scale between 0 and 1
            img_nd = img_nd / 255
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        return img_trans.astype(float)

    def __getitem__(self, i):
        img_name = self.img_names[i]
        img_path = osp.join(self.imgs_dir, img_name)
        mask_path = osp.join(self.masks_dir, img_name)

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        assert img.size == mask.size, \
            f'Image and mask {img_name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}