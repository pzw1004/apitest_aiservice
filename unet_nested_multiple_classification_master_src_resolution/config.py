"""
# -*- coding: utf-8 -*-
# @Time : 2020/7/22 22:19
# @Author  : Codingchaozhang
# @File    : config.py
"""

import os
from unet import HRUNet

# 定义一个类配置文件
#  - 训练轮数
#  - batchsize大小
# - 验证集
class UNetConfig:
    def __init__(self,
                 epochs = 5000,  # Number of epochs
                 batch_size = 1,    # Batch size
                 validation = 20.0,   # Percent of the data that is used as validation (0-100)
                 out_threshold = 0.5,

                 optimizer='Adam',
                 lr = 0.0001,     # learning rate
                 lr_decay_milestones = [1000, 2000],
                 lr_decay_gamma = 0.9,
                 weight_decay=1e-8,
                 momentum=0.9,
                 nesterov=True,

                 n_channels = 3, # Number of channels in input images
                 n_classes = 6,  # Number of classes in the segmentation
                 scale = 0.2,    # Downscaling factor of the images

                 load = False,   # Load model from a .pth file
                 save_cp = True,

                 model='PSPNet', # UNet  NestedUNet HRUNet
                 bilinear = True,
                 deepsupervision = False,
                 ):
        super(UNetConfig, self).__init__()

        self.images_dir = './data/train/imgs'
        self.masks_dir = './data/train/masks'
        self.checkpoints_dir = './data/checkpoints'

        self.epochs = epochs
        self.batch_size = batch_size
        self.validation = validation
        self.out_threshold = out_threshold

        self.optimizer = optimizer
        self.lr = lr
        self.lr_decay_milestones = lr_decay_milestones
        self.lr_decay_gamma = lr_decay_gamma
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.scale = scale

        self.load = load
        self.save_cp = save_cp

        self.model = model
        self.bilinear = bilinear
        self.deepsupervision = deepsupervision

        os.makedirs(self.checkpoints_dir, exist_ok=True)
