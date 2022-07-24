#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/15 13:52
# @Author  : codingchaozhang
# @File    : PSPNet.py
# @Desc    :


import torch
from torch import nn
from torch.nn import functional as F
from module import extractors



class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 配置
        self.n_classes = cfg.n_classes
        sizes = (1, 2, 3, 6)
        psp_size = 512
        deep_features_size = 1024
        backend = 'resnet34'
        pretrained = False

        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 512, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(512, 256)
        self.up_2 = PSPUpsample(256, 128)
        self.up_3 = PSPUpsample(128, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, self.n_classes, kernel_size=1),
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(deep_features_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, self.n_classes)
        # )

    def forward(self, x):
        f, class_f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        # auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        return self.final(p)