"""
# -*- coding: utf-8 -*-
# @Time : 2020/7/23 10:43 
# @Author  : Codingchaozhang
# @File    : resize
"""
import os
from PIL import Image
imgs_path = "./data/masks2/"
resize_imgs_path = "./data/resize_masks2/"

for img_name in os.listdir(imgs_path):
    print(img_name)
    img_path = os.path.join(imgs_path,img_name)

    img = Image.open(img_path)

    dst_img = img.resize((512,256),Image.ANTIALIAS)

    dst_img.save(os.path.join(resize_imgs_path,img_name))

    print("done!")