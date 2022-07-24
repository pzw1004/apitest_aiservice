"""
# -*- coding: utf-8 -*-
# @Time : 2020/7/23 12:53 
# @Author  : Codingchaozhang
# @File    : check_data_shape
"""
import cv2
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

img_path = "data/images/1_85.jpg"
mask_path = "data/masks/1_85.jpg"

#
# img2_path = "data/resize_images2/1582701515096.png"
# mask2_path = "data/resize_masks2/1582701515096.png"




img = cv2.imread(img_path)
mask = cv2.imread(mask_path)

# img2 = cv2.imread(img2_path)
# mask2 = cv2.imread(mask2_path)

print(img.shape,mask.shape)


lbl = np.asanyarray(PIL.Image.open(mask_path).convert("L"))
print(lbl.dtype)


print(lbl.min(),lbl.max())
# print(lbl2.min(),lbl2.max())

print(lbl)
plt.imshow(lbl)
# plt.imshow(lbl2)
plt.show()