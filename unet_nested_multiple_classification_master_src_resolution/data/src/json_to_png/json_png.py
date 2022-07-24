import cv2
import numpy as np
import json
import os

category_types = ["Background","qualified","qualified"]

img_dir = "./images"
json_dir = "./json"
mask_dir = "./masks"


for img_path in os.listdir(img_dir):
    name = img_path.split(".")[0]
    # 读取图片
    img = cv2.imread(os.path.join(img_dir,img_path))
    h, w = img.shape[:2]
    # 创建一个大小和原图相同的空白图像
    mask = np.zeros([h,w,1],np.uint8)
    
    json_name = name + ".json"
    json_path = os.path.join(json_dir,json_name)
    with open(json_path,"r") as f:
        label = json.load(f)
    
    shapes = label["shapes"]
    for shape in shapes:
        category = shape["label"]
        points = shape["points"]
        # 填充
        points_array = np.array(points,dtype=np.int32)
        # mask = cv2.fillPoly(mask,[points_array],category_types.index(category))
        mask = cv2.fillPoly(mask,[points_array],1)
    
    mask_name = name + ".png" 
    mask_path = os.path.join(mask_dir,mask_name)
    cv2.imwrite(mask_path,mask)
    print("process the image:" + name)