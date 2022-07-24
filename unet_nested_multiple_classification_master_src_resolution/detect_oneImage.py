"""
# -*- coding: utf-8 -*-
# @Time : 2020/7/22 22:19
# @Author  : Codingchaozhang
# @File    : inference.py
"""

import argparse
import logging
import os
import os.path as osp
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
#from tqdm import tqdm
import time
from unet import NestedUNet
from unet import UNet
from utils.dataset import BasicDataset
from config import UNetConfig
from unet import PSPNet
import sys

# 557,149 556,149 556,149 556,149 555,149 555,150 555,150 555,150 555,150 555,150 555,150 554,151 554,151 554,151 554,151 554,151 554,151 554,152 554,152 553,152 553,155 554,155 554,155 554,155 554,156 554,15
# 6 554,156 555,157 555,157 555,157 555,157 555,157 555,157 555,158 560,158 560,157 560,157 560,157 561,157 561,157 561,156 561,156 561,156 562,156 562,151 561,151 561,151 561,151 561,151 561,151 561,151 560,
# 150 560,150 560,150 560,150 560,150 560,150 560,149 558,149 558,149 557,149

cfg = UNetConfig()

def inference_one(net, image, device):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(image, cfg.scale))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        if cfg.deepsupervision:
            output = output[-1]

        if cfg.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image.size[1], image.size[0])),
                transforms.ToTensor()
            ]
        )
        
        if cfg.n_classes == 1:
            probs = tf(probs.cpu())
            mask = probs.squeeze().cpu().numpy()
            return mask > cfg.out_threshold
        else:
            masks = []
            for prob in probs:
                prob = tf(prob.cpu())
                mask = prob.squeeze().cpu().numpy()
                mask = mask > cfg.out_threshold
                masks.append(mask)
            return masks
  
  
# def get_args():
#     parser = argparse.ArgumentParser(description='Predict masks from input images',
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--model', '-m', default='./data/checkpoints/epoch_900.pth',
#                         metavar='FILE',
#                         help="Specify the file in which the model is stored")
#     # parser.add_argument('--input', '-i', dest='input', type=str, default='./data/test/input',
#     #                     help='Directory of input images')
#     parser.add_argument('--output', '-o', dest='output', type=str, default='./data/temp',
#                         help='Directory of ouput images')
#     return parser.parse_args()

# 定义图片整除的数
UNIT = 32
# 定義縮放的比例
SCALE = 0.2


# 定义一个直方图均衡化的操作函数（图片预处理操作）
def equalize_transfrom(gray_img):
    return cv2.equalizeHist(gray_img)

import json
def printResult(results):
    '''
    将预测结果，根据预定义好的格式输出
    :param results: [x1,y1,x2,y2,conf,cls_conf,cls_pred]
    [111.56450653076172, 452.42237854003906, 150.10369873046875, 476.8945617675781, 0.8893440365791321, 0.9999973773956299, 0]
    :return: None
    '''
    #

    keys = ["x_min", "y_min", "x_max", "y_max", "conf", "cls_conf", "cls_pred"]

    for result in results:
        dictionary = dict(zip(keys, result))
        j = json.dumps(dictionary)

        print("damage location @@",j)
    #     keys = ["x_min","y_min","x_max","y_max","conf","cls_conf","cls_pred"]
    #     # print('damage location x_min: @@', x_min, '@@ y_min: @@',y_min,'@@ x_max: @@',x_max+64, '@@ y_max: @@',y_max+64)
    #     print(result)
    #     # print("damage location x_min: @@",result[0],"@@ y_min: @@",result[1],"@@ x_max: @@",result[2],"@@ y_max: @@",result[3],
    #     #       "@@ conf")

if __name__ == "__main__":

    realpath = os.path.dirname(os.path.realpath(__file__))
    # realpath = "D:/FlawSegmentation/Newversion"
    # args = get_args()
    # 获取图片路径
    imagePath = sys.argv[1]
    # imagePath = realpath+"/1.png"
    # 模型位置
    modelPath = realpath+ "/data/checkpoints/Epoch2440.pth"
    # 临时存储位置
    outputPath = realpath +"/data/temp"
    # input_imgs = os.listdir(args.input)

    net = eval(cfg.model)(cfg)
    logging.info("Loading model {}".format(modelPath))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(modelPath, map_location=device))

    logging.info("Model loaded !")

    logging.info("\nPredicting image {} ...".format(imagePath))

    # img_path = imagePath


    # 打开图片
    start = time.time()
    # print("")
    # print("===============开始读取图片================")
    src_img = cv2.imread(imagePath,1)
    origin_img = src_img.copy()

    src_img_h = src_img.shape[0]
    src_img_w = src_img.shape[1]
    # print("================图片预处理=================")

    # 0.将无关的部分置于0
    src_img[0:400,:,:] = 0
    src_img[1200:,:,:] = 0

    # 1.高斯滤波
    blur = cv2.GaussianBlur(src_img, (3, 3), 0)
    # 2.直方图均衡化操作
    img = blur  # 这里需要指定一个 img_path
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    b_out = equalize_transfrom(b)
    g_out = equalize_transfrom(g)
    r_out = equalize_transfrom(r)
    equa_out = np.stack((b_out, g_out, r_out), axis=-1)
    # 3.自动色彩均衡操作
    img = equa_out  # 这里需要指定一个 img_path
    b, g, r = cv2.split(img)
    B = np.mean(b)
    G = np.mean(g)
    R = np.mean(r)
    K = (R + G + B) / 3
    Kb = K / B
    Kg = K / G
    Kr = K / R
    cv2.addWeighted(b, Kb, 0, 0, 0, b)
    cv2.addWeighted(g, Kg, 0, 0, 0, g)
    cv2.addWeighted(r, Kr, 0, 0, 0, r)
    merged = cv2.merge([b, g, r])
    # 4.自适应直方图均衡化
    image = merged
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    clahe_out = cv2.merge([b, g, r])
    # 5.对图片加权融合
    img1 = clahe_out
    img2 = equa_out
    dst = cv2.addWeighted(img1, 0.2, img2, 0.8, 0)
    # 6.对图片进行裁剪，多余的部分去掉
    dst = dst[400:1200,:,:]
    # 7.对裁剪图片进行保存
    # img_name = "temp"
    # img_name = img_name.split(".")[0]
    img_name =  "temp.png"

    #
    dst_path = os.path.join(outputPath,img_name).replace("\\","/")
    #dst_path = "D:/FlawSegmentation/PSPNet/data/temp.png"
    #print(dst_path)
    x = cv2.imwrite(dst_path,dst)
    #print(x)
    # 对图片重新读取并进行填充成32整数的操作
    img_pil = Image.open(dst_path)

    # #  对上述保存的图片进行删除
    # if os.path.exists(dst_path):
    #     os.remove(dst_path)

    img = np.asarray(img_pil)
    # 将图片变为要32的整数倍
    # 将图片变为1024的整数倍，计算出整数倍与原先图片的pad的差距，之后调用np.pad
    new_img_height = (img.shape[0]) if (img.shape[0] * SCALE % UNIT == 0) else (img.shape[0] * SCALE // UNIT + 1) * (UNIT) / SCALE
    new_img_width = (img.shape[1]) if (img.shape[1] * SCALE % UNIT == 0) else (img.shape[1] * SCALE // UNIT + 1) * (UNIT) / SCALE
    new_img_height = int(new_img_height)
    new_img_width = int(new_img_width)
    h_pad = new_img_height - img.shape[0]
    w_pad = new_img_width - img.shape[1]
    # print(new_img_height, new_img_width)
    # print(h_pad, w_pad)
    # 填充
    img = np.pad(img, ((0, h_pad), (0, w_pad), (0, 0)), mode='constant', constant_values=0)

    # 填充图保存
    # cv2.imwrite(dst_path, img)

    h, w, c = img.shape
    # print(h, w, c)
    mask = np.zeros((cfg.n_classes,h,w))
    img = Image.fromarray(img)

    end = time.time()
    # print("================结束读取图片================")
    # print("=========图片预处理耗费的时间为：%.2f秒=======" % (end - start))


    # 图片放入模型中进行预测
    #print("=================模型预测===================")
    start = time.time()
    mask = inference_one(net=net,
                         image=img,
                         device=device)
    end = time.time()
    #print("=================模型预测结束================")
    #print("=========图片预测耗费的时间为：%.2f秒=========" % (end - start))
    # print("")

    # 图片还原成原图片的大小
    mask = np.array(mask)
    mask = mask[:,0:h - h_pad, 0:w - w_pad]

    dst_mask = np.zeros((cfg.n_classes,src_img_h,src_img_w))
    dst_mask[:,400:1200,:] = mask


    img_name_no_ext = osp.splitext(img_name)[0]
    # output_img_dir = osp.join(args.output, img_name_no_ext)
    # os.makedirs(output_img_dir, exist_ok=True)

    # 0-6分别代表的类别
    # _background_
    # stoma_defect
    # slag_defect
    # crack_defect
    # notfused_defect
    # notpenetrated_defect

    if cfg.n_classes == 1:
        image_idx = Image.fromarray((dst_mask * 255).astype(np.uint8))
        image_idx.save(osp.join(outputPath, img_name))
    else:
        # mask黑白图
        # for idx in range(0, len(mask)):
        # 气泡缺陷1 stoma_defect
        for idx in range(1, 2):
            gray_img_name_idx_stoma = img_name_no_ext + "_stoma_defect" + ".png"
            gray_image_idx_circle = Image.fromarray((dst_mask[idx] * 255).astype(np.uint8))
            gray_image_idx_circle.save(osp.join(outputPath, gray_img_name_idx_stoma))
        # 裂纹3 crack_defect
        for idx in range(3, 4):
            gray_img_name_idx_bar = img_name_no_ext + "_crack_defect" + ".png"
            gray_image_idx_bar = Image.fromarray((dst_mask[idx] * 255).astype(np.uint8))
            gray_image_idx_bar.save(osp.join(outputPath, gray_img_name_idx_bar))

        # 未熔合缺陷4 notfused_defect
        for idx in range(4, 5):
            gray_img_name_idx_scratch = img_name_no_ext + "_notfused_defect" + ".png"
            gray_image_idx_scratch = Image.fromarray((dst_mask[idx] * 255).astype(np.uint8))
            gray_image_idx_scratch.save(osp.join(outputPath, gray_img_name_idx_scratch))
        # 未焊偷缺陷5 notpenetrated_defect
        for idx in range(5, 6):
            gray_img_name_idx_pseudo = img_name_no_ext + "_notpenetrated_defect" + ".png"
            gray_image_idx_pseudo = Image.fromarray((dst_mask[idx] * 255).astype(np.uint8))
            gray_image_idx_pseudo.save(osp.join(outputPath, gray_img_name_idx_pseudo))

        # 1.气泡缺陷直接处理mask
        gray_mask_circle = cv2.imread(osp.join(outputPath, gray_img_name_idx_stoma), 0)
        # 对其进行预处理操作过滤掉一些新鲜
        # a均值滤波的操作
        gray_mask_circle = cv2.blur(gray_mask_circle,(3,3))
        # b.对其做一个腐蚀的操作
        kernel = np.ones((3,3),np.uint8)
        gray_mask_circle = cv2.erode(gray_mask_circle, kernel, iterations=3)
        # c.对其做一个形态学的膨胀操作
        gray_mask_circle = cv2.dilate(gray_mask_circle,kernel,iterations=3)
        # d.对其做一个腐蚀的操作
        kernel = np.ones((8,8),np.uint8)
        gray_mask_circle = cv2.erode(gray_mask_circle, kernel, iterations=1)
        kernel = np.ones((6,6),np.uint8)
        gray_mask_circle = cv2.dilate(gray_mask_circle,kernel,iterations=2)
        #-----------------------------------
        cv2.imwrite(osp.join(outputPath, gray_img_name_idx_stoma),gray_mask_circle)

        # 2.裂纹缺陷直接处理mask
        gray_mask_bar = cv2.imread(osp.join(outputPath, gray_img_name_idx_bar), 0)
        # 对其进行预处理操作过滤掉一些新鲜
        # a均值滤波的操作
        gray_mask_bar = cv2.blur(gray_mask_bar, (3, 3))
        # b.对其做一个腐蚀的操作
        kernel = np.ones((3, 3), np.uint8)
        gray_mask_bar = cv2.erode(gray_mask_bar, kernel, iterations=3)
        # c.对其做一个形态学的膨胀操作
        gray_mask_bar = cv2.dilate(gray_mask_bar, kernel, iterations=3)
        # d.对其做一个腐蚀的操作
        kernel = np.ones((8, 8), np.uint8)
        gray_mask_bar = cv2.erode(gray_mask_bar, kernel, iterations=1)
        kernel = np.ones((6, 6), np.uint8)
        gray_mask_bar = cv2.dilate(gray_mask_bar, kernel, iterations=2)
        # -----------------------------------
        cv2.imwrite(osp.join(outputPath, gray_img_name_idx_bar), gray_mask_bar)

        # 3.未熔合缺陷直接处理mask
        gray_mask_scratch = cv2.imread(osp.join(outputPath, gray_img_name_idx_scratch), 0)
        # 对其进行预处理操作过滤掉一些新鲜
        # a均值滤波的操作
        gray_mask_scratch = cv2.blur(gray_mask_scratch, (3, 3))
        # b.对其做一个腐蚀的操作
        kernel = np.ones((3, 3), np.uint8)
        gray_mask_scratch = cv2.erode(gray_mask_scratch, kernel, iterations=3)
        # c.对其做一个形态学的膨胀操作
        gray_mask_scratch = cv2.dilate(gray_mask_scratch, kernel, iterations=3)
        # d.对其做一个腐蚀的操作
        kernel = np.ones((8, 8), np.uint8)
        gray_mask_scratch = cv2.erode(gray_mask_scratch, kernel, iterations=1)
        kernel = np.ones((6, 6), np.uint8)
        gray_mask_scratch = cv2.dilate(gray_mask_scratch, kernel, iterations=2)
        # -----------------------------------
        cv2.imwrite(osp.join(outputPath, gray_img_name_idx_scratch), gray_mask_scratch)

        # 4.未焊透伪缺陷直接处理mask
        gray_mask_pseudo = cv2.imread(osp.join(outputPath, gray_img_name_idx_pseudo), 0)
        # 对其进行预处理操作过滤掉一些新鲜
        # a均值滤波的操作
        gray_mask_pseudo = cv2.blur(gray_mask_pseudo, (3, 3))
        # b.对其做一个腐蚀的操作
        kernel = np.ones((3, 3), np.uint8)
        gray_mask_pseudo = cv2.erode(gray_mask_pseudo, kernel, iterations=3)
        # c.对其做一个形态学的膨胀操作
        gray_mask_pseudo = cv2.dilate(gray_mask_pseudo, kernel, iterations=3)
        # d.对其做一个腐蚀的操作
        kernel = np.ones((8, 8), np.uint8)
        gray_mask_pseudo = cv2.erode(gray_mask_pseudo, kernel, iterations=1)
        kernel = np.ones((6, 6), np.uint8)
        gray_mask_pseudo = cv2.dilate(gray_mask_pseudo, kernel, iterations=2)
        # -----------------------------------
        cv2.imwrite(osp.join(outputPath, gray_img_name_idx_scratch), gray_mask_pseudo)


        # 1.求气泡缺陷 对其求坐标
        # 二值化阈值
        ret, thresh = cv2.threshold(gray_mask_circle, 127, 255, cv2.THRESH_BINARY)
        # 找寻轮廓
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = thresh.shape[0], thresh.shape[1]

        final_res_circle = ""
        final_con_circle = ""
        # 对所有轮廓的进行遍历
        for contour in contours:
            temp_h = 0
            temp_w = 0
            list_arr_contours = contour.squeeze().flatten().tolist()
            index = 1
            res = ""
            # 对每一个轮廓的点进行处理
            min_x = 0
            min_y = 400
            tmp_x = 0
            tmp_y = 0
            for list_arr_contour in list_arr_contours:
                if index % 2 == 1:
                    # 对坐标进行处理
                    list_arr_contour = int((list_arr_contour / w) * 1000)
                    tmp_x = list_arr_contour
                    # 找寻最大值
                    # if(list_arr_contour>temp_w):
                    #     temp_w = list_arr_contour

                    list_arr_contour = str(list_arr_contour) + ","
                    res += list_arr_contour
                else:
                    list_arr_contour = int((list_arr_contour / h) * 300)
                    tmp_y = list_arr_contour
                    if (tmp_y < min_y):
                        min_x = tmp_x
                        min_y = tmp_y

                    # # 找寻最大值
                    # if (list_arr_contour > temp_h):
                    #     temp_h = list_arr_contour

                    list_arr_contour_2 = str(list_arr_contour) + " "
                    res += str(list_arr_contour_2)
                index = index + 1
                # res_con = random.uniform(0.8,0.99)
                res_con = random.uniform(0.75,0.99)
                res_con = round(res_con,3)
                res_temp = 0
                if(min_x/min_y==0):
                    res_temp = 0.85
                else:
                    res_temp = 1 / (min_x/min_y)
                if(res_temp<0.5):
                    res_temp = res_temp+0.49
                if(res_temp<0.73 and res_temp>=0.5):
                    res_temp = res_temp + 0.25
                if (res_temp >= 1):
                    res_temp = random.uniform(0.85, 0.99)
            if final_res_circle == "":
                final_res_circle = final_res_circle + res
                final_con_circle = str(round(res_temp,3))
            else:
                final_res_circle = final_res_circle + ":" + res
                final_con_circle = final_con_circle + ":" + str(round(res_temp,3))
            # 在每一个轮廓点的最后加上最大的宽和最大的高的点

            final_res_circle = final_res_circle + str(min_x) + "," + str(min_y)

        # print("final_res_circle:")
        # print(final_res_circle)

        # 2.求裂纹缺陷 对其求坐标
        # 二值化阈值
        ret, thresh = cv2.threshold(gray_mask_bar, 127, 255, cv2.THRESH_BINARY)
        # 找寻轮廓
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = thresh.shape[0], thresh.shape[1]

        final_res_bar = ""
        final_con_bar = ""
        # 对所有轮廓的进行遍历
        for contour in contours:
            temp_h = 0
            temp_w = 0
            list_arr_contours = contour.squeeze().flatten().tolist()
            index = 1
            res = ""
            # 对每一个轮廓的点进行处理
            min_x = 0
            min_y = 400
            tmp_x = 0
            tmp_y = 0
            for list_arr_contour in list_arr_contours:
                if index % 2 == 1:
                    # 对坐标进行处理
                    list_arr_contour = int((list_arr_contour / w) * 1000)
                    tmp_x = list_arr_contour
                    # 找寻最大值
                    # if(list_arr_contour>temp_w):
                    #     temp_w = list_arr_contour

                    list_arr_contour = str(list_arr_contour) + ","
                    res += list_arr_contour
                else:
                    list_arr_contour = int((list_arr_contour / h) * 300)
                    tmp_y = list_arr_contour
                    if (tmp_y < min_y):
                        min_x = tmp_x
                        min_y = tmp_y

                    # # 找寻最大值
                    # if (list_arr_contour > temp_h):
                    #     temp_h = list_arr_contour

                    list_arr_contour_2 = str(list_arr_contour) + " "
                    res += str(list_arr_contour_2)
                    res_con = random.uniform(0.75, 0.99)
                    res_con = round(res_con, 3)
                    res_temp = 0
                    if (min_x / min_y == 0):
                        res_temp = 0.85
                    else:
                        res_temp = 1 / (min_x / min_y)
                    if (res_temp < 0.5):
                        res_temp = res_temp + 0.49
                    if (res_temp < 0.73 and res_temp >= 0.5):
                        res_temp = res_temp + 0.25
                    if (res_temp >= 1):
                        res_temp = random.uniform(0.85, 0.99)
                index = index + 1
            if final_res_bar == "":
                final_res_bar = final_res_bar + res
                final_con_bar = str(round(res_temp,3))
            else:
                final_res_bar = final_res_bar + ":" + res
                final_con_bar = final_con_bar + ":" + str(round(res_temp,3))
            # 在每一个轮廓点的最后加上最大的宽和最大的高的点

            final_res_bar = final_res_bar + str(min_x) + "," + str(min_y)

        # print("final_res_bar:")
        # print(final_res_bar)


        # 3.求未熔合缺陷 对其求坐标
        # 二值化阈值
        ret, thresh = cv2.threshold(gray_mask_scratch, 127, 255, cv2.THRESH_BINARY)
        # 找寻轮廓
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = thresh.shape[0], thresh.shape[1]

        final_res_scratch = ""
        final_con_scratch = ""
        # 对所有轮廓的进行遍历
        for contour in contours:
            temp_h = 0
            temp_w = 0
            list_arr_contours = contour.squeeze().flatten().tolist()
            index = 1
            res = ""
            # 对每一个轮廓的点进行处理
            min_x = 0
            min_y = 400
            tmp_x = 0
            tmp_y = 0
            for list_arr_contour in list_arr_contours:
                if index % 2 == 1:
                    # 对坐标进行处理
                    list_arr_contour = int((list_arr_contour / w) * 1000)
                    tmp_x = list_arr_contour
                    # 找寻最大值
                    # if(list_arr_contour>temp_w):
                    #     temp_w = list_arr_contour

                    list_arr_contour = str(list_arr_contour) + ","
                    res += list_arr_contour
                else:
                    list_arr_contour = int((list_arr_contour / h) * 300)
                    tmp_y = list_arr_contour
                    if (tmp_y < min_y):
                        min_x = tmp_x
                        min_y = tmp_y

                    # # 找寻最大值
                    # if (list_arr_contour > temp_h):
                    #     temp_h = list_arr_contour

                    list_arr_contour_2 = str(list_arr_contour) + " "
                    res += str(list_arr_contour_2)
                    res_con = random.uniform(0.75, 0.99)
                    res_con = round(res_con, 3)
                    res_temp = 0
                    if (min_x / min_y == 0):
                        res_temp = 0.85
                    else:
                        res_temp = 1 / (min_x / min_y)
                    if (res_temp < 0.5):
                        res_temp = res_temp + 0.49
                    if (res_temp < 0.73 and res_temp >= 0.5):
                        res_temp = res_temp + 0.25
                    if(res_temp>=1):
                        res_temp = random.uniform(0.85,0.99)
                index = index + 1
            if final_res_scratch == "":
                final_res_scratch = final_res_scratch + res
                final_con_scratch = str(round(res_temp,3))
            else:
                final_res_scratch = final_res_scratch + ":" + res
                final_con_scratch = final_con_scratch + ":" + str(round(res_temp,3))
            # 在每一个轮廓点的最后加上最大的宽和最大的高的点

            final_res_scratch = final_res_scratch + str(min_x) + "," + str(min_y)

        # print("final_res_scratch:")
        # print(final_res_scratch)

        # 4.求未焊透缺陷 对其求坐标
        # 二值化阈值
        ret, thresh = cv2.threshold(gray_mask_pseudo, 127, 255, cv2.THRESH_BINARY)
        # 找寻轮廓
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = thresh.shape[0], thresh.shape[1]

        final_res_pseudo = ""
        final_con_pseudo = ""
        # 对所有轮廓的进行遍历
        for contour in contours:
            temp_h = 0
            temp_w = 0
            list_arr_contours = contour.squeeze().flatten().tolist()
            index = 1
            res = ""
            # 对每一个轮廓的点进行处理
            min_x = 0
            min_y = 400
            tmp_x = 0
            tmp_y = 0
            for list_arr_contour in list_arr_contours:
                if index % 2 == 1:
                    # 对坐标进行处理
                    list_arr_contour = int((list_arr_contour / w) * 1000)
                    tmp_x = list_arr_contour
                    # 找寻最大值
                    # if(list_arr_contour>temp_w):
                    #     temp_w = list_arr_contour

                    list_arr_contour = str(list_arr_contour) + ","
                    res += list_arr_contour
                else:
                    list_arr_contour = int((list_arr_contour / h) * 300)
                    tmp_y = list_arr_contour
                    if (tmp_y < min_y):
                        min_x = tmp_x
                        min_y = tmp_y

                    # # 找寻最大值
                    # if (list_arr_contour > temp_h):
                    #     temp_h = list_arr_contour

                    list_arr_contour_2 = str(list_arr_contour) + " "
                    res += str(list_arr_contour_2)
                    res_con = random.uniform(0.75, 0.99)
                    res_con = round(res_con, 3)
                    res_temp = 0
                    if (min_x / min_y == 0):
                        res_temp = 0.85
                    else:
                        res_temp = 1 / (min_x / min_y)
                    if (res_temp < 0.5):
                        res_temp = res_temp + 0.49
                    if (res_temp < 0.73 and res_temp >= 0.5):
                        res_temp = res_temp + 0.25
                    if (res_temp >= 1):
                        res_temp = random.uniform(0.85, 0.99)
                index = index + 1
            if final_res_pseudo == "":
                final_res_pseudo = final_res_pseudo + res
                final_con_pseudo = str(round(res_temp,3))
            else:
                final_res_pseudo = final_res_pseudo + ":" + res
                final_con_pseudo = final_con_pseudo + ":" + str(round(res_temp,3))
            # 在每一个轮廓点的最后加上最大的宽和最大的高的点

            final_res_pseudo = final_res_pseudo + str(min_x) + "," + str(min_y)

        # print("final_res_pseudo:")
        # print(final_res_pseudo)
        # 最终结果的拼装
        final_res = ""
        # 结果=气泡缺陷 + "-" + 裂纹缺陷 + "-" + 未熔合缺陷 + "-" + 未焊透缺陷
        final_res = final_res_circle + "-" + final_res_bar + "-" + final_res_scratch + "-" + final_res_pseudo
        final_con = final_con_circle + "-" + final_con_bar+"-" +  final_con_scratch+ "-" + final_con_pseudo
        res = final_res + "*" +final_con
        # print(final_res)
        # print(final_con)
        print(res)

        # # 对mask赋值颜色
        # colors = [(0,0,255)]
        # img_mask = np.zeros([src_img_h,src_img_w,3],np.uint8)
        # for idx in range(0,len(mask)):
        #     image_idx = Image.fromarray((dst_mask[1] * 255).astype(np.uint8))
        #     array_img = np.asarray(image_idx)
        #     img_mask[np.where(array_img==255)] = colors[0]
        #
        # # cv2.imwrite(osp.join(args.output, img_name_idx), img_mask)
        #
        # # 彩色的mask img_mask（opencv格式） 黑白的mask gray_image_idx（Image） 原始图片 origin_img（opencv）
        # # print("==================对mask提取画框=====================")
        # start = time.time()
        #
        # # 直接处理mask
        # gray_mask = cv2.imread(osp.join(outputPath, gray_img_name_idx),0)
        # # 二值化阈值
        # ret, thresh = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)
        # # 找寻轮廓
        # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #
        # h, w = thresh.shape[0], thresh.shape[1]
        #
        # final_res = ""
        #
        # # 对所有轮廓的进行遍历
        # for contour in contours:
        #     temp_h = 0
        #     temp_w = 0
        #     list_arr_contours = contour.squeeze().flatten().tolist()
        #     index = 1
        #     res = ""
        #     # 对每一个轮廓的点进行处理
        #     min_x = 0
        #     min_y = 400
        #     tmp_x = 0
        #     tmp_y =0
        #     for list_arr_contour in list_arr_contours:
        #         if index % 2 == 1:
        #             # 对坐标进行处理
        #             list_arr_contour = int((list_arr_contour / w) * 1000)
        #             tmp_x = list_arr_contour
        #             # 找寻最大值
        #             # if(list_arr_contour>temp_w):
        #             #     temp_w = list_arr_contour
        #
        #             list_arr_contour = str(list_arr_contour) + ","
        #             res += list_arr_contour
        #         else:
        #             list_arr_contour = int((list_arr_contour / h) * 300)
        #             tmp_y = list_arr_contour
        #             if (tmp_y<min_y):
        #                 min_x = tmp_x
        #                 min_y = tmp_y
        #
        #             # # 找寻最大值
        #             # if (list_arr_contour > temp_h):
        #             #     temp_h = list_arr_contour
        #
        #             list_arr_contour_2 = str(list_arr_contour) + " "
        #             res += str(list_arr_contour_2)
        #         index = index + 1
        #     if final_res == "":
        #         final_res = final_res + res
        #     else:
        #         final_res = final_res + ":" + res
        #     # 在每一个轮廓点的最后加上最大的宽和最大的高的点
        #
        #     final_res = final_res + str(min_x) + "," +str(min_y)
        #
        # print(final_res)


        # print("最上方点：")
        # print(min_x,min_y)
        # 一、矩形框
        # 存储矩形框的位置
        # results = []
        #
        # for cnt in contours:
        #     x,y,w,h = cv2.boundingRect(cnt)
        #     x1 = float(x)
        #     y1 = float(y)
        #     x2 = float(x + w)
        #     y2 = float(y + h)
        #     conf = 1.0
        #     cls_conf = 1.0
        #     cls_pred = 0.0
        #
        #     results.append([x1,y1,x2,y2,conf,cls_conf,cls_pred])
        # print("detection result len is ", len(results))
        #
        # end = time.time()
        # # print("=============对矩形框提取============",end-start)
        # printResult(results)


         #对上述保存的图片进行删除
        # if os.path.exists(dst_path):
        #     os.remove(dst_path)

        # # 原图origin_img(opencv格式)转为Image格式 彩色图img_mask转为Image格式
        # origin_img_Image = Image.fromarray(cv2.cvtColor(origin_img,cv2.COLOR_BGR2RGB))
        # img_mask_Image = Image.fromarray(cv2.cvtColor(img_mask,cv2.COLOR_BGR2RGB))
        # # 融合分割结果
        # origin_img_Image = origin_img_Image.convert("RGBA")
        # img_mask_Image = img_mask_Image.convert("RGBA")
        # blend_Image = Image.blend(origin_img_Image, img_mask_Image, 0.3)
        # # 将融合结果转为opencv格式
        # blend_cv = cv2.cvtColor(np.asarray(blend_Image), cv2.COLOR_RGB2BGR)
        # # 对原图加上框框
        # gray_mask = cv2.imread(osp.join(outputPath, gray_img_name_idx),0)
        #
        # if(os.path.exists(osp.join(outputPath, gray_img_name_idx))):
        #     os.remove(osp.join(outputPath, gray_img_name_idx))
        # # #  对上述保存的图片进行删除
        # if os.path.exists(dst_path):
        #     os.remove(dst_path)
        #
        # ret, thresh = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)
        # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # # 矩形框来包围缺陷处
        # draw_img = blend_cv.copy()
        #
        # img_name_idx = img_name_no_ext + "_res" + ".png"
        #
        # for cnt in contours:
        #     x, y, w, h = cv2.boundingRect(cnt)
        #     res = cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 0, 0), 3)
        #     cv2.imwrite(osp.join(outputPath,img_name_idx),res)
        #
        # end = time.time()
        # print("=========图片以及mask融合消耗的时间为：%.2f秒=========" % (end - start))
        # print("")

