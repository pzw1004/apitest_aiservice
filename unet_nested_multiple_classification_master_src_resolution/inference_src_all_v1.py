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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import time
from unet import NestedUNet
from unet import UNet
from utils.dataset import BasicDataset
from config import UNetConfig

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
  
  
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='./data/checkpoints/epoch_900.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', dest='input', type=str, default='./data/test/input',
                        help='Directory of input images')
    parser.add_argument('--output', '-o', dest='output', type=str, default='./data/test/output',
                        help='Directory of ouput images')
    return parser.parse_args()

# 定义图片整除的数
UNIT = 32
# 定義縮放的比例
SCALE = 0.25


# 定义一个直方图均衡化的操作函数（图片预处理操作）
def equalize_transfrom(gray_img):
    return cv2.equalizeHist(gray_img)




if __name__ == "__main__":
    args = get_args()

    input_imgs = os.listdir(args.input)

    net = eval(cfg.model)(cfg)
    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, img_name in tqdm(enumerate(input_imgs)):
        logging.info("\nPredicting image {} ...".format(img_name))

        img_path = osp.join(args.input, img_name)


        # 打开图片
        start = time.time()
        print("")
        print("===============开始读取图片================")
        src_img = cv2.imread(img_path,1)
        src_img_h = src_img.shape[0]
        src_img_w = src_img.shape[1]
        print("================图片预处理=================")

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
        img_name = img_name.split(".")[0]
        img_name = img_name + ".png"
        dst_path = os.path.join(args.input,img_name)
        cv2.imwrite(dst_path,dst)
        # 对图片重新读取并进行填充成32整数的操作
        img = Image.open(dst_path)
        img = np.asarray(img)
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
        cv2.imwrite(dst_path, img)
        h, w, c = img.shape
        # print(h, w, c)
        mask = np.zeros((cfg.n_classes,h,w))
        img = Image.fromarray(img)

        end = time.time()
        print("================结束读取图片================")
        print("=========图片预处理耗费的时间为：%.2f秒=======" % (end - start))


        # 图片放入模型中进行预测
        print("=================模型预测===================")
        start = time.time()
        mask = inference_one(net=net,
                             image=img,
                             device=device)
        end = time.time()
        print("=================模型预测结束================")
        print("=========图片预测耗费的时间为：%.2f秒=========" % (end - start))
        print("")

        # 图片还原成原图片的大小
        mask = np.array(mask)
        mask = mask[:,0:h - h_pad, 0:w - w_pad]

        dst_mask = np.zeros((cfg.n_classes,src_img_h,src_img_w))
        dst_mask[:,400:1200,:] = mask


        img_name_no_ext = osp.splitext(img_name)[0]
        # output_img_dir = osp.join(args.output, img_name_no_ext)
        # os.makedirs(output_img_dir, exist_ok=True)

        if cfg.n_classes == 1:
            image_idx = Image.fromarray((dst_mask * 255).astype(np.uint8))
            image_idx.save(osp.join(args.output, img_name))
        else:
            # for idx in range(0, len(mask)):
            for idx in range(1, 2):
                img_name_idx = img_name_no_ext + "_" + str(idx) + ".png"
                image_idx = Image.fromarray((dst_mask[idx] * 255).astype(np.uint8))
                image_idx.save(osp.join(args.output, img_name_idx))
