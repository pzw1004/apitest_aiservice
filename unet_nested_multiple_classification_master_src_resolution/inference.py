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

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from unet import NestedUNet
from unet import UNet
from utils.dataset import BasicDataset
from config import UNetConfig


# 获得自定义的模型结构
cfg = UNetConfig()

# 单张图片的预测
def inference_one(net, image, device):
    # 网络置于验证
    net.eval()
    # 对图片进行读取处理
    img = torch.from_numpy(BasicDataset.preprocess(image, cfg.scale))
    img = img.unsqueeze(0)
    # 将图片置于设备上
    img = img.to(device=device, dtype=torch.float32)

    # 梯度下降
    with torch.no_grad():
        # 得到输出
        output = net(img)
        if cfg.deepsupervision:
            output = output[-1]
        # 二类及以上用softmax激活函数
        if cfg.n_classes > 1:
            probs = F.softmax(output, dim=1)
        # 一类用sigmoid激活函数
        else:
            probs = torch.sigmoid(output)

        # 去除多余的维度
        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image.size[1], image.size[0])),
                transforms.ToTensor()
            ]
        )
        # 一类
        if cfg.n_classes == 1:
            probs = tf(probs.cpu())
            mask = probs.squeeze().cpu().numpy()
            return mask > cfg.out_threshold
        #二类及以上
        else:
            masks = []
            for prob in probs:
                prob = tf(prob.cpu())
                mask = prob.squeeze().cpu().numpy()
                mask = mask > cfg.out_threshold
                masks.append(mask)
            return masks
  
# 获得参数
def get_args():

    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 训练好的模型位置
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    # 输入图片的位置
    parser.add_argument('--input', '-i', dest='input', type=str, default='',
                        help='Directory of input images')
    # 输出图片的位置
    parser.add_argument('--output', '-o', dest='output', type=str, default='',
                        help='Directory of ouput images')

    # 滑动窗口裁剪出来的unit高和宽
    parser.add_argument("--crop_height", type=int, default=512, help="the height of the crop img")
    parser.add_argument("--crop_width", type=int, default=512, help="the width of the crop img")
    # 滑动窗口的步长
    parser.add_argument("--step_h", type=int, default=512, help="the step h of the crop img")
    parser.add_argument("--step_w", type=int, default=512, help="the step w of the crop img")
    # pad这里值得是给图片补的pad 使原图片/unit可以整除 通過代碼計算得到
    parser.add_argument("--pad_h", type=int, default=0, help="the size of pad height")
    parser.add_argument("--pad_w", type=int, default=0, help="the size of pad width")

    return parser.parse_args()


# 主函数
if __name__ == "__main__":
    args = get_args()
    # 得到输入图片的路径
    input_imgs = os.listdir(args.input)
    # 获得模型
    net = eval(cfg.model)(cfg)
    # 日志信息
    logging.info("Loading model {}".format(args.model))
    # 判断gpu是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    # 加载模型
    net.load_state_dict(torch.load(args.model, map_location=device))
    # 日志信息
    logging.info("Model loaded !")
    # 对输入图片进行遍历
    for i, img_name in tqdm(enumerate(input_imgs)):
        # 打印日志
        logging.info("\nPredicting image {} ...".format(img_name))
        # 获得图片的具体路径位置
        img_path = osp.join(args.input, img_name)
        # 打开图片
        img = Image.open(img_path)
        img = np.asarray(img)
        # 将图片变为要裁剪unit的整数倍
        new_img_height = (img.shape[0] // args.crop_height) * args.crop_height if (
                    img.shape[0] // args.crop_height == 0) else (img.shape[0] // args.crop_height + 1) * args.crop_height
        new_img_width = (img.shape[1] // args.crop_width) * args.crop_width if (
                    img.shape[1] // args.crop_width == 0) else (img.shape[1] // args.crop_width + 1) * args.crop_width
        h_pad = new_img_height - img.shape[0]
        w_pad = new_img_width - img.shape[1]
        # 计算真实的pad
        args.pad_h = h_pad
        args.pad_w = w_pad
        # 填充
        img = np.pad(img, ((0, args.pad_h), (0, args.pad_w), (0, 0)), mode='constant', constant_values=0)
        h, w, c = img.shape
        print(h, w, c)

        # 定义总图片的mask
        mask = np.zeros((cfg.n_classes,h,w))

        # 定义需裁剪的h和w的开端index
        h_index = np.arange(0, h - args.crop_height, args.step_h)
        h_index = np.append(h_index, h - args.crop_height)
        w_index = np.arange(0, w - args.crop_width, args.step_w)
        w_index = np.append(w_index, w - args.crop_width)
        k = 0
        for i in h_index:
            for j in w_index:
                k = k + 1
                print('\rpredict {}/{} unit image'.format(k, len(h_index) * len(w_index)), end='', flush=True)

                img_unit = img[i:i + args.crop_height, j:j + args.crop_width, :]
                # 预测
                img_unit = Image.fromarray(img_unit)
                mask_unit = inference_one(net=net,
                                     image=img_unit,
                                     device=device)

                mask[:,i:i + args.crop_height, j:j + args.crop_width] = mask[:,i:i + args.crop_height,
                                                                        j:j + args.crop_width] + mask_unit

        mask = mask[:,0:h - args.pad_h, 0:w - args.pad_w]
        # 获取带预测图片的name
        img_name_no_ext = osp.splitext(img_name)[0]
        # output_img_dir = osp.join(args.output, img_name_no_ext)
        # os.makedirs(output_img_dir, exist_ok=True)

        if cfg.n_classes == 1:
            image_idx = Image.fromarray((mask * 255).astype(np.uint8))
            image_idx.save(osp.join(args.output, img_name))
        else:
            # for idx in range(0, len(mask)):
            for idx in range(1,2):
                img_name_idx = img_name_no_ext + "_" + str(idx) + ".png"
                image_idx = Image.fromarray((mask[idx] * 255).astype(np.uint8))
                image_idx.save(osp.join(args.output, img_name_idx))
