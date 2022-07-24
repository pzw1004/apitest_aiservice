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
        img = Image.open(img_path)
        img = np.asarray(img)
        # 将图片变为要32的整数倍
        # 将图片变为1024的整数倍，计算出整数倍与原先图片的pad的差距，之后调用np.pad
        new_img_height = (img.shape[0]) if (img.shape[0] * SCALE % UNIT == 0) else (img.shape[0] * SCALE // UNIT + 1) * (UNIT) / SCALE
        new_img_width = (img.shape[1]) if (img.shape[1] * SCALE % UNIT == 0) else (img.shape[1] * SCALE // UNIT + 1) * (UNIT) / SCALE

        new_img_height = int(new_img_height)
        new_img_width = int(new_img_width)

        h_pad = new_img_height - img.shape[0]
        w_pad = new_img_width - img.shape[1]
        print(new_img_height, new_img_width)
        print(h_pad, w_pad)

        # 填充
        img = np.pad(img, ((0, h_pad), (0, w_pad), (0, 0)), mode='constant', constant_values=0)
        h, w, c = img.shape
        print(h, w, c)
        mask = np.zeros((cfg.n_classes,h,w))


        img = Image.fromarray(img)
        mask = inference_one(net=net,
                             image=img,
                             device=device)
        mask = np.array(mask)
        mask = mask[:,0:h - h_pad, 0:w - w_pad]

        img_name_no_ext = osp.splitext(img_name)[0]
        # output_img_dir = osp.join(args.output, img_name_no_ext)
        # os.makedirs(output_img_dir, exist_ok=True)

        if cfg.n_classes == 1:
            image_idx = Image.fromarray((mask * 255).astype(np.uint8))
            image_idx.save(osp.join(args.output, img_name))
        else:
            # for idx in range(0, len(mask)):
            for idx in range(1, 2):
                img_name_idx = img_name_no_ext + "_" + str(idx) + ".png"
                image_idx = Image.fromarray((mask[idx] * 255).astype(np.uint8))
                image_idx.save(osp.join(args.output, img_name_idx))
