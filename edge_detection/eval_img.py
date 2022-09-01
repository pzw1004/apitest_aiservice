import torch
from torch._C import device
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, dataloader
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
import shutil
import argparse
import datetime,cv2
from dataset import get_transforms
from torch.utils.data.sampler import BatchSampler
from tqdm import tqdm
from dataset import MYDataSet
from models.UNet_3Plus import UNet_3Plus
import time
def tensor2img(one_tensor):# [b,c,h,w] [-1,1]
    tensor = one_tensor.squeeze(0) #[c,h,w] [0,1]
    tensor = (tensor*0.5 + 0.5)*255 # [c,h,w] [0,255]
    tensor_cpu = tensor.cpu()
    img = np.array(tensor_cpu,dtype=np.uint8)
    img = np.transpose(img,(1,2,0))
    return img

def img2tensor(np_img):# [h,w,c]
    tensor = get_transforms()(np_img).cuda() # [c,h,w] [-1,1]
    tensor = tensor.unsqueeze(0) # [b,c,h,w] [-1,1]
    return tensor

parser = argparse.ArgumentParser()
parser.add_argument('--img_path',type=str,default='/home/zxl/yolo_data/images/104.jpg',help='Input the image path')
parser.add_argument('--output_folder',type=str,default='./output',help='output folder')
parser.add_argument('--checkpoint',type=str,default='checkpoints/2022-07-28_22_18_11/chk_499.pth',help='checkpoint for generator')
args = parser.parse_args()

if __name__ == "__main__":
    netG = UNet_3Plus().cuda()
    netG.eval()
    with torch.no_grad():
        root = os.getcwd()
        checkpoint_path = os.path.join(root,args.checkpoint)
        checkpoint = torch.load(checkpoint_path)
        netG.load_state_dict(checkpoint)

        img_path = args.img_path
        img_name = img_path.split('/')[-1]
        img = cv2.imread(img_path)
        h,w,c = img.shape
        h=300
        w=1000
        img = cv2.resize(img,(640,224),cv2.INTER_CUBIC)
        img_tensor = img2tensor(img).cuda()
        output_tensor = netG.forward(img_tensor)
        output_img = tensor2img(output_tensor)
        output_img = np.repeat(output_img,3,2)

        output_img = cv2.resize(output_img,(w,h),cv2.INTER_CUBIC)
        save_folder = args.output_folder
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        output_img_binary = output_img.copy()

        gray = cv2.cvtColor(output_img_binary,cv2.COLOR_BGR2GRAY)  
        ret, binary = cv2.threshold(gray,130,255,cv2.THRESH_BINARY) 
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        clean_contours=()
        zhouchanglist = []

        for value in contours:
                zhouchang=cv2.arcLength(curve=value,closed=True)
                zhouchanglist.append(zhouchang)
                if zhouchang>1667:
                    if len(clean_contours)==0:
                        clean_contours=(value,)
                    else:    
                        clean_contours+=(value,)

        for one_contour in clean_contours:
            output_str = ''
            for one_zu in one_contour:
                x,y = one_zu[0]
                output_str+= str(x)+','+str(y)+' '
            print(output_str)
        cv2.drawContours(output_img_binary,clean_contours,-1,(0,0,255),3)  

        img = cv2.resize(img,(w,h),cv2.INTER_CUBIC)
        output_img = np.concatenate((img,output_img_binary),axis=1)

        save_path = os.path.join(save_folder,img_name)
        cv2.imwrite(save_path,output_img)
        # if not os.path.exists('binary'):
        #     os.makedirs('binary')
        # cv2.imwrite(os.path.join('binary',img_name),output_img_binary)
