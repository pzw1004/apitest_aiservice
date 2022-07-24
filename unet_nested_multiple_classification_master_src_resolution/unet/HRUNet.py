"""
# -*- coding: utf-8 -*-
# @Time : 2020/7/22 22:19
# @Author  : Codingchaozhang
# @File    : model.py
"""
import torch.nn as nn
import torch
# coordinate attention
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

# |主干VGG网络
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16,self).__init__()

        # (h,w,64)
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # (h/2,w/2,128)
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),
                        stride=(2,2))
        )

        self.coordatten3 = CoordAtt(inp=128,oup=128)
        # (h/4,w/4,256)
        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),
                         stride=(2,2))
        )

        # (h/8,w/8,512)
        self.coordatten4 = CoordAtt(inp=256,oup=256)
        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),
                         stride=(2,2))
        )

        # (h/16,w/16,512)
        self.coordatten5 = CoordAtt(inp=512,oup=512)
        self.block_5 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3,3),
                      stride=(1,1),
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),
                         stride=(2,2))
        )


    def forward(self,x):

        f1 = self.block_1(x)

        f2 = self.block_2(f1)

        f3 = self.block_3(f2)
        f3 = self.coordatten3(f3)

        f4 = self.block_4(f3)
        f4 = self.coordatten4(f4)

        f5 = self.block_5(f4)
        f5 = self.coordatten5(f5)

        # 返回结果
        return f1,f2,f3,f4,f5

# 边缘细化模块
# Boundary Refinement
class BR(nn.Module):
    def __init__(self,in_channels,middle_channels,out_channels):
        super(BR,self).__init__()
        self.act_func = nn.ReLU(inplace=True)
        self.conv1    = nn.Conv2d(in_channels,middle_channels,3,padding=1)
        self.bn1      = nn.BatchNorm2d(middle_channels)
        self.conv2    = nn.Conv2d(middle_channels,out_channels,3,padding=1)
        self.bn2      = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        identity = self.conv1(x)
        identity = self.bn1(identity)
        identity = self.act_func(identity)


        identity = self.conv2(identity)
        output  = x+identity
        return output

# thrilinear attention三线性注意力模型
# thrilinear attention三线性注意力模型
class tri_attention(nn.Module):
    def __init__(self):
        super(tri_attention, self).__init__()
        self.feature_norm = nn.Softmax(dim=2)
        self.bilinear_norm = nn.Softmax(dim=2)
    def forward(self, x):
        n = x.size(0)
        c = x.size(1)
        h = x.size(2)
        w = x.size(3)

        # f
        f = x.reshape(n,c,-1)

        # 激活
        f_norm = self.feature_norm(f*2)
        # 双线性
        bilinear = f_norm.bmm(f.transpose(1,2))
        bilinear = self.bilinear_norm(bilinear)
        # 三线性
        tri_atten = bilinear.bmm(f).view(n,c,h,w).detach()

        return tri_atten

def degree(x):
    a  = 0.5
    f1 = ( (torch.exp(x)) - (torch.exp(-x)) ) / ( (torch.exp(x)) + (torch.exp(-x)) )
    f1_2 = torch.pow(input=f1,exponent=2)
    f1_3 = a*(1-f1_2)
    return f1_3


# 获取距离差别的绝对值
def dist(x0,xb):
    d = torch.abs(x0-xb)
    return d

# 获取正负号的方向
def direction(x0,xb):
    dist_direction = ((x0-xb))/(torch.abs(x0-xb)+0.00001)
    return dist_direction

# MRFM
# class MRFM(nn.Module):
#     def __init__(self,in_channels,output_channels):
#         super(MRFM,self).__init__()
#
#         self.branch0 = nn.Sequential(
#             nn.Conv2d(in_channels,output_channels,kernel_size=1)
#         )
#
#         self.branch1 = nn.Sequential(
#             nn.Conv2d(in_channels,output_channels,kernel_size=3,padding=1)
#         )
#
#         self.branch2 = nn.Sequential(
#             nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
#             nn.Conv2d(in_channels,output_channels,kernel_size=1),
#             nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
#         )
#
#         # 最后的线性
#         self.ConvLinear = nn.Conv2d(3*in_channels,output_channels,kernel_size=1)
#     def forward(self,x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
#
#
#         # 拼接
#         output = torch.cat((x0,x1,x2),1)
#         output = self.ConvLinear(output)
#
#         return output
class DoubleConv(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    """
    def __init__(self, in_channels, output_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class HRUNet(nn.Module):
    def __init__(self, cfg):
        super(HRUNet, self).__init__()
        self.n_channels = cfg.n_channels
        self.n_classes = cfg.n_classes
        self.bilinear = cfg.bilinear
        self.deepsupervision = cfg.deepsupervision

        self.vgg = VGG16()

        # attention
        self.coordatten1 = CoordAtt(inp=32,oup=32)
        self.coordatten2 = CoordAtt(inp=64,oup=64)
        self.coordatten3 = CoordAtt(inp=128,oup=128)
        self.coordatten4 = CoordAtt(inp=256,oup=256)
        self.coordatten5 = CoordAtt(inp=512,oup=512)

        # 通道数量
        nb_filters = [32, 64, 128, 256, 512]
        self.br = BR(nb_filters[0],nb_filters[0],nb_filters[0])
        # 提升通道数
        self.add_channels_1 = nn.Conv2d(32,64,kernel_size=1)
        # 提升通道数
        self.add_channels_2 = nn.Conv2d(64,128,kernel_size=1)
        # 提升通道数
        self.add_channels_3 = nn.Conv2d(128,256,kernel_size=1)
        # 提升通道数
        self.add_channels_4 = nn.Conv2d(256,512,kernel_size=1)

        # 三线性注意力
        self.trilinear = tri_attention()
        # 第二次64通道
        self.se_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64,64//16,kernel_size=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64//16,64,kernel_size=1,padding=0),
            nn.Sigmoid()
        )
        # 第三层通道
        self.se_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128,128//16,kernel_size=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128//16,128,kernel_size=1,padding=0),
            nn.Sigmoid()
        )
        # 第四层
        self.se_3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 256 // 16, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256 // 16, 256, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        # 第五层
        self.se_4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 512 // 16, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 // 16, 512, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        # 对其gap的操作
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 对其最大池化
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        # 对第二层的MRFM
        self.MRFM2_3 = DoubleConv(in_channels=64,output_channels=64)
        # 对第三层的MRFM
        self.MRFM3_3 = DoubleConv(in_channels=128,output_channels=128)
        # 对第四层的MRFM
        self.MRFM4_3 = DoubleConv(in_channels=256,output_channels=256)
        # 对第五层的MRFM
        self.MRFM5_3 = DoubleConv(in_channels=512, output_channels=512)

        # 上采样的操作
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear",align_corners=True)
        self.up3 = nn.Upsample(scale_factor=4, mode="bilinear",align_corners=True)
        self.up4 = nn.Upsample(scale_factor=8, mode="bilinear",align_corners=True)
        self.up5 = nn.Upsample(scale_factor=16, mode="bilinear",align_corners=True)


        self.final1 = nn.Conv2d(nb_filters[0], self.n_classes, kernel_size=1)
        self.final2 = nn.Conv2d(nb_filters[1], self.n_classes, kernel_size=1)
        self.final3 = nn.Conv2d(nb_filters[2], self.n_classes, kernel_size=1)
        self.final4 = nn.Conv2d(nb_filters[3], self.n_classes, kernel_size=1)
        self.final5 = nn.Conv2d(nb_filters[4], self.n_classes, kernel_size=1)

        self.final = nn.Conv2d(self.n_classes, self.n_classes, kernel_size=1)

        # 深度监督
        # if self.deepsupervision:
        #     self.final1 = nn.Conv2d(nb_filters[0],self.n_classes,kernel_size=1)
        #     self.final2 = nn.Conv2d(nb_filters[1],self.n_classes,kernel_size=1)
        #     self.final3 = nn.Conv2d(nb_filters[2],self.n_classes,kernel_size=1)
        #     self.final4 = nn.Conv2d(nb_filters[3],self.n_classes,kernel_size=1)
        #     self.final5 = nn.Conv2d(nb_filters[4],self.n_classes,kernel_size=1)

    def forward(self, x):
        conv1_1, conv1_2, conv1_3, conv1_4,conv1_5 = self.vgg(x)

        # 对第一层的处理
        br1_1 = self.br(conv1_1)
        br1_2 = self.br(br1_1)
        br1_3 = self.br(br1_2)

        # 最后添加一层注意力
        br1_3 = self.coordatten1(br1_3)

        # 不需要上采样得到第一个输出
        output1 = self.final1(br1_3)

        # 第二层的处理
        br1_1_channel64 = self.add_channels_1(br1_1)
        br1_1_tril = self.trilinear(br1_1_channel64)
        # print("br1_1_tril:",br1_1_tril.shape)
        w1_2       = self.se_1(conv1_2)
        # print("w1_2:",w1_2.shape)
        # 得到第一个的值
        tlgam2_1 = br1_1_tril*w1_2
        # 对其求一个Vbackground
        vbg2_1  = self.gap(tlgam2_1)
        # print("vbg2_1:",vbg2_1.shape)
        # repeat操作使其与low-level的分辨率相同
        # print(vbg2_1)
        # print("==========repeat=========")
        br1_1_channel64_h = br1_1_channel64.shape[2]
        br1_1_channel64_w = br1_1_channel64.shape[3]
        vbg2_1_reapeat = vbg2_1.repeat(1,1,br1_1_channel64_h,br1_1_channel64_w)
        # print("vbg2_1_reapeat:", vbg2_1_reapeat.shape)
        # # repeat操作使其与low-level的分辨率相同
        # print(vbg2_1_reapeat)
        # 对其差值做一个程度上的增强
        # 1.先求出距离差别的绝对值
        dist_2_1_repeat = dist(br1_1_channel64, vbg2_1_reapeat)
        # 2.对差值做一个程度上的增强
        dist_2_1_repeat_aug = degree(dist_2_1_repeat)
        # 3.计算抑制方向
        dist_2_1_repeat_direction = direction(br1_1_channel64,vbg2_1_reapeat)
        # 4.抑制方向*抑制值+之前的值得到新的
        suppres_feature2_2 = br1_1_channel64 + (dist_2_1_repeat_direction*dist_2_1_repeat_aug)
        # 池化一下
        suppres_feature2_2_pool = self.max_pool(suppres_feature2_2)
        bs2_2 = suppres_feature2_2_pool + conv1_2

        # 加入CA协同注意力
        bs2_2 = self.coordatten2(bs2_2)

        # 送入MRFM
        mrfm2_3 = self.MRFM2_3(bs2_2)
        # 1*1 减少通道
        mrfm2_3_channel = self.final2(mrfm2_3)
        # 上采样得到输出结果
        output2 = self.up2(mrfm2_3_channel)

        # print("bs2_2.shape",bs2_2.shape)
        # 第三层的处理
        # 提升通道数
        bs2_2_channel128 = self.add_channels_2(bs2_2)
        # 浅层数据做三线性增强
        bs2_2_tril = self.trilinear(bs2_2_channel128)
        # 求深层的权重信息
        w2_2 = self.se_2(conv1_3)
        # 得到第一个
        tlgam3_1 = bs2_2_tril*w2_2
        # 对其求background
        vgb3_1 = self.gap(tlgam3_1)
        # 对其repeat
        bs2_2_channel128_h = bs2_2_channel128.shape[2]
        bs2_2_channel128_w = bs2_2_channel128.shape[3]
        vgb3_1_repeat = vgb3_1.repeat(1,1,bs2_2_channel128_h,bs2_2_channel128_w)
        # 对其做背景抑制
        # 1.求出距离差别的绝对值
        dist_3_1_repeat = dist(bs2_2_channel128,vgb3_1_repeat)
        # 2.对差值做一个程度上的增强
        dist_3_1_repeat_aug = degree(dist_3_1_repeat)
        # 3.计算抑制方向
        dist_3_1_repeat_direction = direction(bs2_2_channel128,vgb3_1_repeat)
        # 4.抑制方向*抑制值+之前的值得到最新的
        suppres_feature3_2 = bs2_2_channel128 + (dist_3_1_repeat_direction*dist_3_1_repeat_aug)
        # 池化一下
        suppres_feature3_2_pool = self.max_pool(suppres_feature3_2)
        bs3_2 = suppres_feature3_2_pool + conv1_3
        # 加入注意力机制
        bs3_2 = self.coordatten3(bs3_2)

        # 送入MRFM
        mrfm3_3 = self.MRFM3_3(bs3_2)
        # 1*1减少通道
        mrfm3_3_channel = self.final3(mrfm3_3)
        # 上采样得到输出结果
        output3 = self.up3(mrfm3_3_channel)


        # 第四层的处理
        # 提升通道数
        bs3_2_channel256 = self.add_channels_3(bs3_2)
        # 浅层做三线性增强
        bs3_2_tril = self.trilinear(bs3_2_channel256)
        # 求深层的权重信息
        w3_3 = self.se_3(conv1_4)
        # 得到第一个
        tlgam4_1 = bs3_2_tril*w3_3
        # 求background
        vgb4_1 = self.gap(tlgam4_1)
        # 对其reapeat
        bs3_2_channel1256_h = bs3_2_channel256.shape[2]
        bs3_2_channel1256_w = bs3_2_channel256.shape[3]
        vgb4_1_repeat = vgb4_1.repeat(1,1,bs3_2_channel1256_h,bs3_2_channel1256_w)
        # 对其做背景抑制
        # 1.求出距离差别的绝对值
        dist_4_1_repeat = dist(bs3_2_channel256,vgb4_1_repeat)
        # 2.对差值做一个程度上的增强
        dist_4_1_repeat_aug = degree(dist_4_1_repeat)
        # 3.计算抑制方向
        dist_4_1_repeat_direction = direction(bs3_2_channel256,vgb4_1_repeat)
        # 4.抑制方向*抑制值+之前得到的值
        suppres_feature4_2 = bs3_2_channel256 + (dist_4_1_repeat_direction*dist_4_1_repeat_aug)
        # 池化一下
        suppres_feature4_2_pool = self.max_pool(suppres_feature4_2)
        bs4_2 = suppres_feature4_2_pool + conv1_4
        # 送入MRFM
        bs4_2 = self.coordatten4(bs4_2)

        mrfm4_3 = self.MRFM4_3(bs4_2)
        # 1*1减少通道
        mrfm4_3_channel = self.final4(mrfm4_3)
        # 上采样得到输出结果
        output4 = self.up4(mrfm4_3_channel)

        # 第五层
        # 提升通道数
        bs4_2_channel1512 = self.add_channels_4(bs4_2)
        # 浅层做三线性增强
        bs4_2_tril = self.trilinear(bs4_2_channel1512)
        # 求深层的权重
        w4_3 = self.se_4(conv1_5)
        # 得到第一个
        tlgam5_1 = bs4_2_tril*w4_3
        # 求background
        vgb5_1 = self.gap(tlgam5_1)
        # 对其repeat
        bs4_2_channel1512_h = bs4_2_channel1512.shape[2]
        bs4_2_channel1512_w = bs4_2_channel1512.shape[3]
        vgb5_1_repeat = vgb5_1.repeat(1,1,bs4_2_channel1512_h,bs4_2_channel1512_w)
        # 对其做背景抑制
        # 1.求出距离差别的绝对值
        dist_5_1_repeat = dist(bs4_2_channel1512, vgb5_1_repeat)
        # 2.对差值做一个程度上的增强
        dist_5_1_repeat_aug = degree(dist_5_1_repeat)
        # 3.计算抑制方向
        dist_5_1_repeat_direction = direction(bs4_2_channel1512, dist_5_1_repeat)
        # 4.抑制方向*抑制值+之前得到的值
        suppres_feature5_2 = bs4_2_channel1512 + (dist_5_1_repeat_direction * dist_5_1_repeat_aug)
        # 池化一下
        suppres_feature5_2_pool = self.max_pool(suppres_feature5_2)
        bs5_2 = suppres_feature5_2_pool + conv1_5
        # 加入ca协同注意力
        bs5_2 = self.coordatten5(bs5_2)

        # 送入MRFM
        mrfm5_3 = self.MRFM5_3(bs5_2)
        # 1*1减少通道
        mrfm5_3_channel = self.final5(mrfm5_3)
        # 上采样得到输出结果
        output5 = self.up5(mrfm5_3_channel)

        # 最终输出
        output = output1 + output2 + output3 + output4 + output5
        output = self.final(output)
        # 监督
        # if self.deepsupervision:
        #     output1 = self.final1(br1_3)
        # else:
        #     output1 = self.final1(br1_3)
        # print("output1.shape", output1.shape)
        # print("output2.shape", output2.shape)
        # print("output3.shape",output3.shape)
        # print("output4.shape",output4.shape)
        # print("output5.shape",output5.shape)

        # 未上采样的
        # print("output1_ori.shape", output1.shape)
        # print("output2_ori.shape", mrfm2_3_channel.shape)
        # print("output3_ori.shape", mrfm3_3_channel.shape)
        # print("output4_ori.shape", mrfm4_3_channel.shape)
        # print("output5_ori.shape", mrfm5_3_channel.shape)
        # print("output.shape",output.shape)

        # output1.shape torch.Size([1, 3, 512, 256])
        # output2.shape torch.Size([1, 3, 512, 256])
        # output3.shape torch.Size([1, 3, 512, 256])
        # output4.shape torch.Size([1, 3, 512, 256])
        # output5.shape torch.Size([1, 3, 512, 256])
        # output1_ori.shape torch.Size([1, 3, 512, 256])
        # output2_ori.shape torch.Size([1, 3, 256, 128])
        # output3_ori.shape torch.Size([1, 3, 128, 64])
        # output4_ori.shape torch.Size([1, 3, 64, 32])
        # output5_ori.shape torch.Size([1, 3, 32, 16])
        # output.shape torch.Size([1, 3, 512, 256])
        if self.deepsupervision:
            return  [output, output1, output2, output3, output4, output5]
        else:
            return output


