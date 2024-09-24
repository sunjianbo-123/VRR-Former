import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmcv.cnn import build_norm_layer
import math
" pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html"

"SEAFORMER: SQUEEZE-ENHANCED AXIAL TRANSFORMER FOR MOBILE SEMANTIC SEGMENTATION"


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)



class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]), requires_grad=True)

    def forward(self, x):
        # (B,C_qk,H)
        B, C, N = x.shape
        x = x + F.interpolate(self.pos_embed, size=(N), mode='linear', align_corners=False)
        return x



class Sea_Attention(torch.nn.Module):
    def __init__(self, dim, win_size, num_heads,
                 attn_ratio=2,
                 activation=nn.ReLU,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 norm_cfg=dict(type='BN', requires_grad=True), ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        key_dim = dim // num_heads
        self.scale = key_dim ** -0.5
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)



        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_row = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.proj_encode_column = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.dwconv = Conv2d_BN(2 * self.dh, 2 * self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.dh, norm_cfg=norm_cfg)
        self.act = activation()
        self.pwconv = Conv2d_BN(2 * self.dh, dim, ks=1, norm_cfg=norm_cfg)
        # self.sigmoid = torch.sigmoid()
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # batch_size*num_windows, window_h*window_w, embed_dim
        B_, N, C = x.shape
        H, W = int(math.sqrt(N)), int(math.sqrt(N))
        assert H * W == N, "input feature has wrong size"
        x = x.view(B_, C, H, W)   # batch_size*num_windows,embed_dim, window_h, window_w

        q = self.to_q(x)  # 生成query: (B_,C,H,W)-->(B_,C_qk,H,W)
        k = self.to_k(x)  # 生成key: (B_,C,H,W)-->(B_,C_qk,H,W)
        v = self.to_v(x)  # 生成value: (B_,C,H,W)-->(B_,C_v,H,W)

        # Detail enhancement kernel
        qkv = torch.cat([q, k, v], dim=1)  # 将qkv拼接: (B,2*C_qk+C_v,H,W)
        qkv = self.act(self.dwconv(qkv))   # 执行3×3卷积,建模局部空间依赖,从而增强局部细节感知: (B,2*C_qk+C_v,H,W)-->(B,2*C_qk+C_v,H,W)
        qkv = self.pwconv(qkv)             # 执行1×1卷积,将通道数量从(2*C_qk+C_v)映射到C,从而生成细节增强特征: (B,2C_qk+C_v,H,W)-->(B,C,H,W)

        # squeeze axial attention
        ## squeeze row, squeeze操作将全局信息保留到单个轴上，然后分别应用自注意力建模对应轴的长期依赖
        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B_, self.num_heads, -1, H).permute(0, 1, 3, 2) #通过平均池化压缩水平方向,并为垂直方向的空间位置添加位置嵌入: (B,C_qk,H,W)-->mean-->(B,C_qk,H)-->reshape-->(B,h,d,H)-->permute-->(B,h,H,d);   C_qk=h*d, h:注意力头的个数；d:每个注意力头的通道数
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(B_, self.num_heads, -1, H) #通过平均池化压缩水平方向,并为垂直方向的空间位置添加位置嵌入: (B,C_qk,H,W)-->mean-->(B,C_qk,H)-->reshape-->(B,h,d,H)
        vrow = v.mean(-1).reshape(B_, self.num_heads, -1, H).permute(0, 1, 3, 2) #通过平均池化压缩水平方向: (B,C_v,H,W)-->mean-->(B,C_v,H)-->reshape-->(B,h,d_v,H)-->permute-->(B,h,H,d_v);   C_v=h*d_v, h:注意力头的个数；d_v:Value矩阵中每个注意力头的通道数

        attn_row = torch.matmul(qrow, krow) * self.scale  # 计算水平方向压缩之后的自注意力机制：(B,h,H,d) @ (B,h,d,H) = (B,h,H,H)
        attn_row = attn_row.softmax(dim=-1) # 执行softmax操作
        xx_row = torch.matmul(attn_row, vrow)  # 对Value进行加权求和: (B,h,H,H) @ (B,h,H,d_v) = (B,h,H,d_v)
        xx_row = self.proj_encode_row(xx_row.permute(0, 1, 3, 2).reshape(B_, self.dh, H, 1)) # 对注意力机制的输出进行reshape操作,并进行卷积：(B,h,H,d_v)-->permute-->(B,h,d_v,H)-->reshape-->(B,C_v,H,1);   C_v=h*d_v

        ## squeeze column
        qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(B_, self.num_heads, -1, W).permute(0, 1, 3, 2) # 通过平均池化压缩垂直方向,并为水平方向的空间位置添加位置嵌入: (B,C_qk,H,W)-->mean-->(B,C_qk,W)-->reshape-->(B,h,d,W)-->permute-->(B,h,W,d);  C_qk=h*d, h:注意力头的个数；d:每个注意力头的通道数
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B_, self.num_heads, -1, W) # 通过平均池化压缩垂直方向,并为水平方向的空间位置添加位置嵌入: (B,C_qk,H,W)-->mean-->(B,C_qk,W)-->reshape-->(B,h,d,W)
        vcolumn = v.mean(-2).reshape(B_, self.num_heads, -1, W).permute(0, 1, 3, 2)  #通过平均池化压缩垂直方向: (B,C_v,H,W)-->mean-->(B,C_v,W)-->reshape-->(B,h,d_v,W)-->permute-->(B,h,W,d_v)

        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale # 计算垂直方向压缩之后的自注意力机制：(B,h,W,d) @ (B,h,d,W) = (B,h,W,W)
        attn_column = attn_column.softmax(dim=-1) # 执行softmax操作
        xx_column = torch.matmul(attn_column, vcolumn)  # 对Value进行加权求和: (B,h,W,W) @ (B,h,W,d_v) = (B,h,W,d_v)
        xx_column = self.proj_encode_column(xx_column.permute(0, 1, 3, 2).reshape(B_, self.dh, 1, W)) # 对注意力机制的输出进行reshape操作,并进行卷积：(B,h,W,d_v)-->permute-->(B,h,d_v,W)-->reshape-->(B,C_v,1,W);  C_v=h*d_v

        xx = xx_row.add(xx_column) # 将两个注意力机制的输出进行相加,这是一种broadcast操作: (B,C_v,H,1) + (B,C_v,1,W) =(B,C_v,H,W)
        xx = v.add(xx) # 添加残差连接
        xx = self.proj(xx) # 应用1×1Conv得到Squeeze Axial attention的输出
        xx = xx.sigmoid() * qkv  # 为Squeeze Axial attention的输出应用门控机制获得权重, 然后与Detail enhancement kernel的输出进行逐点乘法
        xx = xx.flatten(2).permute(0, 2, 1).contiguous()    # 将输出展平,并转置: (B,C,H,W)-->flatten-->(B,C,H*W)-->transpose-->(B,H*W,C)
        # xx = nn.Linear(self.dim, self.dim)(xx)
        # xx = self.proj_drop(xx)
        return xx


if __name__ == '__main__':
    # (B,C,H,W)
    input=torch.randn(64, 64, 96)
    Model = Sea_Attention(dim=96, win_size=8, num_heads=3)
    output=Model(input)
    print(output.shape)

