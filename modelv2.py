import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
# from torch.utils.tensorboard import SummaryWriter



def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)
# padding = (kernel_size-1)//2  保证输出feature map尺寸不变


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.skip_layer = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)
        self.skip_layer11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        conv_out = self.conv_block(x)
        skip_out = self.skip_layer(x)
        skip_out1 = self.skip_layer11(x)
        out = conv_out + skip_out + skip_out1
        return out

    def flops(self, H, W):
        flops = H * W * self.in_channel * self.out_channel * (
                    3 * 3 + 1) + H * W * self.out_channel * self.out_channel * 3 * 3
        return flops


## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


class UNet(nn.Module):
    def __init__(self, block=ConvBlock, dim=32):
        super(UNet, self).__init__()

        self.dim = dim
        self.ConvBlock1 = ConvBlock(3, dim, strides=1)
        self.pool1 = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = block(dim, dim * 2, strides=1)
        self.pool2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=4, stride=2, padding=1)

        self.ConvBlock3 = block(dim * 2, dim * 4, strides=1)
        self.pool3 = nn.Conv2d(dim * 4, dim * 4, kernel_size=4, stride=2, padding=1)

        self.ConvBlock4 = block(dim * 4, dim * 8, strides=1)
        self.pool4 = nn.Conv2d(dim * 8, dim * 8, kernel_size=4, stride=2, padding=1)

        self.ConvBlock5 = block(dim * 8, dim * 16, strides=1)

        self.upv6 = nn.ConvTranspose2d(dim * 16, dim * 8, 2, stride=2)
        self.ConvBlock6 = block(dim * 16, dim * 8, strides=1)

        self.upv7 = nn.ConvTranspose2d(dim * 8, dim * 4, 2, stride=2)
        self.ConvBlock7 = block(dim * 8, dim * 4, strides=1)

        self.upv8 = nn.ConvTranspose2d(dim * 4, dim * 2, 2, stride=2)
        self.ConvBlock8 = block(dim * 4, dim * 2, strides=1)

        self.upv9 = nn.ConvTranspose2d(dim * 2, dim, 2, stride=2)
        self.ConvBlock9 = block(dim * 2, dim, strides=1)

        self.conv10 = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.ConvBlock1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.ConvBlock5(pool4)

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.ConvBlock9(up9)

        conv10 = self.conv10(conv9)
        out = x + conv10

        return out

    def flops(self, H, W):
        flops = 0
        flops += self.ConvBlock1.flops(H, W)
        flops += H / 2 * W / 2 * self.dim * self.dim * 4 * 4
        flops += self.ConvBlock2.flops(H / 2, W / 2)
        flops += H / 4 * W / 4 * self.dim * 2 * self.dim * 2 * 4 * 4
        flops += self.ConvBlock3.flops(H / 4, W / 4)
        flops += H / 8 * W / 8 * self.dim * 4 * self.dim * 4 * 4 * 4
        flops += self.ConvBlock4.flops(H / 8, W / 8)
        flops += H / 16 * W / 16 * self.dim * 8 * self.dim * 8 * 4 * 4

        flops += self.ConvBlock5.flops(H / 16, W / 16)

        flops += H / 8 * W / 8 * self.dim * 16 * self.dim * 8 * 2 * 2
        flops += self.ConvBlock6.flops(H / 8, W / 8)
        flops += H / 4 * W / 4 * self.dim * 8 * self.dim * 4 * 2 * 2
        flops += self.ConvBlock7.flops(H / 4, W / 4)
        flops += H / 2 * W / 2 * self.dim * 4 * self.dim * 2 * 2 * 2
        flops += self.ConvBlock8.flops(H / 2, W / 2)
        flops += H * W * self.dim * 2 * self.dim * 2 * 2
        flops += self.ConvBlock9.flops(H, W)

        flops += H * W * self.dim * 3 * 3 * 3
        return flops


class LPU(nn.Module):
    """
    Local Perception Unit to extract local infomation.
    LPU(X) = DWConv(X) + X
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(LPU, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, groups=in_channels, bias=True
                                   )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        result = (self.depthwise(x) + x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return result

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * W * self.out_channels * 3 * 3
        return flops


#########################################
class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim))
        self.s = s

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        H = H or int(math.sqrt(N))
        W = W or int(math.sqrt(N))
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1, act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x

    def flops(self, HW):
        flops = 0
        flops += HW * self.in_channels * self.kernel_size ** 2 / self.stride ** 2
        flops += HW * self.in_channels * self.out_channels
        print("SeqConv2d:{%.2f}" % (flops / 1e9))
        return flops


######## Embedding for q,k,v ########
class ConvProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout=0.,
                 last_stage=False, bias=True):
        super().__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        pad = (kernel_size - q_stride) // 2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x, attn_kv=None):
        # batch_size*num_windows, window_h*window_w, embed_dim
        b, n, c = x.shape
        h = self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q, k, v

    def flops(self, q_L, kv_L=None):
        kv_L = kv_L or q_L
        flops = 0
        flops += self.to_q.flops(q_L)
        flops += self.to_k.flops(kv_L)
        flops += self.to_v.flops(kv_L)
        return flops


class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        # batch_size*num_windows, window_h*window_w 64, embed_dim
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_, 1, 1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)

        # q and kv are reshaped to separate the heads for multi-head attention
        # to_qkv: batch_size*num_windows, window_h*window_w, embed_dim             -->   batch_size*num_windows, window_h*window_w, embed_dim*heads
        # reshape: batch_size*num_windows, window_h*window_w, embed_dim*heads      -->   batch_size*num_windows, window_h*window_w, 1, heads, embed_dim
        # permute: batch_size*num_windows, window_h*window_w, 1, heads, embed_dim  -->   1, batch_size*num_windows, heads, window_h*window_w, embed_dim
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).contiguous()
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).contiguous()
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v


    def flops(self, q_L, kv_L=None):
        kv_L = kv_L or q_L
        flops = q_L * self.dim * self.inner_dim + kv_L * self.dim * self.inner_dim * 2
        return flops
    ###########################################




########### self-attention #############
class Attention(nn.Module):
    def __init__(self, dim, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # ratio = attn.size(-1)//relative_position_bias.size(-1)
        # relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)

        # attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            # mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

    def flops(self, q_num, kv_num):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        # N = self.win_size[0]*self.win_size[1]
        # nW = H*W/N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(q_num, kv_num)
        # attn = (q @ k.transpose(-2, -1))

        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num
        #  x = (attn @ v)
        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num

        # x = self.proj(x)
        flops += q_num * self.dim * self.dim
        print("MCA:{%.2f}" % (flops / 1e9))
        return flops


########### window-based Low frequency self-attention ###
class Window_Lo_Attention(nn.Module):
    def __init__(self, dim, win_size, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0., alpha=0.5):

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = dim // num_heads  # 每个head的维度 C//num_heads
        self.dim = dim

        # self-attention heads in Lo-Fi
        self.l_heads = int(num_heads * alpha)  # 根据alpha来确定分配给低频注意力的注意力头的数量
        # token dimension in Lo-Fi
        self.l_dim = self.l_heads * head_dim  # 低频注意力的通道数


        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        self.scale = qk_scale or head_dim ** -0.5


        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        # Low frequence attention (Lo-Fi)
        # 如果低频注意力头的个数大于0, 那就说明存在低频注意力机制。 然后,如果窗口尺寸不为1, 那么应当为每一个窗口应用平均池化操作获得低频信息,这样有助于降低低频注意力机制的计算复杂度 （如果窗口尺寸为1,那么池化层就没有意义了）
        if self.l_heads > 0:
            if self.win_size != 1:
                self.sr = nn.AvgPool2d(kernel_size=self.win_size[0],
                                       stride=self.win_size[0])  # 通过平均池化操作获得低频信息 输入图像尺寸H*W, 窗口尺寸为s, 那么输出图像尺寸为H/s*W/s
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)

    def lofi(self, x):
        B, H, W, C = x.shape
        # to_q:    (B,H,W,C)     --> (B,H,W,l_dim)
        # reshape: (B,H,W,l_dim) --> (B,HW,l_heads,head_dim)
        # permute: (B,HW,l_heads,head_dim) --> (B,l_heads,HW,head_dim)     l_dim=l_heads*head_dim;
        q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3).contiguous()

        # 如果窗口尺寸大于1, 在每个窗口执行池化 (如果窗口尺寸等于1,没有池化的必要)
        if self.win_size[0] > 1:
            # (B,H,W,C) --> (B,C,H,W)
            x_ = x.permute(0, 3, 1, 2).contiguous()
            # 在每个窗口执行池化操作
            # HW=patch的总数, 每个池化窗口内有: (ws^2)个patch, 池化完还剩下：HW/(ws^2)个patch;
            # (B,C,H,W) --sr-> (B,C,H/ws,W/ws)
            # reshape: (B,C,H/ws,W/ws) --> (B,C,HW/(ws^2))
            # permute: (B,C,HW/(ws^2)) --> (B, HW/(ws^2), C)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            # 将池化后的输出通过线性层生成kv:(B,HW/(ws^2),C) --l_kv-> (B,HW/(ws^2),l_dim*2) --reshape-> (B,HW/(ws^2),2,l_heads,head_dim) --permute-> (2,B,l_heads,HW/(ws^2),head_dim)
            # to_kv:   (B,HW/(ws^2),C)       --> (B,HW/(ws^2),l_dim*2)
            # reshape: (B,HW/(ws^2),l_dim*2) --> (B,HW/(ws^2),2,l_heads,head_dim)
            # permute: (B,HW/(ws^2),2,l_heads,head_dim) --> (2,B,l_heads,HW/(ws^2),head_dim)
            kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4).contiguous()
        else:
            # 如果窗口尺寸等于1, 那么kv和q一样, 来源于原始输入x: (B,H,W,C) --l_kv-> (B,H,W,l_dim*2) --reshape-> (B,HW,2,l_heads,head_dim) --permute-> (2,B,l_heads,HW,head_dim);  【注意: 如果窗口尺寸为1,那就不会执行池化操作,所以patch的数量也不会减少,依然是HW个patch】
            kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4).contiuous()

        # k:(B,l_heads,HW/(ws^2),head_dim)
        # v:(B,l_heads,HW/(ws^2),head_dim)
        k, v = kv[0], kv[1]

        # 计算q和k之间的注意力矩阵
        # (B,l_heads,HW,head_dim) @ (B,l_heads,head_dim,HW/(ws^2)) = (B,l_heads,HW,HW/(ws^2))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)

        # 通过注意力矩阵对Value矩阵进行加权:
        # (B,l_heads,HW,HW/(ws^2)) @ (B,l_heads,HW/(ws^2),head_dim) = (B,l_heads,HW,head_dim)
        # transpose: (B,HW,l_heads,head_dim) --> (B,l_heads,HW,head_dim)
        # reshape:   (B,l_heads,HW,head_dim) --> (B,H,W,l_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        # (B,H,W,l_dim) --> (B,H,W,l_dim)
        x = self.l_proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x, mask=None):
        # (B, H, W, C) --> (B, H, W, l_dim)
        lofi_out = self.lofi(x)
        return lofi_out




########### window-based Hi self-attention ###
class Window_Hi_Attention(nn.Module):
    def __init__(self, dim, win_size, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0., alpha=0.5):

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = dim // num_heads    # 每个head的维度 C//num_heads
        self.dim = dim

        # self-attention heads in Lo-Fi
        self.l_heads = int(num_heads * alpha)    # 根据alpha来确定分配给低频注意力的注意力头的数量
        # token dimension in Lo-Fi
        self.l_dim = self.l_heads * head_dim     # 低频注意力的通道数

        # self-attention heads in Hi-Fi
        self.h_heads = num_heads - self.l_heads  # 总的注意力头个数-低频注意力头的个数=高频注意力头的个数
        # token dimension in Hi-Fi
        self.h_dim = self.h_heads * head_dim     # 高频注意力的通道数


        self.win_size = win_size                 # Wh, Ww
        self.num_heads = num_heads
        self.scale = qk_scale or head_dim ** -0.5



        # High frequence attention (Hi-Fi)
        # 如果高频注意力头的个数大于0, 那就说明存在高频注意力机制
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)


        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), self.h_heads))    # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])                                     # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])                                     # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))                    # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)                                     # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]     # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()               # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1                              # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)                             # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)




        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)


    def hifi(self, x, mask=None):
        # B*num_windows, window_h*window_w, dim
        B_, N, C = x.shape
        # to_qkv:    B*num_windows, window_h*window_w, dim                    -->   B*num_windows, window_h*window_w, 3*h_dim
        # reshape:  B*num_windows, window_h*window_w, 3*h_dim                 -->   B*num_windows, window_h*window_w, 3, h_head, h_dim/h_head
        # permute:  B*num_windows, window_h*window_w, 3, h_head, h_dim/h_head -->   3, B*num_windows, h_head, window_h*window_w, h_dim/h_head
        qkv = self.h_qkv(x).reshape(B_, N, 3, self.h_heads, self.h_dim // self.h_heads).permute(2, 0, 3, 1, 4).contiguous()
        #  q:(B*num_windows, h_head, window_h*window_w, h_dim/h_head)
        #  k:(B*num_windows, h_head, window_h*window_w, h_dim/h_head)
        #  v:(B*num_windows, h_head, window_h*window_w, h_dim/h_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 每个窗口内计算: 所有patch之间注意力矩阵
        #   (B*num_windows, h_head, window_h*window_w, h_dim/h_head) @ (B*num_windows, h_head, h_dim/h_head, window_h*window_w)
        # = (B*num_windows, h_head, window_h*window_w, window_h*window_w)
        attn = (q @ k.transpose(-2, -1)) * self.scale


        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # attn @ v:        (B*num_windows, h_head, window_h*window_w, window_h*window_w) @ (B*num_windows, h_head, window_h*window_w, h_dim/h_head) = (B*num_windows, h_head, window_h*window_w, h_dim/h_head)
        # transpose(1, 2): (B*num_windows, window_h*window_w, h_head,  h_dim/h_head)
        # reshape:         (B*num_windows, window_h*window_w, h_dim)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.h_dim)
        # 映射层进行输出:(B*num_windows, window_h*window_w, h_dim) --> (B*num_windows, window_h*window_w, h_dim)
        x = self.h_proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x, mask=None):
        # B*num_windows, window_h*window_w, dim -->  B*num_windows, window_h*window_w, h_dim
        hifi_out = self.hifi(x)
        return hifi_out



########### window-based self-attention ###
class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size        # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads     # 每个head的维度 C//num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))    # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])                                     # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])                                     # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))                    # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)                                     # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]     # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()               # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1                              # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)                             # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        # ToDo:如何得到QKV
        if token_projection == 'conv':
            self.qkv = ConvProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)

        elif token_projection == 'linear':
            # cross-attention
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)

            # self-attention
            # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            raise Exception("Projection error!")


        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        # self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

    def forward(self, x, attn_kv=None, mask=None):
        # batch_size*num_windows, window_h*window_w, embed_dim
        B_, N, C = x.shape
        # cross-attention
        q, k, v = self.qkv(x, attn_kv)

        # q and kv are reshaped to separate the heads for multi-head attention
        # to_qkv: batch_size*num_windows, window_h*window_w, embed_dim             -->   batch_size*num_windows, window_h*window_w, embed_dim*heads
        # reshape: batch_size*num_windows, window_h*window_w, embed_dim*heads      -->   batch_size*num_windows, window_h*window_w, 1, heads, embed_dim
        # permute: batch_size*num_windows, window_h*window_w, 1, heads, embed_dim  -->   1, batch_size*num_windows, heads, window_h*window_w, embed_dim
        # self-attention
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # cosine attention
        # attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        # logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        # attn = attn * logit_scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'

    def flops(self, H, W):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        N = self.win_size[0] * self.win_size[1]
        nW = H * W / N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(H * W, H * W)

        # attn = (q @ k.transpose(-2, -1))

        flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += nW * self.num_heads * N * N * (self.dim // self.num_heads)

        # x = self.proj(x)
        flops += nW * N * self.dim * self.dim
        print("W-MSA:{%.2f}" % (flops / 1e9))
        return flops


    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

    def flops(self, q_num, kv_num):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        # N = self.win_size[0]*self.win_size[1]
        # nW = H*W/N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(q_num, kv_num)
        # attn = (q @ k.transpose(-2, -1))

        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num
        #  x = (attn @ v)
        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num

        # x = self.proj(x)
        flops += q_num * self.dim * self.dim
        print("MCA:{%.2f}" % (flops / 1e9))
        return flops



########### feed-forward network #############
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H * W * self.in_features * self.hidden_features
        # fc2
        flops += H * W * self.hidden_features * self.out_features
        print("MLP:{%.2f}" % (flops / 1e9))
        return flops


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

    def flops(self):
        flops = 0
        flops += self.channel * self.channel * self.k_size

        return flops


class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1, sigmoid=True):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size = k, padding = (k-1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        y = self.avg_pool(x)                # 在空间方向执行全局平均池化: (B,C,H,W)-->(B,C,1,1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class eca_layer_1d(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        # b hw c
        # feature descriptor on the global spatial information
        y = self.avg_pool(x.transpose(-1, -2))

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2))

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

    def flops(self):
        flops = 0
        flops += self.channel * self.channel * self.k_size

        return flops


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False):
        super(LeFF, self).__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())

        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = eca_layer_1d(dim) if use_eca else nn.Identity()

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore / reshape
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)

        # the depth-wise convolution between the two linear layers
        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)

        x = self.linear2(x)

        x = self.eca(x)

        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H * W * self.dim * self.hidden_dim
        # dwconv
        flops += H * W * self.hidden_dim * 3 * 3
        # fc2
        flops += H * W * self.hidden_dim * self.dim
        print("LeFF:{%.2f}" % (flops / 1e9))
        # eca 
        if hasattr(self.eca, 'flops'):
            flops += self.eca.flops()
        return flops


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


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


##########################################################################
## ------ Channel Attention --------------
class ca_layer(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(ca_layer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
##---------- Dual Attention Unit (DAU) ----------
class DAU(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=8,
            bias=False, bn=False, act=nn.PReLU(), res_scale=1):
        super(DAU, self).__init__()
        modules_body = [conv(n_feat, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias)]
        self.body = nn.Sequential(*modules_body)

        ## Spatial Attention
        self.SA = spatial_attn_layer()

        ## Channel Attention
        self.CA = ca_layer(n_feat, reduction, bias=bias)

        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res


class LocalityFeedForward(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act='hs+eca', dp_conv=True, act_layer=nn.GELU, dp_first=False,
                 reduction=4):
        """
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
                    hs+se: h_swish and SE module
                    hs+eca: h_swish and ECA module
                    hs+ecah: h_swish and ECA module. Compared with eca, h_sigmoid is used.
        :param reduction: reduction rate in SE module.
        :param wo_dp_conv: without depth-wise convolution.
        :param dp_first: place depth-wise convolution as the first layer.
        通过在FFN中引入深度卷积和门控机制
        使得网络能够更有效地处理图像等二维数据的空间结构信息
        门控深度卷积的使用增加了模型的参数效率和表达能力
        特别是在处理具有复杂空间关系的数据时
        """

        super(LocalityFeedForward, self).__init__()

        # the first linear layer is replaced by 1x1 convolution.
        self.project_in = nn.Sequential(nn.Conv2d(dim, hidden_dim, 1, 1, 0, bias=False),
                                        # nn.LayerNorm(hidden_dim,1,1),
                                        act_layer())

        # the depth-wise convolution between the two linear layers
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                                    # LN则是对每个样本在所有特征上进行归一化
                                    # 而不是跨批次  这意味着它不依赖于批次的大小
                                    # 因此在批次大小较小或动态变化的情况下仍然能够保持性
                                    # nn.LayerNorm(hidden_dim,1,1),
                                    act_layer())

        if act.find('+') >= 0:
            attn = act.split('+')[1]
            if attn == 'se':
                self.attn = SELayer(hidden_dim, reduction=reduction)
            elif attn.find('eca') >= 0:
                self.attn = ECALayer(hidden_dim, sigmoid=attn == 'eca')
            else:
                raise NotImplementedError('Activation type {} is not implemented'.format(act))

        # the second linear layer is replaced by 1x1 convolution.
        self.project_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=False),
                                         # nn.LayerNorm(dim, 1, 1),
                                         act_layer())

    def forward(self, x):
        # x = x + self.conv(x)
        x1 = self.project_in(x)
        x2 = self.dwconv(x1)
        x3 = self.attn(x2)
        y = self.project_out(x3)
        return x + y

# Gated Locality FeedForward Network with SE Layer
class Gated_LocalityFeedForward(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act='hs+eca', dp_conv=True, act_layer=nn.GELU, dp_first=False, reduction=4):
        """
        引入深度卷积和门控机制
        使得网络能够更有效地处理图像等二维数据的空间结构信息
        门控深度卷积的使用增加了模型的参数效率和表达能力
        特别是在处理具有复杂空间关系的数据时
        """
        super(Gated_LocalityFeedForward, self).__init__()
        self.project_in = nn.Sequential(nn.Conv2d(dim, hidden_dim, 1, 1, 0, bias=False))
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False))


        self.dwconv = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim, bias=False),
            # Pointwise Convolution
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            ECALayer(hidden_dim, sigmoid= "True")
        )

        self.act_layer = act_layer()
        self.project_out = nn.Sequential(nn.Conv2d(hidden_dim//2, dim, 1, 1, 0, bias=False),
                                         act_layer()
                                         )

    def forward(self, x):
        x1 = self.project_in(x)
        x2, x3 = self.dwconv(x1).chunk(2, dim=1)
        x4 = self.act_layer(x2)*x3
        y = self.project_out(x4)
        return x + y






#########################################
########### window operation#############
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                     stride=win_size)  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.permute(0, 2, 3, 1).contiguous()  # B' ,Wh ,Ww ,C
    else:
        # B, C, H, W --> B, H/Wh, Wh, W/Ww, Ww, C
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)

        # B, H/Wh, Wh, W/Ww, Ww, C --> B, H/Wh, W/Ww, Wh, Ww, C --> B*num_windows, Wh, Ww, C
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)
    return windows


def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()  # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                   stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


"""
可以将Downsample Block和PatchMerging Block结合起来，在模型中充分利用两者的优势。这样的组合可以在降低分辨率的同时保留更多的信息，并提高模型提取特征的能力。以下是一种可能的结合方式：

序列结合：在模型的不同层次使用不同的下采样方法。例如，在模型的初级阶段使用Downsample Block进行特征提取，因为在这个阶段，保留边缘等细节信息是很重要的。然后，在模型的更深层次使用PatchMerging Block进行更高级别的特征合成和信息压缩。

并行结合：在同一层级内同时使用两种下采样方法，分别处理输入数据，然后将两者的输出进行合并或融合。这种方式可以让模型同时学习到由不同下采样方法提取的特征，可能会带来更丰富的表征能力。

自适应结合：设计一个机制来决定在模型的每个阶段使用哪种下采样方法，可能基于当前层的输入特征、训练进度或其他指标。这种方法更加灵活，可以根据模型的实时表现自动调整下采样策略。

在实现这些结合策略时，需要仔细考虑模型的复杂度和计算成本，确保增加的复杂度是合理的，并且能够带来性能的提升。实验和验证是关键，通过对比实验可以确定哪种结合方式对特定任务最有效。

"""


##############################################################################
#######################  DownSample ##########################################
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()      # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H / 2 * W / 2 * self.in_channel * self.out_channel * 4 * 4
        print("Downsample:{%.2f}" % (flops / 1e9))
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class ParallelDownsampleMerge(nn.Module):
    def __init__(self, in_channel, out_channel, input_resolution, norm_layer=nn.LayerNorm):
        super(ParallelDownsampleMerge, self).__init__()
        self.downsample = Downsample(in_channel, out_channel)
        self.patch_merging = PatchMerging(input_resolution, in_channel, norm_layer)

        # Adjusting out channels after concatenation
        self.adjust_channels = nn.Conv2d(out_channel + 2 * in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        # Downsample path
        downsample_output = self.downsample(x)  # (B, out_channel, H/2, W/2)

        # Patch merging path
        B, C, H, W = x.shape
        patch_merging_output = self.patch_merging(x.permute(0, 2, 3, 1).contiguous())  # (B, H*W, C) to (B, H, W, C)
        patch_merging_output = patch_merging_output.permute(0, 2, 1).view(B, -1, H // 2,
                                                                          W // 2).contiguous()  # Reshape back to (B, C', H/2, W/2)

        # Concatenate along channel dimension
        combined_output = torch.cat([downsample_output, patch_merging_output], dim=1)  # (B, out_channel + C', H/2, W/2)

        # Adjust channels
        combined_output = self.adjust_channels(combined_output)  # (B, out_channel, H/2, W/2)
        return combined_output


############################################################################
#######################  UpSample ##########################################
# Upsample Block
class Transpose_Upsample(nn.Module):
    def __init__(self, in_channel, out_channel, last_stage=False):
        super(Transpose_Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.last_stage = last_stage

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        if self.last_stage == True:
            # (B,C,H,W) --> (B,C/2,2H,2W) --> (B,2H,2W,C/2)
            out = self.deconv(x).permute(0, 2, 3, 1).contiguous()
        else:
            # (B,C,H,W) --> (B,C/2,2H,2W) --> (B,C/2,2H*2W) --> (B,2H*2W,C/2)
            out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * 2 * W * 2 * self.in_channel * self.out_channel * 2 * 2
        print("Upsample:{%.2f}" % (flops / 1e9))
        return flops


# Dual up-sample
class UpSample_Dual(nn.Module):
    def __init__(self, input_resolution, in_channels, scale_factor, last_stage=False):
        super(UpSample_Dual, self).__init__()
        self.input_resolution = input_resolution
        self.factor = scale_factor
        self.last_stage = last_stage

        if self.factor == 2:
            self.conv = nn.Conv2d(in_channels, in_channels // 2, 1, 1, 0, bias=False)
            self.up_p = nn.Sequential(nn.Conv2d(in_channels, 2 * in_channels, 1, 1, 0, bias=False),
                                      nn.PReLU(),
                                      nn.PixelShuffle(scale_factor),
                                      nn.Conv2d(in_channels // 2, in_channels // 2, 1, stride=1, padding=0, bias=False))

            self.up_b = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                                      nn.PReLU(),
                                      nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                                      nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=False))
        elif self.factor == 4:
            self.conv = nn.Conv2d(2 * in_channels, in_channels, 1, 1, 0, bias=False)
            self.up_p = nn.Sequential(nn.Conv2d(in_channels, 16 * in_channels, 1, 1, 0, bias=False),
                                      nn.PReLU(),
                                      nn.PixelShuffle(scale_factor),
                                      nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

            self.up_b = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                                      nn.PReLU(),
                                      nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                                      nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        """
        x: B, L = H*W, C
        """
        # if type(self.input_resolution) == int:
        #     H = self.input_resolution
        #     W = self.input_resolution
        #
        # elif type(self.input_resolution) == tuple:
        #     H, W = self.input_resolution

        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.view(B, H, W, C)      # B, H, W, C
        x = x.permute(0, 3, 1, 2).contiguous()   # B, C, H, W

        # factor = 2
        # 1x1 Conv: (B, C, H, W) --> (B, 2C, H, W)
        # Pixel Shuffle: (B, 2C, H, W) --> (B, C/2, 2H, 2W)
        # 1x1 Conv: (B, C/2, 2H, 2W) --> (B, C/2, 2H, 2W)

        # factor = 4
        # 1x1 Conv: (B, C, H, W) --> (B, 16C, H, W)
        # Pixel Shuffle: (B, 16C, H, W) --> (B, C, 4H, 4W)
        # 1x1 Conv: (B, C, 4H, 4W) --> (B, C, 4H, 4W)
        x_p = self.up_p(x)

        # factor = 2
        # 1x1 Conv: (B, C, H, W) --> (B, C, H, W)
        # Bilinear: (B, C, H, W) --> (B, C, 2H, 2W)
        # 1x1 Conv: (B, C, 2H, 2W) --> (B, C/2, 2H, 2W)

        # factor = 4
        # 1x1 Conv: (B, C, H, W) --> (B, C, H, W)
        # Bilinear: (B, C, H, W) --> (B, C, 4H, 4W)
        # 1x1 Conv: (B, C, 4H, 4W) --> (B, C, 4H, 4W)
        x_b = self.up_b(x)

        # factor = 2
        # Concat: (B, C/2, 2H, 2W) + (B, C/2, 2H, 2W) --> (B, C, 2H, 2W)
        # conv:   (B, C, 2H, 2W) -->  (B, C/2, 2H, 2W)

        # factor = 4
        # Concat: (B, C, 4H, 4W) + (B, C, 4H, 4W) --> (B, 2C, 4H, 4W)
        # conv:   (B, 2C, 4H, 4W) -->  (B, C, 4H, 4W)
        out = self.conv(torch.cat([x_p, x_b], dim=1))

        # (B, C/2, 2H, 2W) --> (B, 2H, 2W, C/2)
        out = out.permute(0, 2, 3, 1).contiguous()
        if self.last_stage == False:
            # (B, 2H, 2W, C/2) --> (B, 2H*2W, C/2)
            out = out.view(B, -1, C // 2)
        return out


class upsample_last(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(upsample_last, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=4),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * 2 * W * 2 * self.in_channel * self.out_channel * 2 * 2
        print("Upsample:{%.2f}" % (flops / 1e9))
        return flops


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


#####################################################################################
###                           Input Projection                                    ###
#####################################################################################
# Plan1: U-former Input Projection
class InputProj(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=256, in_channel=3, embed_dim=96, patch_size=4, norm_layer=nn.LayerNorm,
                 act_layer=nn.LeakyReLU):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size),
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = embed_dim

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()                   # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * W * self.in_channel * self.out_channel * 3 * 3

        if self.norm is not None:
            flops += H * W * self.out_channel
        print("Input_proj:{%.2f}" % (flops / 1e9))
        return flops


# Plan 2: Swin-UNet Input Projection: part partition + linear embedding
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


#####################################################################################
###                           Output Projection                                   ###
#####################################################################################
# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * W * self.in_channel * self.out_channel * 3 * 3

        if self.norm is not None:
            flops += H * W * self.out_channel
        print("Output_proj:{%.2f}" % (flops / 1e9))
        return flops


#########################################
########### LeWinTransformer ############
class LeWinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear', token_mlp='g_ffn',
                 attention_type = "w-msa", modulator=False, cross_modulator=False, alpha=0.5):
        super().__init__()
        self.dim = dim
        self.h_dim = int(dim * alpha)
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        self.attention_type = attention_type
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        if modulator:
            self.modulator = nn.Embedding(win_size * win_size, dim)  # modulator
        else:
            self.modulator = None

        if cross_modulator:
            self.cross_modulator = nn.Embedding(win_size * win_size, dim)  # cross_modulator
            self.cross_attn = Attention(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                        proj_drop=drop,
                                        token_projection=token_projection, )
            self.norm_cross = norm_layer(dim)
        else:
            self.cross_modulator = None

        self.norm1 = norm_layer(dim)

        if self.attention_type == 'w-msa':
            self.attn = WindowAttention(
                dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                token_projection=token_projection)

        elif self.attention_type == 'HiLo_attention':
            self.Hi_attn = Window_Hi_Attention(
                dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                token_projection=token_projection, alpha=0.5)

            self.Lo_attn = Window_Lo_Attention(
                dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                token_projection=token_projection, alpha=0.5)


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if token_mlp in ['ffn', 'mlp']:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif token_mlp == 'leff':
            self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif token_mlp == 'lff':
            self.conv = LocalityFeedForward(dim, mlp_hidden_dim, act='hs+eca', dp_conv=True)
        elif token_mlp == 'g_ffn':
            self.conv = Gated_LocalityFeedForward(dim, mlp_hidden_dim, act='hs+eca', dp_conv=True)
        else:
            raise Exception("FFN error!")

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio},modulator={self.modulator}"

    def forward(self, x, mask=None):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        # H, W = self.input_resolution
        # assert H * W == L, "input feature has wrong size"

        ## input mask
        if mask != None:
            input_mask = F.interpolate(mask, size=(H, W)).permute(0, 2, 3, 1).contiguous()
            input_mask_windows = window_partition(input_mask, self.win_size)          # nW, win_size, win_size, 1
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size)    # nW, win_size*win_size
            attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(1)               # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(
                2)  # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(
                shift_attn_mask == 0, float(0.0))
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask

        if self.cross_modulator is not None:
            shortcut = x
            x_cross = self.norm_cross(x)
            x_cross = self.cross_attn(x, self.cross_modulator.weight)
            x = shortcut + x_cross

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)


        if self.attention_type == 'HiLo_attention':
            # 窗口内执行低频注意力
            Lo_x = self.Lo_attn(x, mask=attn_mask)                               # B, H', W', dim --> B, H, W, l_dim


        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x


        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)               # num_windows*B, win_size, win_size, C      N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)     # nW*B, win_size*win_size, C

        # with_modulator
        if self.modulator is not None:
            wmsa_in = self.with_pos_embed(x_windows, self.modulator.weight)
        else:
            wmsa_in = x_windows


        # W-MSA/SW-MSA
        if self.attention_type == 'HiLo_attention':
            # 窗口内执行高频注意力
            Hi_attn_windows = self.Hi_attn(wmsa_in, mask=attn_mask)                  # nW*B, win_size*win_size, C --> nW*B, win_size*win_size, h_dim

            # merge windows
            Hi_attn_windows = Hi_attn_windows.view(-1, self.win_size, self.win_size, self.h_dim)        # nW*B, Wh*Ww, h_dim  --> nW*B, Wh, Ww, h_dim
            shifted_Hi_x = window_reverse(Hi_attn_windows, self.win_size, H, W)                         # nW*B, Wh, Ww, h_dim --> B, H', W', h_dim

            # reverse cyclic shift
            if self.shift_size > 0:
                Hi_x = torch.roll(shifted_Hi_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                Hi_x = shifted_Hi_x


            # 在通道方向上拼接高频注意力和低频注意力的输出: (B, H', W', h_dim+l_dim) = (B, H', W', dim)
            HiLo_x = torch.cat((Lo_x, Hi_x), dim=-1)
            HiLo_x = HiLo_x.view(B, H * W, C)

            x = shortcut + self.drop_path(HiLo_x)


        elif self.attention_type == 'window_attention':
            # 窗口内执行高频注意力
            attn_windows = self.attn(wmsa_in,mask=attn_mask)                       # nW*B, win_size*win_size, dim --> nW*B, win_size*win_size, dim

            # merge windows
            attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)  # nW*B, Wh*Ww, h_dim  --> nW*B, Wh, Ww, dim
            shifted_x = window_reverse(attn_windows, self.win_size, H, W)          # nW*B, Wh, Ww, dim --> B, H', W', dim

            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x
            x = x.view(B, H * W, C)
            x = shortcut + self.drop_path(x)




        # FFN
        if self.token_mlp == 'lff':
            x = self.norm2(x)
            x = self.conv(x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous())
            x = x.permute(0, 2, 3, 1).contoguous().view(B, H * W, C)
        elif self.token_mlp == "g_ffn":
            x = self.norm2(x)
            x = self.conv(x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous())
            x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        del attn_mask

        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution

        if self.cross_modulator is not None:
            flops += self.dim * H * W
            flops += self.cross_attn.flops(H * W, self.win_size * self.win_size)

        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += self.mlp.flops(H, W)
        # print("LeWin:{%.2f}"%(flops/1e9))
        return flops





class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        # (h, w, dim, 2)，最后一维2，表示实部和虚部
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C).contiguous()

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        # 将张量(h, w, dim, 2)转换为形状为(h, w, dim)的复数张量
        weight = torch.view_as_complex(self.complex_weight.clone().contiguous())


        # 将复数张量x与复数张量weight逐元素相乘
        # 根据weight定义的滤波器修改x的频率分量   在频率域对输入张量应用一个全局滤波器
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C).contiguous()

        return x


class SpectBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x



#########################################
########### Basic layer of Uformer ######
class BasicUformerLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, depth_spectblock, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear', token_mlp='ffn',attention_type="window_attention", shift_flag=True,
                 modulator=False, cross_modulator=False,spect_block=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.depth_spectblock = depth_spectblock
        self.use_checkpoint = use_checkpoint
        self.spect_block = spect_block
        # build blocks
        if shift_flag:
            self.blocks = nn.ModuleList([
                LeWinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                      num_heads=num_heads, win_size=win_size,
                                      shift_size=0 if (i % 2 == 0) else win_size // 2,
                                      mlp_ratio=mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop, attn_drop=attn_drop,
                                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                      norm_layer=norm_layer, token_projection=token_projection, token_mlp=token_mlp,attention_type=attention_type,
                                      modulator=modulator, cross_modulator=cross_modulator)
                for i in range(depth)])

            if self.spect_block:
                for _ in range(depth_spectblock):
                    self.blocks.insert(0, SpectBlock(dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=input_resolution[0], w=input_resolution[1]//2+1))

        else:
            self.blocks = nn.ModuleList([
                LeWinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                      num_heads=num_heads, win_size=win_size,
                                      shift_size=0,
                                      mlp_ratio=mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop, attn_drop=attn_drop,
                                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                      norm_layer=norm_layer, token_projection=token_projection, token_mlp=token_mlp,attention_type=attention_type,
                                      modulator=modulator, cross_modulator=cross_modulator)
                for i in range(depth)])

            if self.spect_block:
                for _ in range(depth_spectblock):
                    self.blocks.insert(0, SpectBlock(dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=input_resolution[0], w=input_resolution[1]//2+1))

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops



class Uformer(nn.Module):
    def __init__(self, img_size=256, in_chans = 3, dd_in=3, patch_size = 4,
                 embed_dim=96, depths = [2, 2, 4, 8], depth_spectblock=[2,2,2,2], num_heads = [2, 4, 6, 8],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='g_ffn', attention_type='w-msa',
                 shift_flag=True, modulator=False,cross_modulator=False, final_upsample = "dual_upsample",
                 upsample_style="dual_upsample", downsample_style="patch_merging",spect_block=False, **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths)
        self.num_dec_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.dd_in = dd_in
        self.final_upsample = final_upsample
        self.attention_type = attention_type
        self.downsample_style = downsample_style
        self.upsample_style = upsample_style
        self.spect_block = spect_block

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]

        # build layers

        # Input/Output    256,256,3 -> 64,64,96
        self.input_proj = InputProj(in_channel=dd_in, embed_dim=embed_dim, patch_size=patch_size,
                                    act_layer=nn.LeakyReLU)
        patches_resolution = self.input_proj.patches_resolution
        self.patches_resolution = patches_resolution

        self.output_proj = OutputProj(in_channel=2 * embed_dim, out_channel=in_chans, kernel_size=3, stride=1)

        # Encoder        64,64,96 -> 64,64,96
        self.encoderlayer_0 = BasicUformerLayer(dim=embed_dim,
                                                output_dim=embed_dim,
                                                input_resolution=(img_size // 4,
                                                                  img_size // 4),
                                                depth=depths[0],
                                                depth_spectblock=depth_spectblock[0],
                                                num_heads=num_heads[0],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,attention_type=attention_type,
                                                shift_flag=shift_flag,
                                                spect_block=spect_block)
        # Downsample    64,64,96 -> 32,32,192
        if downsample_style == "patch_merging":
            self.dowsample_0 = PatchMerging(input_resolution=(img_size // 4, img_size // 4), dim=embed_dim)
        elif downsample_style == "conv_downsample":
            self.dowsample_0 = Downsample(embed_dim, embed_dim * 2)
        self.encoderlayer_1 = BasicUformerLayer(dim=embed_dim * 2,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size // 8,
                                                                  img_size // 8),
                                                depth=depths[1],
                                                depth_spectblock=depth_spectblock[1],
                                                num_heads=num_heads[1],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,attention_type=attention_type,
                                                shift_flag=shift_flag,
                                                spect_block=spect_block)
        if downsample_style == "patch_merging":
            self.dowsample_1 = PatchMerging(input_resolution=(img_size // 8, img_size // 8),dim=embed_dim * 2)
        elif downsample_style == "conv_downsample":
            self.dowsample_1 = Downsample(embed_dim * 2, embed_dim * 4)
        self.encoderlayer_2 = BasicUformerLayer(dim=embed_dim * 4,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(img_size // 16,
                                                                  img_size // 16),
                                                depth=depths[2],
                                                depth_spectblock=depth_spectblock[2],
                                                num_heads=num_heads[2],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,attention_type=attention_type,
                                                shift_flag=shift_flag,
                                                spect_block=spect_block)
        if downsample_style == "patch_merging":
            self.dowsample_2 = PatchMerging(input_resolution=(img_size // 16,img_size // 16),dim=embed_dim * 4)
        elif downsample_style == "conv_downsample":
            self.dowsample_2 = Downsample(embed_dim * 4, embed_dim * 8)

        # Bottleneck
        self.conv = BasicUformerLayer(dim=embed_dim * 8,
                                      output_dim=embed_dim * 8,
                                      input_resolution=(img_size // 32,
                                                        img_size // 32),
                                      depth=depths[3],
                                      depth_spectblock=depth_spectblock[3],
                                      num_heads=num_heads[3],
                                      win_size=win_size,
                                      mlp_ratio=self.mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop_rate, attn_drop=attn_drop_rate,
                                      drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
                                      norm_layer=norm_layer,
                                      use_checkpoint=use_checkpoint,
                                      token_projection=token_projection, token_mlp=token_mlp,attention_type=attention_type,
                                      shift_flag=shift_flag,
                                      spect_block=spect_block)

        # Decoder
        if self.upsample_style == "dual_upsample":
            self.up_0 = UpSample_Dual(input_resolution=(img_size // 32, img_size // 32),
                                  in_channels=embed_dim * 8, scale_factor=2)
        elif self.upsample_style == "transpose_conv_upsample":
            self.up_0 = Transpose_Upsample(embed_dim * 8, embed_dim * 4)
        self.concat_back_dim_0 = nn.Linear(embed_dim * 8, embed_dim * 4)
        self.decoderlayer_0 = BasicUformerLayer(dim=embed_dim * 4,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(img_size // 16,
                                                                  img_size // 16),
                                                depth=depths[2],
                                                depth_spectblock=depth_spectblock[2],
                                                num_heads=num_heads[2],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,attention_type=attention_type,
                                                shift_flag=shift_flag,
                                                spect_block=spect_block,
                                                )
        if self.upsample_style == "dual_upsample":
            self.up_1 = UpSample_Dual(input_resolution=(img_size // 16, img_size // 16),
                                  in_channels=embed_dim * 4, scale_factor=2)
        elif self.upsample_style == "transpose_conv_upsample":
            self.up_1 = Transpose_Upsample(embed_dim * 4, embed_dim * 2)
        self.concat_back_dim_1 = nn.Linear(embed_dim * 4, embed_dim * 2)
        self.decoderlayer_1 = BasicUformerLayer(dim=embed_dim * 2,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size // 8,
                                                                  img_size // 8),
                                                depth=depths[1],
                                                depth_spectblock=depth_spectblock[1],
                                                num_heads=num_heads[1],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,attention_type=attention_type,
                                                shift_flag=shift_flag,
                                                spect_block=spect_block,
                                                )
        if self.upsample_style == "dual_upsample":
            self.up_2 = UpSample_Dual(input_resolution=(img_size // 8, img_size // 8),
                                  in_channels=embed_dim * 2, scale_factor=2)
        elif self.upsample_style == "transpose_conv_upsample":
            self.up_2 = Transpose_Upsample(embed_dim * 2, embed_dim)
        self.concat_back_dim_2 = nn.Linear(embed_dim * 2, embed_dim)
        self.decoderlayer_2 = BasicUformerLayer(dim=embed_dim,
                                                output_dim=embed_dim,
                                                input_resolution=(img_size // 4,
                                                                  img_size // 4),
                                                depth=depths[0],
                                                depth_spectblock=depth_spectblock[0],
                                                num_heads=num_heads[0],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,attention_type=attention_type,
                                                shift_flag=shift_flag,
                                                spect_block=spect_block,
                                                )

        if self.final_upsample == "dual_upsample":
            self.up_3 = UpSample_Dual(input_resolution=(img_size // 4, img_size // 4),
                                    in_channels=embed_dim, scale_factor=2)
            self.up_4 = UpSample_Dual(input_resolution=(img_size // 2, img_size // 2),
                                      in_channels=embed_dim//2, scale_factor=2, last_stage=True)
            self.output = nn.Conv2d(in_channels=embed_dim//4, out_channels=3, kernel_size=3, stride=1,
                                    padding=1, bias=False)
        elif self.final_upsample == "transpose_conv_upsample":
            self.up_3 = Transpose_Upsample(embed_dim, embed_dim//2)
            self.up_4 = Transpose_Upsample(embed_dim//2, embed_dim//4, last_stage=True)
            self.output = nn.Conv2d(in_channels=embed_dim//4, out_channels=3, kernel_size=1, stride=1,
                                    padding=0, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"


    def forward(self, x):
        # Input Projection
        y = self.input_proj(x)                                      # 1, 3, 256, 256 -> 1,4096, 96
        y = self.pos_drop(y)
        # Encoder
        conv0 = self.encoderlayer_0(y)                              # 1, 4096, 96
        pool0 = self.dowsample_0(conv0)                             # 1, 4096, 96 -> 1, 1024, 192
        conv1 = self.encoderlayer_1(pool0)                          # 1, 1024, 192
        pool1 = self.dowsample_1(conv1)                             # 1, 1024, 192 -> 1, 256, 384
        conv2 = self.encoderlayer_2(pool1)                          # 1, 256, 384
        pool2 = self.dowsample_2(conv2)                             # 1, 256, 384 -> 1, 64, 768

        # Bottleneck
        conv3 = self.conv(pool2)                                    # 64, 768

        # Decoder
        up0 = self.up_0(conv3)                                      # 64, 768 -> 256, 384
        deconv0 = torch.cat([up0, conv2], -1)                # 256, 768
        deconv0_back_dim = self.concat_back_dim_0(deconv0)          # 256, 768 -> 256, 384
        deconv0 = self.decoderlayer_0(deconv0_back_dim)             # 256, 384

        up1 = self.up_1(deconv0)                                           # 256, 384 -> 1024, 192
        deconv1 = torch.cat([up1, conv1], -1)                       # 1024, 384
        deconv1_back_dim = self.concat_back_dim_1(deconv1)                 # 1024, 384 -> 1024, 192
        deconv1 = self.decoderlayer_1(deconv1_back_dim)                    # 1024, 192

        up2 = self.up_2(deconv1)                                    # 1024, 192 -> 4096, 96
        deconv2 = torch.cat([up2, conv0], -1)                # 4096, 192
        deconv2_back_dim = self.concat_back_dim_2(deconv2)          # 4096, 192 -> 4096, 96
        deconv2 = self.decoderlayer_2(deconv2_back_dim)             # 4096, 96

        # Output Projection
        y = self.up_3(deconv2)                                       # 1, 64*64, 96    -> 1, 128*128, 48



        y1 = self.up_4(y)                                            # 1, 128*128, 48  -> 1, 256,256, 24
        y2 = y1.permute(0, 3, 1, 2).contiguous()                     # 1, 256, 256, 24 -> 1, 24, 256, 256
        out = self.output(y2)                                        # 1, 24, 256, 256 -> 1, 3, 256, 256


        return x + out if self.dd_in == 3 else out

    def flops(self):
        flops = 0
        # Input Projection
        flops += self.input_proj.flops(self.reso, self.reso)
        # Encoder
        flops += self.encoderlayer_0.flops() + self.dowsample_0.flops(self.reso, self.reso)
        flops += self.encoderlayer_1.flops() + self.dowsample_1.flops(self.reso // 2, self.reso // 2)
        flops += self.encoderlayer_2.flops() + self.dowsample_2.flops(self.reso // 2 ** 2, self.reso // 2 ** 2)
        flops += self.encoderlayer_3.flops() + self.dowsample_3.flops(self.reso // 2 ** 3, self.reso // 2 ** 3)

        # Bottleneck
        flops += self.conv.flops()

        # Decoder
        flops += self.upsample_0.flops(self.reso // 2 ** 4, self.reso // 2 ** 4) + self.decoderlayer_0.flops()
        flops += self.upsample_1.flops(self.reso // 2 ** 3, self.reso // 2 ** 3) + self.decoderlayer_1.flops()
        flops += self.upsample_2.flops(self.reso // 2 ** 2, self.reso // 2 ** 2) + self.decoderlayer_2.flops()
        flops += self.upsample_3.flops(self.reso // 2, self.reso // 2) + self.decoderlayer_3.flops()

        # Output Projection
        flops += self.output_proj.flops(self.reso, self.reso)
        return flops


if __name__ == "__main__":
    input_size = 256
    arch = Uformer
    depths = [2, 2, 2, 2]
    depths_spectblock = [2, 2, 2, 2]
    model_restoration = Uformer(img_size=input_size, embed_dim=96, patch_size=4, depths=depths, depths_spectblock=depths, depth_spectblock=depths_spectblock,
                                win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='g_ffn', attention_type="HiLo_attention", modulator=False,cross_modulator=False,
                                shift_flag=True, upsample_style="dual_upsample", downsample_style="conv_downsample", final_upsample="dual_upsample", spect_block=True)
    print(model_restoration)
    input_tensor = torch.randn(1, 3, input_size, input_size)
    output_tensor = model_restoration(input_tensor)
    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(model_restoration, (3, input_size, input_size), as_strings=True,
    #                                             print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # print('# model_restoration parameters: %.2f M' % (sum(param.numel() for param in model_restoration.parameters()) / 1e6))
    # print("number of GFLOPs: %.2f G" % (model_restoration.flops() / 1e9))

    # 将模型及输入张量添加到 TensorBoard
    # with SummaryWriter("./network_visualization") as w:
    #     w.add_graph(model_restoration, input_tensor)

    # writer.add_graph(conv_block, input_tensor)

    # # 关闭 writer

    # writer.close()

    # 在命令行中运行以下命令来启动 TensorBoard：
    # tensorboard --logdir=network_visualization   --load_fast=false

