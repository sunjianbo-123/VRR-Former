import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from torchvision.models import vgg19
import warnings
from typing import List, Optional, Tuple, Union
from torch import Tensor


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = torch.add(x , - y)
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        error = torch.sqrt(diff * diff + self.eps * self.eps)
        loss = torch.mean(error)                                            # 1/N * sum(sqrt(diff^2 + eps^2))
        return loss



"""
为什么使用高斯滤波器？

高斯滤波器是一种在图像处理中常用的平滑滤波器，它通过对图像进行加权平均来减少细节和噪声。
其权重由高斯分布决定，这意味着中心像素的权重最大，越远离中心的像素权重越小。
使用高斯滤波器的目的是保留图像的重要结构特征，同时去除细节层面的干扰，如噪声。
在SSIM的上下文中，这有助于更准确地评估两幅图像在结构上的相似度。

为什么使用高斯而非其他滤波器
高斯滤波器对图像的局部区域进行加权平均，从而计算出局部区域的均值、方差和协方差。这种加权平均方式更符合人眼对图像结构的感知方式，能够有效捕捉图像的结构信息。
权重分布：高斯分布提供了一种自然的、中心倾斜的权重分布，这与人眼的视觉敏感度相符。人眼对于视野中心的细节更为敏感，而对于边缘部分则不那么敏感。高斯滤波器模拟了这一点，使得中心像素对最终的SSIM计算影响更大。
平滑性质：高斯滤波器是一种有效的平滑技术，能够在保留重要图像特征的同时，减少噪声和不重要的细节。这对于结构相似度的评估是至关重要的，因为我们希望评估的是图像的整体结构和质量，而非噪声或微小细节的差异。

在SSIM中的作用
局部特征评估：SSIM是在图像的局部窗口上计算的，而非整幅图像。高斯窗口用于定义这些局部区域的大小，并且通过平滑操作，保证计算窗口内的像素贡献不仅仅基于它们的空间位置，而是以一种在视觉上更有意义的方式反映出来。
计算均值、方差和协方差：在SSIM的公式中，需要计算图像的局部均值、方差和协方差。通过应用高斯窗口作为权重，我们可以获得更加平滑和代表性的局部统计量，这有助于更准确地评估图像质量。

生成一个高斯窗口，该窗口作为SSIM计算中的高斯平滑滤波器
它通过高斯公式计算权重，确保窗口中心的权重最高，越靠近边缘权重越低
size参数指定了窗口的尺寸，sigma参数控制了高斯分布的标准差，影响权重的分布范围


结合两种损失函数的优势：
CharbonnierLoss提供一种基于像素差异的损失，有利于细节恢复；
而SSIMLoss则关注图像的结构相似性，有利于保持图像整体的视觉质量。
这种组合对于图像去反射等需要同时关注细节和整体质量的任务尤为重要。
"""

# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
#     return gauss / gauss.sum()
#
#
# def create_window(window_size, channel):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
#     return window
#
#
# def _ssim(img1, img2, window, window_size, channel, size_average=True):
#     mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
#
#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2
#
#     sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
#
#
#
#     ssim_map = ((2 * mu1_mu2 + 0.0001) * (2 * sigma12 + 2.7e-08)) / ((mu1_sq + mu2_sq + 0.0001) * (sigma1_sq + sigma2_sq + 2.7e-08))
#     # ssim_map = ssim_map.clamp(-1, 1)  # Ensure SSIM does not go beyond the -1 to 1 range
#
#     if size_average:
#         return ssim_map.mean()
#     else:
#         return ssim_map.mean(1).mean(1).mean(1)
#
#
# class SSIMLoss(torch.nn.Module):
#     def __init__(self, window_size=11, size_average=True):
#         super(SSIMLoss, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.channel = 1
#         self.window = create_window(window_size, self.channel)
#
#     def forward(self, img1, img2):
#         (_, channel, _, _) = img1.size()
#
#         if channel == self.channel and self.window.data.type() == img1.data.type():
#             window = self.window
#         else:
#             window = create_window(self.window_size, channel)
#
#             if img1.is_cuda:
#                 window = window.cuda(img1.get_device())
#             window = window.type_as(img1)
#
#             self.window = window
#             self.channel = channel
#
#         return   _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def _fspecial_gauss_1d(size: int, sigma: float) -> Tensor:
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input: Tensor, win: Tensor) -> Tensor:
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float,
    win: Tensor,
    size_average: bool = True,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03)
) -> Tuple[Tensor, Tensor]:
    r""" Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        data_range (float or int): value range of input images. (usually 1.0 or 255)
        win (torch.Tensor): 1-D gauss kernel
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    nonnegative_ssim: bool = False,
) -> Tensor:
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    #if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ms_ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    weights: Optional[List[float]] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03)
) -> Tensor:
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    #if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
        2 ** 4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights_tensor = X.new_tensor(weights)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights_tensor.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # type: ignore  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights_tensor.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        spatial_dims: int = 2,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
        nonnegative_ssim: bool = False,
    ) -> None:
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        )


class MS_SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        spatial_dims: int = 2,
        weights: Optional[List[float]] = None,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    ) -> None:
        r""" class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return ms_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
        )



"""
允许研究人员根据特定任务的需求调整不同阶段特征的重要性。
例如，如果任务对细节的恢复更为敏感，可以通过增加靠近输入层（较低阶段）的权重来强调低级特征；
相反，如果任务更侧重于保持图像的整体布局和语义，可以为更深层的特征（高级阶段）分配更高的权重。
"""
class PerceptualLossVGG19(nn.Module):
    def __init__(self, requires_grad=False, resize=True, layer_weights=None):
        super(PerceptualLossVGG19, self).__init__()
        vgg_pretrained_features = vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.loss = CharbonnierLoss()

        # 如果没有提供特定层的权重，则默认每层权重相等
        if layer_weights is None:
            self.layer_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        else:
            self.layer_weights = layer_weights


        for x in range(4):       # Conv1_1 ~ Conv1_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):    # Conv2_1 ~ Conv2_2
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 14):   # Conv3_1 ~ Conv3_4
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 23):  # Conv4_1 ~ Conv4_4
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 32):  # Conv5_1 ~ Conv5_4
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input, target):
        # Resize input and target to 224x224 to extract features
        # VGG模型默认使用的输入范围是[0, 1]，且在ImageNet数据集上进行预训练，需要对输入数据进行相应的归一化处理
        input = F.interpolate(input, mode='bilinear', size=(224, 224), align_corners=False)
        target = F.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)

        input_slices = [input]
        target_slices = [target]
        for i in range(5):  # Iterate through all five slices
            input_slices.append(getattr(self, f'slice{i + 1}')(input_slices[-1]))
            target_slices.append(getattr(self, f'slice{i + 1}')(target_slices[-1]))

        # 使用层权重计算加权损失
        weighted_losses = [w * self.loss(input_slice, target_slice)
                           for w, input_slice, target_slice in
                           zip(self.layer_weights, input_slices[0:], target_slices[0:])]

        # Combine losses from all slices. Weights can be adjusted.
        total_loss = sum(weighted_losses)
        return total_loss


"""
https://github.com/swz30/MPRNet/blob/main/Denoising/losses.py
"""
class EdgeLoss_MPRNet(nn.Module):
    def __init__(self):
        super(EdgeLoss_MPRNet, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)            # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter)         # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss







"""
边缘损失（Edge Loss）旨在强化模型对图像边缘的重建能力。
这对于去反射任务尤其重要，因为边缘信息通常对于图像的视觉感知至关重要。
"""
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        # Define Sobel operators with groups=3 to process each channel independently
        self.sobel_x = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=3)
        self.sobel_y = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=3)
        self.loss = CharbonnierLoss()

        # Initialize Sobel kernels for each channel
        sobel_kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)

        # Repeat kernels for each input channel and use groups to apply separately
        self.sobel_x.weight = nn.Parameter(sobel_kernel_x.repeat(3, 1, 1, 1), requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y.repeat(3, 1, 1, 1), requires_grad=False)

    def forward(self, input, target):
        # Calculate edge maps for input and target
        edge_input_x = self.sobel_x(input)
        edge_input_y = self.sobel_y(input)
        edge_target_x = self.sobel_x(target)
        edge_target_y = self.sobel_y(target)

        # Compute L1 loss between edge maps of input and target
        loss_x = self.loss(edge_input_x, edge_target_x)
        loss_y = self.loss(edge_input_y, edge_target_y)

        # Total edge loss
        loss = loss_x + loss_y
        return loss

