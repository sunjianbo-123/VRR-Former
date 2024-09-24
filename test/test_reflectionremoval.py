import numpy as np
import os, sys, math
import argparse
from tqdm import tqdm
from einops import rearrange, repeat

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ptflops import get_model_complexity_info

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, '../dataset/'))
sys.path.append(os.path.join(dir_name, '..'))

import scipy.io as sio
from dataset.dataset_motiondeblur import *



from model_DualPathFusion import UNet, Uformer
from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss



import options
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='reflectionremoval')).parse_args()
import utils


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

results_dir = os.path.join(opt.result_dir, 'reflectionremoval', opt.dataset, opt.arch + opt.env)
utils.mkdir(results_dir)

# test:输入图片大小不变 没有进行crop    val:中心点crop   train:随机crop
test_dataset = get_validation_rr_data(opt.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

model_restoration = utils.get_arch(opt)
# model_restoration = torch.nn.DataParallel(model_restoration)

utils.load_checkpoint(model_restoration, opt.weights)
print("===>Testing using weights: ", opt.weights)

model_restoration.cuda()
model_restoration.eval()


def expand2square(timg, factor=16.0):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h, w) / float(factor)) * factor)

    img = torch.zeros(1, 3, X, X).type_as(timg)      # 3, h,w
    mask = torch.zeros(1, 1, X, X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
    mask[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)].fill_(1)

    return img, mask


with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_gt = data_test[0].numpy().squeeze().transpose((1, 2, 0))
        rgb_noisy, mask = expand2square(data_test[1].cuda(), factor=128)
        filenames = data_test[2]

        rgb_restored = model_restoration(rgb_noisy)
        rgb_restored = torch.masked_select(rgb_restored, mask.bool()).reshape(1, 3, rgb_gt.shape[0], rgb_gt.shape[1])
        rgb_restored = torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))

        psnr = psnr_loss(rgb_restored, rgb_gt)
        ssim = ssim_loss(rgb_restored, rgb_gt, channel_axis=2, data_range=1.0, multichannel=True)
        psnr_val_rgb.append(psnr)
        ssim_val_rgb.append(ssim)
        print("PSNR:", psnr, ", SSIM:", ssim, filenames[0], rgb_restored.shape)
        utils.save_img(os.path.join(results_dir, filenames[0] + '.PNG'), img_as_ubyte(rgb_restored))
        with open(os.path.join(results_dir, 'psnr_ssim.txt'), 'a') as f:
            f.write(filenames[0] + '.PNG ---->' + "PSNR: %.4f, SSIM: %.4f] " % (psnr, ssim) + '\n')
psnr_val_rgb = sum(psnr_val_rgb) / len(test_dataset)
ssim_val_rgb = sum(ssim_val_rgb) / len(test_dataset)
print("PSNR: %f, SSIM: %f " % (psnr_val_rgb, ssim_val_rgb))
with open(os.path.join(results_dir, 'psnr_ssim.txt'), 'a') as f:
    f.write("Arch:" + opt.arch + opt.env + ", PSNR: %.4f, SSIM: %.4f] " % (psnr_val_rgb, ssim_val_rgb) + '\n')