import os
import sys
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, '../dataset/'))
sys.path.append(os.path.join(dir_name, '..'))
print(dir_name)

import argparse
import options

######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='reflectionremoval')).parse_args()
print(opt)


import utils
from dataset.dataset_motiondeblur import *

######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu


import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx
from losses import CharbonnierLoss,PerceptualLossVGG19,EdgeLoss_MPRNet,ssim, ms_ssim, SSIM, MS_SSIM
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler
from autogluon.core import HyperparameterTuner, RandomSearcher
import torch
import torch.nn as nn


# 选择模型
######### Model ###########
model_restoration = utils.get_arch(opt)



######### Loss ###########

criterion_charbonnier = CharbonnierLoss().cuda()
criterion_perceptual = PerceptualLossVGG19(resize=True,
                                           layer_weights=[1, 1 / 2.6, 1 / 4.8, 1 / 3.7, 1 / 5.6, 1 / 7.8]).cuda()
criterion_edge = EdgeLoss_MPRNet().cuda()



######### DataLoader ###########
img_options_train = {'patch_size': opt.train_ps}
train_dataset = get_training_data(opt.train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                          num_workers=opt.train_workers, pin_memory=False, drop_last=False)
img_options_val = {'patch_size': opt.val_ps}
val_dataset = get_validation_rr_data(opt.val_dir, img_options_val)
val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False,
                        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)




# 定义损失函数
def custom_loss(weights):
    def loss_fn(outputs, labels):
        loss1 = criterion_charbonnier()(outputs, labels)
        loss2 = criterion_perceptual()(outputs, labels)
        loss3 = criterion_edge()(outputs, labels)
        return weights[0] * loss1 + weights[1] * loss2 + weights[2] * loss3
    return loss_fn

# 损失函数权重的搜索空间
search_space = {
    'weights': (0.1, 10.0, 3)  # 这里的 (0.1, 10.0, 3) 指从 0.1 到 10.0 之间均匀采样 3 个权重
}

# 配置调参器
hp_tuner = HyperparameterTuner(
    model_restoration,
    custom_loss,
    resource={'num_cpus': 1, 'num_gpus': 3},
    searcher='random',  # 使用随机搜索
    search_space=search_space,
    num_trials=100  # 进行 100 次试验
)

# 运行调参
hp_tuner.fit(train_loader, val_loader)