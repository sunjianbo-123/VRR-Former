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



######### Logs dir ###########
log_dir = os.path.join(opt.save_dir, 'reflectionremoval', opt.dataset, opt.arch + opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat() + '.txt')
print("Now time is : ", datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# 选择模型
######### Model ###########
model_restoration = utils.get_arch(opt)

with open(logname, 'a') as f:
    f.write(str(opt) + '\n')
    f.write(str(model_restoration) + '\n')

######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
    optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")

######### DataParallel ###########
model_restoration = torch.nn.DataParallel(model_restoration)
model_restoration.cuda()


######### Scheduler ###########
if opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch - warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
else:
    step = 50
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()

######### Resume ###########
# 是否需要从之前保存的模型状态恢复。如果设置为True，则执行恢复过程
if opt.resume:
    path_chk_rest = opt.pretrain_weights
    checkpoint = torch.load(path_chk_rest)
    model_restoration.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])  # 加载学习率调度器状态
    print("Resume from " + path_chk_rest)
    start_epoch = checkpoint['epoch'] + 1
    new_lr = scheduler.get_last_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6)

######### Loss ###########
# 实例化损失函数
criterion_charbonnier = CharbonnierLoss().cuda()

# 可以更精细地控制不同特征层在总感知损失中的贡献，以此来调整模型对不同层级特征的重视程度
criterion_perceptual = PerceptualLossVGG19(resize=True,
                                           layer_weights=[1, 1 / 2.6, 1 / 4.8, 1 / 3.7, 1 / 5.6, 1 / 7.8]).cuda()
criterion_edge = EdgeLoss_MPRNet().cuda()


charbonnier_weight = 2
ssim_weight = 1
edge_weight = 1.5
perceptual_weight = 0.5



######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size': opt.train_ps}
train_dataset = get_training_data(opt.train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                          num_workers=opt.train_workers, pin_memory=False, drop_last=False)
img_options_val = {'patch_size': opt.val_ps}
val_dataset = get_validation_rr_data(opt.val_dir, img_options_val)
val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False,
                        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)
len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset, ", sizeof validation set: ", len_valset)



######### validation ###########
with torch.no_grad():
    model_restoration.eval()
    psnr_dataset = []
    psnr_model_init = []
    for ii, data_val in enumerate((val_loader), 0):
        target = data_val[0].cuda()
        input_ = data_val[1].cuda()
        with torch.cuda.amp.autocast():
            restored = model_restoration(input_)
            restored = torch.clamp(restored, 0, 1)
        psnr_dataset.append(utils.batch_PSNR(input_, target, False).item())
        psnr_model_init.append(utils.batch_PSNR(restored, target, False).item())
    psnr_dataset = sum(psnr_dataset) / len_valset
    psnr_model_init = sum(psnr_model_init) / len_valset
    print('Input & GT (PSNR) -->%.4f dB' % (psnr_dataset), ', Model_init & GT (PSNR) -->%.4f dB' % (psnr_model_init))

######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.nepoch))
best_psnr = 0
best_ssim = 0
best_epoch = 0
best_iter = 0
eval_now = len(train_loader) // 4
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()

# 初始化损失记录字典
loss_history = {
    'loss_charbonnier': [],
    'loss_edge': [],
    'perceptual_loss': [],
    'loss_total': []
         }

for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1


    for i, data in enumerate(tqdm(train_loader), 0):
        # zero_grad
        optimizer.zero_grad()
        target = data[0].cuda()
        input_ = data[1].cuda()

        # AMP(Automatic Mixed Precision) to accelerate training
        # 一种优化技术，可以减少计算过程中所需的内存带宽，
        # 同时加速模型的训练，而不会显著影响模型的准确性。
        with torch.cuda.amp.autocast():
            restored = model_restoration(input_)
            restored = torch.clamp(restored, 0, 1)

            loss_charbonnier = criterion_charbonnier(restored, target)
            # loss_ssim = criterion_ssim(restored, target)
            loss_perceptual = criterion_perceptual(restored, target)
            loss_edge = criterion_edge(restored, target)

            # # calculate ssim & ms-ssim for each image
            # ssim_val = ssim(restored, target, data_range=255, size_average=False, nonnegtive_ssim=True)  # return (N,)
            # ms_ssim_val = ms_ssim(restored, target, data_range=255, size_average=False)                  # (N,)

            # set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details
            # ssim_loss = 1 - ssim(restored, target, data_range=255, size_average=True,nonnegtive_ssim=True)     # return a scalar
            # ms_ssim_loss = 1 - ms_ssim(restored, target, data_range=255, size_average=True)

            # reuse the gaussian kernel with SSIM & MS_SSIM.
            # ssim_module = SSIM(win_size=11,win_sigma=1.5,data_range=1, size_average=True, channel=3, nonnegative_ssim=True)       # channel=1 for grayscale images
            # ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=3)

            # ssim_loss = 1 - ssim_module(restored, target)
            # ms_ssim_loss = 1 - ms_ssim_module(restored, target)

            # total_loss = loss_charbonnier * charbonnier_weight + loss_ssim * ssim_weight + loss_perceptual * perceptual_weight + loss_edge * edge_weight
            total_loss = loss_charbonnier * charbonnier_weight + loss_perceptual * perceptual_weight + loss_edge * edge_weight

            # 记录损失
            loss_history['loss_charbonnier'].append(loss_charbonnier.item())
            # loss_history['loss_msssim'].append(ssim_loss.item())
            loss_history['perceptual_loss'].append(loss_perceptual.item())
            loss_history['loss_edge'].append(loss_edge.item())
            loss_history['loss_total'].append(total_loss.item())
            # 将字典转换为DataFrame
            df_losses = pd.DataFrame(loss_history)
            # 保存DataFrame到CSV文件
            df_losses.to_csv('./loss_history.csv', index=False)

        # 损失本身在autocast块内部计算，但梯度的缩放和应用（即反向传播和优化）通常在该块之外进行，以确保使用的是完整的精度。
        loss_scaler(total_loss, optimizer, parameters=model_restoration.parameters())
        epoch_loss += total_loss.item()



        #### Evaluation ####
        if (i + 1) % eval_now == 0 and i > 0:
            with torch.no_grad():
                model_restoration.eval()
                psnr_val_rgb = []
                ssim_val_rgb = []
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    filenames = data_val[2]
                    with torch.cuda.amp.autocast():
                        restored = model_restoration(input_)

                    restored = torch.clamp(restored, 0, 1)
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())
                    ssim_val_rgb.append(utils.batch_SSIM(restored, target, False).item())

                psnr_val_rgb = sum(psnr_val_rgb) / len_valset
                ssim_val_rgb = sum(ssim_val_rgb) / len_valset

                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch_psnr = epoch
                    best_iter_psnr = i
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict()
                                }, os.path.join(model_dir, "model_best.pth"))
                if ssim_val_rgb > best_ssim:
                    best_ssim = ssim_val_rgb
                    best_epoch_ssim = epoch
                    best_iter_ssim = i

                print(f"[Epoch:{epoch}/Iter:{i} PSNR:{psnr_val_rgb:.4f}\tSSIM:{ssim_val_rgb:.4f}]  ----  [Best_PSNR:{best_psnr:.4f}  {best_epoch_psnr}/{best_iter_psnr}\tBest_SSIM:{best_ssim:.4f}  {best_epoch_ssim}/{best_iter_ssim}]")
                with open(logname, 'a') as f:
                    f.write(
                     f"[Epoch:{epoch}/Iter:{i}  PSNR:{psnr_val_rgb:.4f}\tSSIM:{ssim_val_rgb:.4f}]  ----  [Best_PSNR:{best_psnr:.4f}  {best_epoch_psnr}/{best_iter_psnr}\tBest_SSIM:{best_ssim:.4f}  {best_epoch_ssim}/{best_iter_ssim}" + '\n')

                model_restoration.train()
                torch.cuda.empty_cache()
    scheduler.step()


    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(logname, 'a') as f:
        f.write(
            "Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                                epoch_loss,
                                                                                scheduler.get_lr()[0]) + '\n')

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

    if epoch % opt.checkpoint == 0:
        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                    }, os.path.join(model_dir, "model_epoch_{}.pth".format(epoch)))
print("Now time is : ", datetime.datetime.now().isoformat())
