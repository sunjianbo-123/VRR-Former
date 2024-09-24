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
from losses import CharbonnierLoss
from losses import SSIMLoss
from losses import PerceptualLossVGG19
from losses import EdgeLoss
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
criterion_ssim = SSIMLoss(window_size=11, size_average=True).cuda()
# 可以更精细地控制不同特征层在总感知损失中的贡献，以此来调整模型对不同层级特征的重视程度
criterion_perceptual = PerceptualLossVGG19(resize=True,
                                           layer_weights=[1, 1 / 2.6, 1 / 4.8, 1 / 3.7, 1 / 5.6, 10 / 1.5]).cuda()
criterion_edge = EdgeLoss().cuda()
# ToDo: set the weights of different losses
charbonnier_weight = 1
ssim_weight = 1
perceptual_weight = 1
edge_weight = 1

# Pareto optimization to balance muilti-objective optimization
class ParetoMultiObjectiveOptimizer:
    def __init__(self, initial_weights, num_objectives):
        self.num_objectives = num_objectives
        self.weights = torch.tensor(initial_weights, dtype=torch.float32, device='cuda')
        self.pareto_front = []

    def update_weights(self, current_losses):
        # 将当前损失向量添加到Pareto前沿跟踪列表中
        is_dominated = False
        non_dominated = []
        for front in self.pareto_front:
            if self.dominates(current_losses, front):
                is_dominated = True
            elif not self.dominates(front, current_losses):
                non_dominated.append(front)
        if not is_dominated:
            non_dominated.append(current_losses)

        self.pareto_front = non_dominated

        # 更新权重以反映Pareto前沿的新状态
        self.weights = self.calculate_weights_from_pareto_front()

    def dominates(self, a, b):
        # 检查向量a是否支配向量b
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

    def calculate_weights_from_pareto_front(self):
        # 初始化权重为零
        weights = torch.zeros(self.num_objectives, device="cuda")
        for loss in self.pareto_front:
            normalized_loss = loss / torch.sum(loss)
            weights += normalized_loss
        # 归一化权重和，确保权重和为1
        weights = torch.clamp(weights, min=0)  # 首先确保没有负权重
        sum_weights = torch.sum(weights)
        if sum_weights > 0:
            weights = weights / sum_weights
        return weights

    def compute_total_loss(self, current_losses):
        if not isinstance(current_losses, torch.Tensor):
            current_losses = torch.tensor(current_losses, device='cuda', dtype=torch.float32)
        # 计算总损失
        total_loss = torch.dot(self.weights, current_losses)
        return total_loss

# 初始化损失权重
initial_weights = [1.0] * 4
optimizer_weights = ParetoMultiObjectiveOptimizer(initial_weights, num_objectives=4)



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
# ######### validation ###########
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
best_epoch = 0
best_iter = 0
eval_now = len(train_loader) // 4
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()

# 初始化损失记录字典
loss_history = {
    'loss_charbonnier': [],
    'loss_ssim': [],
    'loss_perceptual': [],
    'loss_edge': [],
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

            loss_charbonnier = criterion_charbonnier(restored, target)
            loss_ssim = criterion_ssim(restored, target)
            loss_perceptual = criterion_perceptual(restored, target)
            loss_edge = criterion_edge(restored, target)
            current_losses = torch.stack([loss_charbonnier, loss_ssim, loss_perceptual, loss_edge])
            total_loss = optimizer_weights.compute_total_loss(current_losses)
            # total_loss = loss_charbonnier * charbonnier_weight + loss_ssim * ssim_weight + loss_perceptual * perceptual_weight + loss_edge * edge_weight

            # 记录损失
            loss_history['loss_charbonnier'].append(criterion_charbonnier(restored, target).item())
            loss_history['loss_ssim'].append(criterion_ssim(restored, target).item())
            loss_history['loss_perceptual'].append(criterion_perceptual(restored, target).item())
            loss_history['loss_edge'].append(criterion_edge(restored, target).item())
            loss_history['loss_total'].append(total_loss.item())
            # 将字典转换为DataFrame
            df_losses = pd.DataFrame(loss_history)
            # 保存DataFrame到CSV文件
            df_losses.to_csv('./loss_history.csv', index=False)

        # 损失本身在autocast块内部计算，但梯度的缩放和应用（即反向传播和优化）通常在该块之外进行，以确保使用的是完整的精度。
        loss_scaler(total_loss, optimizer, parameters=model_restoration.parameters())
        epoch_loss += total_loss.item()

        # 调整损失权重
        optimizer_weights.update_weights(current_losses.detach())
        # 管理好current_losses的图状态：使用.detach()
        # 确保从图中分离出用于权重更新计算的损失，避免因旧图状态造成错误。

        #### Evaluation ####
        if (i + 1) % eval_now == 0 and i > 0:
            with torch.no_grad():
                model_restoration.eval()
                psnr_val_rgb = []
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    filenames = data_val[2]
                    with torch.cuda.amp.autocast():
                        restored = model_restoration(input_)
                    restored = torch.clamp(restored, 0, 1)
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())

                psnr_val_rgb = sum(psnr_val_rgb) / len_valset

                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict()
                                }, os.path.join(model_dir, "model_best.pth"))

                print(
                    "[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (
                    epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr))
                with open(logname, 'a') as f:
                    f.write(
                        "[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " \
                        % (epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr) + '\n')
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
