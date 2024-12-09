import datetime
import os
import time
import torch
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import sys
from torch.cuda import amp
import models
import argparse
import math
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from copy import deepcopy
import pandas as pd
_seed_ = 2022
import random
import json
from spikingjelly.activation_based import functional
from models.spiking_mlp_hlop_multimodal import PretrainMultimodalSpikingNN
random.seed(_seed_)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np
np.random.seed(_seed_)

torch.set_num_threads(4)


def test(args, model, img_x, audio_x, y, task_id):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    bar = Bar('Processing', max=((img_x.size(0)-1)//args.b+1))

    test_loss = 0
    test_acc = 0
    test_samples = 0
    batch_idx = 0

    r=np.arange(img_x.size(0))
    with torch.no_grad():
        for i in range(0, len(r), args.b):
            if i + args.b <= len(r):
                index = r[i : i + args.b]
            else:
                index = r[i:]
            batch_idx += 1
            img_input = img_x[index].float().cuda()
            audio_input = audio_x[index].float().cuda()
            # repeat for time steps
            img_input = img_input.unsqueeze(1)
            img_input = img_input.repeat(1, args.timesteps, 1, 1, 1)
            audio_input = audio_input.unsqueeze(1)
            audio_input = audio_input.repeat(1, args.timesteps, 1)

            label = y[index].cuda()

            out = model(img_input, audio_input, task_id, projection=False, update_hlop=False)
            label = label.squeeze(1).long()
            loss = F.cross_entropy(out, label)
                
            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out.argmax(1) == label).float().sum().item()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(out.data, label.data, topk=(1, 5))
            losses.update(loss, img_input.size(0))
            top1.update(prec1.item(), img_input.size(0))
            top5.update(prec5.item(), img_input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx,
                        size=((img_x.size(0)-1)//args.b+1),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
    bar.finish()

    test_loss /= test_samples
    test_acc /= test_samples

    return test_loss, test_acc

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from dataloader.scene import ImgAudioDataset,normalize,img_train_transform,audio_val_transform,audio_train_transform,img_val_transform

def main():
    learning_rate = 0.001
    epochs = 50
    device = "cuda:0"
    with open('models/multimodal_config.json', 'r') as json_file:
        multi_fusion_dim = json.load(json_file)
    # model = Autoencoder().to(device)
    model = PretrainMultimodalSpikingNN(multi_fusion_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    reconstruction_loss_fn = torch.nn.MSELoss()  # 你可以选择 MSE 或 L1 损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        ##数据集加载
        model_weight = "./pre_weight"
        os.makedirs(model_weight, exist_ok=True)  # 如果文件夹不存在，则创建
        best_accuracy = 0.0
        data_dir='/home/haichao/tzq/HLOP-SNN/datasets/scene'
        data = pd.read_csv(os.path.join(data_dir, 'dataset.csv'), delimiter=',', nrows=None)
        data_train = np.array(data)

        audio = data_train[:,1:-2].astype('float32') #last index of the interval isn't included in the range : CLASS1
        labels = data_train[:,-1]
        img_paths = data['IMAGE']

        classes = ["FOREST", "CLASSROOM", "CITY", "RIVER", "GROCERY-STORE","JUNGLE","BEACH","FOOTBALL-MATCH","RESTAURANT"]
        for index,class_name in enumerate(classes):
            labels = np.where(labels == class_name, index, labels)

        labels.astype('int32')
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        random.seed(0)
        np.random.seed(0)
        ## 划分数据集
        img_train, img_temp, audio_train, audio_temp, labels_train, labels_temp = train_test_split(img_paths, 
                                                                                                audio, labels, train_size=0.6)
        img_val, img_test, audio_val, audio_test, labels_val, labels_test = train_test_split(img_temp, 
                                                                                            audio_temp, labels_temp, train_size=0.5)

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        random.seed(0)
        np.random.seed(0)

        audio_train = normalize(audio_train)
        audio_val = normalize(audio_val)

        train_data = ImgAudioDataset(root_dir=data_dir, 
                                    img_data=img_train, audio_data=audio_train, labels=labels_train, 
                                img_transform = img_train_transform, audio_transform=audio_train_transform)
        val_data = ImgAudioDataset(root_dir=data_dir, 
                                img_data=img_val, audio_data=audio_val, labels=labels_val, 
                                img_transform = img_val_transform, audio_transform=audio_val_transform)
        
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4, pin_memory=True,drop_last=True )
        valid_loader = DataLoader(val_data, batch_size=16, shuffle=True, num_workers=4, pin_memory=True,drop_last=True )
        
        # from dataloader import cifar100 as cf100
        # data, taskcla, inputsize = cf100.get(data_dir=args.data_dir, seed=_seed_)
        
        total_loss = 0.0
        correct = 0
        total = 0

        # 训练阶段
        for visual_input, auditory_input, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()

            # 数据转移到指定设备
            visual_input = visual_input.to(device)
            auditory_input = auditory_input.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(visual_input, auditory_input)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播与优化
            loss.backward()
            optimizer.step()
            functional.reset_net(model)

            # 累计损失和统计指标
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # 打印训练结果
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

        # 每 5 个 epoch 验证一次
        if (epoch + 1) % 5 == 0:
            # 模型切换到评估模式
            model.eval()

            # 初始化验证阶段的统计变量
            total_val_loss = 0.0
            correct_val = 0
            total_val = 0

            # 禁用梯度计算
            with torch.no_grad():
                for visual_input, auditory_input, labels in tqdm(valid_loader, desc=f"Validating Epoch {epoch + 1}/{epochs}"):
                    # 数据转移到指定设备
                    visual_input = visual_input.to(device)
                    auditory_input = auditory_input.to(device)
                    labels = labels.to(device)

                    # 前向传播
                    outputs = model(visual_input, auditory_input)

                    # 计算损失
                    loss = criterion(outputs, labels)
                    functional.reset_net(model)
                    # 累计损失和统计指标
                    total_val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct_val += (predicted == labels).sum().item()
                    total_val += labels.size(0)

            # 计算验证集准确率
            val_accuracy = 100 * correct_val / total_val
            print(f"Validation Loss: {total_val_loss / len(valid_loader):.4f}, Accuracy: {val_accuracy:.2f}%")

            # 保存最佳模型
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model_path = os.path.join(model_weight, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved to {best_model_path}")

        # 保存每轮最后的模型
        last_model_path = os.path.join(model_weight, "last_model.pth")
        torch.save(model.state_dict(), last_model_path)
        print(f"Last model saved to {last_model_path}")
            

if __name__ == '__main__':
    main()