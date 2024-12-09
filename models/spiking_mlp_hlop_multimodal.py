import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from modules.neuron_dsr import LIFNeuron, IFNeuron
from modules.neuron_dsr import rate_spikes, weight_rate_spikes
from modules.proj_conv import Conv2dProj, SSConv2dProj
from modules.proj_linear import LinearProj, SSLinear, SSLinearProj, FALinear, FALinearProj
from modules.hlop_module import HLOP
import numpy as np
import torchvision.models as models
# from models.spiking_mlp_hlop_multimodal import MultimodalSpikingNN
import json
# spikingjelly.activation_based.examples.conv_fashion_mnist
import torchvision
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from torch.cuda import amp
import sys
import datetime
from spikingjelly import visualizing


__all__ = [
    'spiking_MLP'
]
feat = 784
class spiking_MLP_multimodal(nn.Module):
    def __init__(self, snn_setting, num_classes=10, n_hidden=800, share_classifier=True, neuron_type='lif', ss=False, fa=False, hlop_with_wfr=True, hlop_spiking=False, hlop_spiking_scale=20., hlop_spiking_timesteps=1000.):
        super(spiking_MLP_multimodal, self).__init__()
        self.timesteps = snn_setting['timesteps']
        self.snn_setting = snn_setting
        self.neuron_type = neuron_type

        self.share_classifier = share_classifier
        self.n_hidden = n_hidden
        self.ss = ss
        self.fa = fa
        self.hlop_modules = nn.ModuleList([])
        self.hlop_with_wfr = hlop_with_wfr
        self.hlop_spiking = hlop_spiking
        self.hlop_spiking_scale = hlop_spiking_scale
        self.hlop_spiking_timesteps = hlop_spiking_timesteps

        if self.neuron_type == 'lif':
            self.tau = snn_setting['tau']
            self.delta_t = snn_setting['delta_t']
            self.weight_avg = True
        elif self.neuron_type == 'if':
            self.weight_avg = False
        else:
            raise NotImplementedError('Please use IF or LIF model.')
        if ss:
            self.fc1 = SSLinearProj(feat, n_hidden, bias=False)
            self.fc2 = SSLinearProj(n_hidden, n_hidden, bias=False)
        elif fa:
            self.fc1 = FALinearProj(feat, n_hidden, bias=False)
            self.fc2 = FALinearProj(n_hidden, n_hidden, bias=False)
        else:
            self.fc1 = LinearProj(feat, n_hidden, bias=False)
            self.fc2 = LinearProj(n_hidden, n_hidden, bias=False)

        if self.neuron_type == 'lif':
            self.sn1 = LIFNeuron(self.snn_setting)
            self.sn2 = LIFNeuron(self.snn_setting)
        elif self.neuron_type == 'if':
            self.sn1 = IFNeuron(self.snn_setting)
            self.sn2 = IFNeuron(self.snn_setting)
        else:
            raise NotImplementedError('Please use IF or LIF model.')

        self.hlop_modules.append(HLOP(feat, lr=0.05, spiking=self.hlop_spiking, spiking_scale=self.hlop_spiking_scale, spiking_timesteps=self.hlop_spiking_timesteps))
        self.hlop_modules.append(HLOP(n_hidden, lr=0.05, spiking=self.hlop_spiking, spiking_scale=self.hlop_spiking_scale, spiking_timesteps=self.hlop_spiking_timesteps))
        if share_classifier:
            self.hlop_modules.append(HLOP(n_hidden, lr=0.05, spiking=self.hlop_spiking, spiking_scale=self.hlop_spiking_scale, spiking_timesteps=self.hlop_spiking_timesteps))
            if ss:
                self.classifiers = nn.ModuleList([SSLinearProj(n_hidden, num_classes, bias=False)])
            elif fa:
                self.classifiers = nn.ModuleList([FALinearProj(n_hidden, num_classes, bias=False)])
            else:
                self.classifiers = nn.ModuleList([LinearProj(n_hidden, num_classes, bias=False)])
        else:
            if ss:
                self.classifiers = nn.ModuleList([SSLinear(n_hidden, num_classes, bias=False)])
            elif fa:
                self.classifiers = nn.ModuleList([FALinear(n_hidden, num_classes, bias=False)])
            else:
                self.classifiers = nn.ModuleList([nn.Linear(n_hidden, num_classes, bias=False)])
        self.classifier_num = 1
        # resnet = models.resnet50(pretrained=True)
        # # 2. 去除最后的全连接层，保留卷积部分作为特征提取器
        # self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        with open('models/multimodal_config.json', 'r') as json_file:
            multi_fusion_dim = json.load(json_file)
        self.feature_fusion =  MultimodalSpikingNN(multi_fusion_dim) 
        # self.feature_fusion = nn.Sequential(*list(self.feature_fusion.children())[:-1])     
        pre_weight = "./pre_weight/scene/last_model.pth"
        model_path = pre_weight  # 替换为实际路径
        # 加载模型的状态字典
        checkpoint = torch.load(model_path)
        # 获取模型的状态字典，排除分类头部分（假设分类头是fc）
        model_dict = self.feature_fusion.state_dict()
        # print(f"model_dict = {model_dict}")
        # for k, v in checkpoint.items():
        #     print(f"k = {k}")
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        # 更新模型的权重
        model_dict.update(pretrained_dict)
        self.feature_fusion.load_state_dict(model_dict)
        # 冻结所有层的权重
        for param in self.feature_fusion.parameters():
            param.requires_grad = False

    def set_hlop_value(self, weight, index=0, **kwargs):
        self.hlop_modules[index].set_subspace(weight, **kwargs)

    def get_hlop_value(self, index=0, **kwargs):
        return self.hlop_modules[index].get_weight_value(**kwargs)

    def forward(self, x, task_id=None, projection=False, proj_id_list=[0], update_hlop=False, fix_subspace_id_list=None):
        x1 = x["x1"]
        x2 = x["x2"]
        # x1 = torch.cat([x1[:,_,:,:,:] for _ in range(self.timesteps)], 0)
        # x2 = torch.cat([x2[:,_,:] for _ in range(self.timesteps)], 0)
        # print("shape:", x1.shape)
        # print("shape:", x2.shape)
        with torch.no_grad():  # 不需要计算梯度
            # 多模态
            x1 = x1.permute(1, 0, 2, 3,4)
            x2 = x2.permute(1, 0, 2).unsqueeze(1)
            x = self.feature_fusion(x1,x2)
            print(f"x = {x.shape}")
            x = x.mean(0)
            print(f"x = {x.shape}")
            # 声音
            # x1 = torch.zeros_like(x1)
            # x = self.feature_fusion(x1,x2)
            # 图片
            # x2 = torch.zeros_like(x2)  
            # x = self.feature_fusion(x1,x2)

            # x = self.feature_extractor(x1)
            # x = x.squeeze(2).squeeze(2)
        # print("shape:", x.shape)
        # exit()
        x= x.repeat(self.timesteps,1,1)
        x = x.view(-1, 784)

        # print("x_mean:", torch.mean(x, dim=1))
        # print('x:', x)
        # exit()
        if projection:
            proj_func = self.hlop_modules[0].get_proj_func(subspace_id_list=proj_id_list)
            x_ = self.fc1(x, projection=True, proj_func=proj_func)
        else:
            x_ = self.fc1(x, projection=False)
        if update_hlop:
            if self.hlop_with_wfr:
                # update hlop by weighted firing rate
                x = weight_rate_spikes(x, self.timesteps, self.tau, self.delta_t)
            with torch.no_grad():
                self.hlop_modules[0].forward_with_update(x, fix_subspace_id_list=fix_subspace_id_list)
        x = self.sn1(x_)
        if projection:
            proj_func = self.hlop_modules[1].get_proj_func(subspace_id_list=proj_id_list)
            x_ = self.fc2(x, projection=True, proj_func=proj_func)
        else:
            x_ = self.fc2(x, projection=False)
        if update_hlop:
            if self.hlop_with_wfr:
                # update hlop by weighted firing rate
                x = weight_rate_spikes(x, self.timesteps, self.tau, self.delta_t)
            with torch.no_grad():
                self.hlop_modules[1].forward_with_update(x, fix_subspace_id_list=fix_subspace_id_list)
        x = self.sn2(x_)
        if not self.share_classifier:
            assert task_id is not None
            x = self.classifiers[task_id](x)
        else:
            m = self.classifiers[0]
            if projection:
                proj_func = self.hlop_modules[2].get_proj_func(subspace_id_list=proj_id_list)
                x_ = m(x, projection=True, proj_func=proj_func)
            else:
                x_ = m(x, projection=False)
            if update_hlop:
                if self.hlop_with_wfr:
                    # update hlop by weighted firing rate
                    x = weight_rate_spikes(x, self.timesteps, self.tau, self.delta_t)
                with torch.no_grad():
                    self.hlop_modules[2].forward_with_update(x, fix_subspace_id_list=fix_subspace_id_list)
            x = x_

        out = x
        if self.weight_avg:
            out = weight_rate_spikes(out, self.timesteps, self.tau, self.delta_t)
        else:
            out = rate_spikes(out, self.timesteps)
        return out

    def forward_features(self, x):
        x = torch.cat([x[:,_,:,:,:] for _ in range(self.timesteps)], 0)
        inputs = x.view(-1, 784)
        feature_list = []
        x_ = self.fc1(inputs, projection=False)
        if self.hlop_with_wfr:
            # calculate weighted firing rate
            inputs = weight_rate_spikes(inputs, self.timesteps, self.tau, self.delta_t)
        feature_list.append(inputs.detach().cpu())
        inputs = self.sn1(x_)
        x_ = self.fc2(inputs, projection=False)
        if self.hlop_with_wfr:
            # calculate weighted firing rate
            inputs = weight_rate_spikes(inputs, self.timesteps, self.tau, self.delta_t)
        feature_list.append(inputs.detach().cpu())
        inputs = self.sn2(x_)
        if self.share_classifier:
            if self.hlop_with_wfr:
                # calculate weighted firing rate
                inputs = weight_rate_spikes(inputs, self.timesteps, self.tau, self.delta_t)
            feature_list.append(inputs.detach().cpu())

        return feature_list

    def add_classifier(self, num_classes):
        self.classifier_num += 1
        if self.ss:
            self.classifiers.append(SSLinear(self.n_hidden, num_classes).to(self.classifiers[0].weight.device))
        elif self.fa:
            self.classifiers.append(FALinear(self.n_hidden, num_classes).to(self.classifiers[0].weight.device))
        else:
            self.classifiers.append(nn.Linear(self.n_hidden, num_classes).to(self.classifiers[0].weight.device))

    def merge_hlop_subspace(self):
        for m in self.hlop_modules:
            m.merge_subspace()

    def add_hlop_subspace(self, out_numbers):
        if isinstance(out_numbers, list):
            for i in range(len(self.hlop_modules)):
                self.hlop_modules[i].add_subspace(out_numbers[i])
        else:
            for m in self.hlop_modules:
                m.add_subspace(out_numbers)

    #def adjust_hlop_lr(self, gamma):
    #    for m in self.hlop_modules:
    #        m.adjust_lr(gamma)

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

            if isinstance(m, LIFNeuron) or isinstance(m, IFNeuron):
                if self.snn_setting['train_Vth']:
                    m.Vth.requires_grad = False



class FC_Reslayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(FC_Reslayer, self).__init__()
        self.fc1 = layer.Linear(in_features, out_features)
        self.fc2 = layer.Linear(out_features, out_features)
        self.sn1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.sn2 = neuron.IFNode(surrogate_function=surrogate.ATan())
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = self.sn1(self.fc1(x))
        x = x+self.fc2(x)
        x = self.sn2(x)
        return x
    
class FClayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(FClayer, self).__init__()
        self.fc = layer.Linear(in_features, out_features)
        self.sn = neuron.IFNode(surrogate_function=surrogate.ATan())
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = self.fc(x)
        x = self.sn(x)
        return x
    
class Nonact_FClayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(Nonact_FClayer, self).__init__()
        self.fc = layer.Linear(in_features, out_features)
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = self.fc(x)
        return x
        
# 定义视觉模态网络
class VisualModality(nn.Module):
    def __init__(self, dim):
        super(VisualModality, self).__init__()
        self.conv1 = layer.Conv2d(dim[0], 12, kernel_size=3, stride=2, padding=1)
        self.pool1 = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = layer.BatchNorm2d(12)
        self.conv2 = layer.Conv2d(12, 12, kernel_size=3, stride=2, padding=1)
        self.pool2 = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn2 = layer.BatchNorm2d(12)
        self.fc1 = FClayer(14 * 14 * 12, dim[1])
        self.fc2 = FClayer(dim[1], dim[2])
        self.fc3 = Nonact_FClayer(dim[2], dim[3])
        self.sn1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.sn2 = neuron.IFNode(surrogate_function=surrogate.ATan())
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = self.conv1(x)
        x = self.sn1(self.bn1(x))
        x = self.pool1(x)
        x = self.sn2(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
# 定义听觉模态网络
class AuditoryModality(nn.Module):
    def __init__(self, dim):
        super(AuditoryModality, self).__init__()
        self.fc_res1 = FC_Reslayer(dim[0], dim[1]) 
        self.fc_res2 = FC_Reslayer(dim[1], dim[2])
        self.fc_res3 = FC_Reslayer(dim[2], dim[3])
        self.fc_res4 = Nonact_FClayer(dim[3], dim[4])
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = self.fc_res1(x)
        x = self.fc_res2(x)
        x = self.fc_res3(x)
        x = self.fc_res4(x)
        return x

# 定义注意力机制
class AttentionMechanism(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionMechanism, self).__init__()
        self.fc1 = layer.Linear(input_dim, output_dim)
        self.fc2 = layer.Linear(input_dim, output_dim)
        self.sn = neuron.IFNode(surrogate_function=surrogate.ATan())
        functional.set_step_mode(self, step_mode='m')
    
    def forward(self, v, a, fused):
        # 计算注意力权重
        w_v = self.fc1(fused)
        w_a = self.fc2(fused)
        # 归一化权重
        epsilon = 1e-10
        # 对w_v和w_a进行归一化
        norm_factor = w_v.sum(dim=1, keepdim=True) + w_a.sum(dim=1, keepdim=True) + epsilon
        w_v = w_v / norm_factor
        w_a = w_a / norm_factor

        # 加权融合
        fused = w_v * v + w_a * a
        fused = self.sn(fused)  ## 特征脉冲化
        return fused
    
# 跨模态融合网络
class MultimodalSpikingNN(nn.Module):
    def __init__(self, dim):
        super(MultimodalSpikingNN, self).__init__()
        Visual_dim = dim["Visual"]
        Audio_dim = dim["Audio"]
        Attcross_dim = dim["Attention_cross"]
        self.visual_modality = VisualModality(Visual_dim)
        self.auditory_modality = AuditoryModality(Audio_dim)
        self.concat_fc = FClayer(Attcross_dim[0], Attcross_dim[1])
        self.attention = AttentionMechanism(Attcross_dim[1], Attcross_dim[2])  
        functional.set_step_mode(self, step_mode='m')

    def forward(self, visual_input, auditory_input):
        # 视觉模态
        v = self.visual_modality(visual_input)
        # 听觉模态
        a = self.auditory_modality(auditory_input)
        # 融合
        a = a.squeeze(1)
        fused = torch.cat((v, a), dim=2)  # 特征拼接
        fused = self.concat_fc(fused)
        # 注意力加权
        fused = self.attention(v, a, fused)
        return fused


class PretrainMultimodalSpikingNN(nn.Module):
    def __init__(self, dim,T = 4):
        super(PretrainMultimodalSpikingNN, self).__init__()
        self.T = T
        Visual_dim = dim["Visual"]
        Audio_dim = dim["Audio"]
        Attcross_dim = dim["Attention_cross"]
        self.visual_modality = VisualModality(Visual_dim)
        self.auditory_modality = AuditoryModality(Audio_dim)
        self.concat_fc = FClayer(Attcross_dim[0], Attcross_dim[1])
        self.attention = AttentionMechanism(Attcross_dim[1], Attcross_dim[2])  
        # num_classes = 9
        self.classifier = layer.Linear(Attcross_dim[2], 9)
        functional.set_step_mode(self, step_mode='m')

    def forward(self, visual_input, auditory_input):
        visual_input = visual_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1) 
        auditory_input = auditory_input.unsqueeze(0).repeat(self.T, 1, 1, 1)  
        # 视觉模态
        v = self.visual_modality(visual_input)
        # 听觉模态
        a = self.auditory_modality(auditory_input)
        # 融合
        a = a.squeeze(1)
        fused = torch.cat((v, a), dim=2)  # 特征拼接
        fused = self.concat_fc(fused)
        # 注意力加权
        fused = self.attention(v, a, fused)
        # print(f"fused = {fused[-1,100:150]}")
        out = self.classifier(fused)

        out = out.mean(0)
        functional.set_step_mode(self, step_mode='m')
        return out

