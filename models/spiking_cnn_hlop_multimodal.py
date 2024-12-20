import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from modules.neuron_dsr import LIFNeuron, IFNeuron
from modules.neuron_dsr import rate_spikes, weight_rate_spikes
from modules.proj_conv import Conv2dProj, SSConv2dProj
from modules.proj_linear import LinearProj, SSLinear, SSLinearProj
from modules.hlop_module import HLOP

__all__ = [
    'spiking_cnn',
]


cfg = {
    'A': [64, 'M', 128, 'M', 256, 'M'],
}


class CNNmultimodal(nn.Module):
    def __init__(self, dim, snn_setting, cnn_name, num_classes=10, share_classifier=False, neuron_type='lif', fc_size=4096, ss=False, hlop_with_wfr=True, hlop_spiking=False, hlop_spiking_scale=20., hlop_spiking_timesteps=1000., proj_type='input'):
        super(CNNmultimodal, self).__init__()

        self.timesteps = snn_setting['timesteps']
        self.snn_setting = snn_setting
        self.neuron_type = neuron_type

        self.share_classifier = share_classifier
        self.ss = ss
        self.hlop_with_wfr = hlop_with_wfr
        self.hlop_spiking = hlop_spiking
        self.hlop_spiking_scale = hlop_spiking_scale
        self.hlop_spiking_timesteps = hlop_spiking_timesteps
        # choice for projection type in bottom implementation
        # it is theoretically equivalent for input and weight, while weight enables acceleration of convolutional operations
        self.proj_type = proj_type

        self.init_channels = 3
        self.features, self.hlop_modules = self._make_layers(cfg[cnn_name])

        self.fc_size = fc_size

        if self.neuron_type == 'lif':
            self.tau = snn_setting['tau']
            self.delta_t = snn_setting['delta_t']
            self.weight_avg = True
        elif self.neuron_type == 'if':
            self.weight_avg = False
        else:
            raise NotImplementedError('Please use IF or LIF model.')

        if share_classifier:
            self.hlop_modules.append(HLOP(fc_size, spiking=self.hlop_spiking, spiking_scale=self.hlop_spiking_scale, spiking_timesteps=self.hlop_spiking_timesteps))
            if self.ss:
                self.classifiers = nn.ModuleList([SSLinearProj(fc_size, num_classes, bias=False)])
            else:
                self.classifiers = nn.ModuleList([LinearProj(fc_size, num_classes, bias=False)])
        else:
            if self.ss:
                self.classifiers = nn.ModuleList([SSLinear(fc_size, num_classes)])
            else:
                self.classifiers = nn.ModuleList([nn.Linear(fc_size, num_classes)])
        self.classifier_num = 1

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, Conv2dProj):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, LinearProj):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.feature_fusion =  MultimodalSpikingNN(dim)
                    
        pre_weight = "./pre_weight/scene/best_model.pth"
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

    def _make_layers(self, cfg):
        layers = []
        hlop_modules = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                if self.ss:
                    layers.append(SSConv2dProj(self.init_channels, x, kernel_size=3, padding=1, bias=False, proj_type=self.proj_type))
                else:
                    layers.append(Conv2dProj(self.init_channels, x, kernel_size=3, padding=1, bias=False, proj_type=self.proj_type))
                layers.append(nn.BatchNorm2d(x))
                if self.neuron_type == 'lif':
                    layers.append(LIFNeuron(self.snn_setting))
                elif self.neuron_type == 'if':
                    layers.append(IFNeuron(self.snn_setting))
                else:
                    raise NotImplementedError('Please use IF or LIF model.')
                hlop_modules.append(HLOP(self.init_channels*3*3, spiking=self.hlop_spiking, spiking_scale=self.hlop_spiking_scale, spiking_timesteps=self.hlop_spiking_timesteps))
                self.init_channels = x
        return nn.Sequential(*layers), nn.ModuleList(hlop_modules)

    def forward(self, v, a, task_id=None, projection=False, proj_id_list=[0], update_hlop=False, fix_subspace_id_list=None):
        # inputs = torch.cat([x[:,_,:,:,:] for _ in range(self.timesteps)], 0)
        x = self.feature_fusion(v, a)

        # x = x.repeat(1, self.timesteps, 1, 1, 1)
        # x = torch.cat([x[:,_,:,:,:] for _ in range(self.timesteps)], 0)

        x = x.unsqueeze(1)
        inputs = x.repeat(1, self.timesteps, 1, )
        # inputs = x.view(-1, 784)

        index = 0
        for m in self.features:
            if isinstance(m, Conv2dProj) or isinstance(m, LinearProj):
                if projection:
                    proj_func = self.hlop_modules[index].get_proj_func(subspace_id_list=proj_id_list)
                    x_ = m(inputs, projection=True, proj_func=proj_func)
                else:
                    x_ = m(inputs, projection=False)
                if update_hlop:
                    if self.hlop_with_wfr:
                        # update hlop by weighted firing rate
                        inputs = weight_rate_spikes(inputs, self.timesteps, self.tau, self.delta_t)
                    if isinstance(m, Conv2dProj):
                        inputs = F.unfold(inputs, m.kernel_size, dilation=m.dilation, padding=m.padding, stride=m.stride).transpose(1, 2)
                        inputs = inputs.reshape(-1, inputs.shape[2])
                    with torch.no_grad():
                        self.hlop_modules[index].forward_with_update(inputs, fix_subspace_id_list=fix_subspace_id_list)
                index += 1
                inputs = x_
            else:
                inputs = m(inputs)

        out = inputs.view(inputs.size(0), -1)
        if not self.share_classifier:
            assert task_id is not None
            out = self.classifiers[task_id](out)
        else:
            m = self.classifiers[0]
            if projection:
                proj_func = self.hlop_modules[index].get_proj_func(subspace_id_list=proj_id_list)
                out_ = m(out, projection=True, proj_func = proj_func)
            else:
                out_ = m(out, projection=False)
            if update_hlop:
                if self.hlop_with_wfr:
                    # update hlop by weighted firing rate
                    out = weight_rate_spikes(out, self.timesteps, self.tau, self.delta_t)
                with torch.no_grad():
                    self.hlop_modules[index].forward_with_update(out, fix_subspace_id_list=fix_subspace_id_list)
            out = out_

        if self.weight_avg:
            out = weight_rate_spikes(out, self.timesteps, self.tau, self.delta_t)
        else:
            out = rate_spikes(out, self.timesteps)
        return out

    def forward_features(self, x):
        inputs = torch.cat([x[:,_,:,:,:] for _ in range(self.timesteps)], 0)
        index = 0
        feature_list = []
        for m in self.features:
            if isinstance(m, Conv2dProj) or isinstance(m, LinearProj):
                x_ = m(inputs, projection=False)
                if self.hlop_with_wfr:
                    # calculate weighted firing rate
                    inputs = weight_rate_spikes(inputs, self.timesteps, self.tau, self.delta_t)
                if isinstance(m, Conv2dProj):
                    inputs = F.unfold(inputs, m.kernel_size, dilation=m.dilation, padding=m.padding, stride=m.stride).transpose(1, 2)
                    inputs = inputs.reshape(-1, inputs.shape[2])
                feature_list.append(inputs.detach().cpu())
                index += 1
                inputs = x_
            else:
                inputs = m(inputs)

        if self.share_classifier:
            inputs = self.pool(inputs)
            if self.hlop_with_wfr:
                # calculate weighted firing rate
                inputs = weight_rate_spikes(inputs, self.timesteps, self.tau, self.delta_t)
            inputs = inputs.view(inputs.size(0), -1)
            feature_list.append(inputs.detach().cpu())

        return feature_list

    def add_classifier(self, num_classes):
        self.classifier_num += 1
        if self.ss:
            self.classifiers.append(SSLinear(self.fc_size, num_classes).to(self.classifiers[0].weight.device))
        else:
            self.classifiers.append(nn.Linear(self.fc_size, num_classes).to(self.classifiers[0].weight.device))
        m = self.classifiers[-1]
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()

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
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = x+self.fc2(x)
        x = self.act(x)
        return x
    
class FClayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(FClayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        return x
    
class Nonact_FClayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(Nonact_FClayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.fc(x)
        return x
        
# 定义视觉模态网络
class VisualModality(nn.Module):
    def __init__(self, dim):
        super(VisualModality, self).__init__()
        self.conv1 = nn.Conv2d(dim[0], 12, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.fc1 = FClayer(14 * 14 * 12, dim[1])
        self.fc2 = FClayer(dim[1], dim[2])
        self.fc3 = Nonact_FClayer(dim[2], dim[3])
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(self.bn1(x))
        x = self.pool1(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # 展平
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
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)
    
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

    def forward(self, visual_input, auditory_input):
        # 视觉模态
        v = self.visual_modality(visual_input)
        # 听觉模态
        a = self.auditory_modality(auditory_input)
        # 融合
        fused = torch.cat((v, a), dim=1)  # 特征拼接
        fused = self.concat_fc(fused)
        # 注意力加权
        fused = self.attention(v, a, fused)

        return fused


class PretrainMultimodalSpikingNN(nn.Module):
    def __init__(self, dim):
        super(PretrainMultimodalSpikingNN, self).__init__()
        Visual_dim = dim["Visual"]
        Audio_dim = dim["Audio"]
        Attcross_dim = dim["Attention_cross"]
        self.visual_modality = VisualModality(Visual_dim)
        self.auditory_modality = AuditoryModality(Audio_dim)
        self.concat_fc = FClayer(Attcross_dim[0], Attcross_dim[1])
        self.attention = AttentionMechanism(Attcross_dim[1], Attcross_dim[2])  
        # num_classes = 9
        self.classifier = nn.Linear(Attcross_dim[2], 9)

    def forward(self, visual_input, auditory_input):
        # 视觉模态
        v = self.visual_modality(visual_input)
        # 听觉模态
        a = self.auditory_modality(auditory_input)
        # 融合
        fused = torch.cat((v, a), dim=1)  # 特征拼接
        fused = self.concat_fc(fused)
        # 注意力加权
        fused = self.attention(v, a, fused)
        # print(f"fused = {fused[-1,100:150]}")
        out = self.classifier(fused)
        return out

def spiking_cnn(dim, snn_setting, **kwargs):
    return CNNmultimodal(dim, snn_setting, 'A', fc_size=4*4*256, **kwargs)
