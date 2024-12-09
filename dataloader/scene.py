import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import random
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class ImgAudioDataset(Dataset):
    def __init__(self, root_dir, img_data, audio_data, labels=None, img_transform=None, audio_transform=None):
        self.root_dir = root_dir
        self.img_data = img_data
        self.audio_data = audio_data
        self.labels = labels
        self.img_transform = img_transform
        self.audio_transform = audio_transform
        
    def __len__(self):
        return len(self.img_data)
        
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root_dir, self.img_data.iloc[idx]))
        audio = self.audio_data[idx,:]
        if self.img_transform:
            img = self.img_transform(img)
        if self.audio_transform:
            audio = self.audio_transform(audio)        
                               
        return ((img, audio) if self.labels is None else (img, audio, int(self.labels[idx])))

def normalize(data):
    return (data - np.mean(data, axis=0))/np.std(data, axis=0)


class AddGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RandomCircularShift:      
    def __call__(self, tensor):
        return torch.roll(tensor, 13*np.random.randint(8),dims=0)

img_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])
# img_train_transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
# ])
img_val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

audio_train_transform = transforms.Compose([
    torch.from_numpy,
    AddGaussianNoise(0.0, 0.1),
    RandomCircularShift()
])

audio_val_transform = transforms.Compose([
    torch.from_numpy,
    AddGaussianNoise(0.0, 0.1),
    RandomCircularShift()
])

def get(data_dir='/home/haichao/tzq/HLOP-SNN/datasets/scene', seed=0, fixed_order=False):

    data = pd.read_csv(os.path.join(data_dir, 'dataset.csv'), delimiter=',', nrows=None)
    data_train = np.array(data)

    audio = data_train[:,1:-2].astype('float32') #last index of the interval isn't included in the range : CLASS1
    labels = data_train[:,-1]
    img_paths = data['IMAGE']

    classes = ["FOREST", "CLASSROOM", "CITY", "RIVER", "GROCERY-STORE","JUNGLE","BEACH","FOOTBALL-MATCH","RESTAURANT"]
    for index,class_name in enumerate(classes):
        labels = np.where(labels == class_name, index, labels)

    labels.astype('int32')
    
    ## 划分数据集
    img_train, img_temp, audio_train, audio_temp, labels_train, labels_temp = train_test_split(img_paths, 
                                                                                            audio, labels, train_size=0.8)
    img_val, img_test, audio_val, audio_test, labels_val, labels_test = train_test_split(img_temp, 
                                                                                        audio_temp, labels_temp, train_size=0.1)

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

    data = {}
    taskcla = []
    img_size=[3,224,224]
    audio_size = [104]

    multasks_dir = os.path.join(data_dir, 'scene_multasks')

    nperm = len(classes)//2 + 1  # tasks numbers 
    seeds = np.array(list(range(nperm)), dtype=int)
    if not fixed_order:
        seeds = shuffle(seeds, random_state=seed)

    if not os.path.isdir(multasks_dir):
        os.makedirs(multasks_dir)

        for n in range(nperm):
            data[n]={}
            data[n]['name']='scene'
            data[n]['ncla']= 9
            data[n]['train']={'img': [], 'audio': [],'labels':[]}
            data[n]['val']={'img': [], 'audio': [],'labels':[]}

        for s in ['train', 'val']:
            if(s == "train"):
                loader = DataLoader(train_data, batch_size=1, shuffle=True)
            elif(s == "val"):
                loader = DataLoader(val_data, batch_size=1, shuffle=True)

            for _, temp in enumerate(loader):
                img, audio, labels = temp
                n=labels.numpy()[0]
                c = n // 2
                print(f"n = 这是第{n}类")
                print(f"c = 这是第{c}个任务")
                print(f"labels = {labels}")
                # print(f"n = {n}")
                # print(f"s = {s}")
                # print(f"data = {data}")
                data[c][s]['img'].append(img) # 255 
                data[c][s]['audio'].append(audio)
                data[c][s]['labels'].append(labels)

        # "Unify" and save
        for t in data.keys():
            print(f"t = {t}")
            for s in ['train','val']:
                data[t][s]['img']=torch.stack(data[t][s]['img']).view(-1,img_size[0],img_size[1],img_size[2])
                data[t][s]['audio']=torch.stack(data[t][s]['audio']).view(-1,audio_size[0])
                data[t][s]['labels']=torch.stack(data[t][s]['labels']).view(-1,1)
                torch.save(data[t][s]['img'], os.path.join(os.path.expanduser(multasks_dir),'data'+str(t)+s+'img.bin'))
                torch.save(data[t][s]['audio'], os.path.join(os.path.expanduser(multasks_dir),'data'+str(t)+s+'audio.bin'))
                torch.save(data[t][s]['labels'], os.path.join(os.path.expanduser(multasks_dir),'data'+str(t)+s+'labels.bin'))
        print("已经成功划分子任务")
    # Load binary files
    data={}
    # ids=list(shuffle(np.arange(5),random_state=seed))
    ids=list(np.arange(nperm))
    print('Task order =',ids)
    for i in range(nperm):
        data[i] = dict.fromkeys(['name','ncla','train'])
        for s in ['train','val']:
            data[i][s]={'img': [], 'audio': [],'class':[]}
            data[i][s]['img']=torch.load(os.path.join(os.path.expanduser(multasks_dir),'data'+str(ids[i])+s+'img.bin'))
            data[i][s]['audio']=torch.load(os.path.join(os.path.expanduser(multasks_dir),'data'+str(ids[i])+s+'audio.bin'))
            data[i][s]['labels']=torch.load(os.path.join(os.path.expanduser(multasks_dir),'data'+str(ids[i])+s+'labels.bin'))
        # data[i]['ncla']=len(np.unique(data[i]['train']['labels'].numpy()))
        if data[i]['ncla']==2:
            data[i]['name']='scene-'+str(ids[i])
        else:
            data[i]['name']='scene-'+str(ids[i])
    # Validation
    #for t in data.keys():
    #    r=np.arange(data[t]['train']['x'].size(0))
    #    # r=np.array(shuffle(r,random_state=seed),dtype=int)
    #    r=np.array(r,dtype=int)
    #    nvalid=int(pc_valid*len(r))
    #    ivalid=torch.LongTensor(r[:nvalid])
    #    itrain=torch.LongTensor(r[nvalid:])
    #    data[t]['valid'] = {}
    #    data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
    #    data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
    #    data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
    #    data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

    # Others
    # n = 0
    for t in data.keys():
        taskcla.append((t, 9))
    #     n += data[t]['ncla']
    # data['ncla'] = n

    return data, taskcla, img_size, audio_size

########################################################################################################################
