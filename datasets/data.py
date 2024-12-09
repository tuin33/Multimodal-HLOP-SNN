import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import random
import os

from sklearn.model_selection import train_test_split

data = pd.read_csv('/home/haichao/tzq/HLOP-SNN/datasets/scene/dataset.csv', delimiter=',', nrows=None)
data_train = np.array(data)

audio = data_train[:,1:-2].astype('float32') #last index of the interval isn't included in the range : CLASS1
labels = data_train[:,-1]
img_paths = data['IMAGE']

classes = ["FOREST", "CLASSROOM", "CITY", "RIVER", "GROCERY-STORE","JUNGLE","BEACH","FOOTBALL-MATCH","RESTAURANT"]
for index,class_name in enumerate(classes):
    labels = np.where(labels == class_name, index, labels)

labels.astype('int32')

## 训练集 验证集 测试集
img_train, img_temp, audio_train, audio_temp, labels_train, labels_temp = train_test_split(img_paths, 
                                                                                           audio, labels, train_size=0.6)
img_val, img_test, audio_val, audio_test, labels_val, labels_test = train_test_split(img_temp, 
                                                                                     audio_temp, labels_temp, train_size=0.5)


from torchvision import transforms

# DA : Data Augmentation
# DP : Data Preparation --> transform data to a more ergonomic data format

img_train_transform = transforms.Compose([ #Compose is used to chain multiple transforms to create a transformation pipeline
    transforms.RandomResizedCrop(224), #DA
    transforms.RandomHorizontalFlip(), #DA
    transforms.ToTensor(), #DP to compute on GPU
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) #DP
])
img_val_transform = transforms.Compose([
    transforms.Resize(256), #DA fixed resize and crop for reliability
    transforms.CenterCrop(224),# DA
    transforms.ToTensor(), #DP
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) #DP
])

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)

from torch.utils.data import Dataset
from PIL import Image

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
                               
        return ((img, audio) if labels is None else (img, audio, int(self.labels[idx])))
    

from torch.utils.data import Dataset, DataLoader

train_data = ImgAudioDataset(root_dir='/kaggle/input/scene-classification-images-and-audio', 
                             img_data=img_train, audio_data=audio_train, labels=labels_train, 
                             img_transform=img_train_transform)
val_data = ImgAudioDataset(root_dir='/kaggle/input/scene-classification-images-and-audio/', 
                           img_data=img_val,  audio_data=audio_val,labels=labels_val, 
                           img_transform=img_val_transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

