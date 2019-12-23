import os
import json
import random
import cv2
import numpy as np 
import torchvision.transforms as transforms 
from torch.utils.data import Dataset, DataLoader

'''
数据加载，套用torch的DataLoader
'''

class Data(Dataset):
    def __init__(self, data_root, category, input_size, transform, interval=10, landmarks=False, mode='train'):
        super(Data, self).__init__()
        self.data_root = data_root 
        self.category = category 
        self.input_size = input_size
        self.interval = interval
        self.landmarks = landmarks
        self.transform = transform
        self.mode = mode 

        self.imgs, self.labels = self.make_dataset()
    
    def __getitem__(self, item):
        img_path = self.imgs[item]
        label = self.labels[item]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        if self.landmarks:
            pass 
        else:
            return self.transform(img), label
    
    def __len__(self):
        return len(self.imgs)
    
    def make_dataset(self):
        imgs = []
        labels = []
        dataset_json = os.path.join(self.data_root, "dataset.json")
        with open(dataset_json, 'r') as fp:
            dataset = json.load(fp)
            for video, info in dataset.items():
                if info['set'] != self.mode:
                    continue

                if info['label'] == 'real':
                    label = 0
                else:
                    label = 1
                video_frame_path = os.path.join(self.data_root, 'heads', self.category, video[:-4])
                if os.path.exists(video_frame_path):
                    for img in sorted(os.listdir(video_frame_path)):
                        num = int(img[:-4])
                        if num % self.interval == 0:
                            imgs.append(os.path.join(video_frame_path, img))
                            labels.append(label)
        return imgs, labels 

def get_data_loader(data_root, category, batch_size, num_workers, input_size, interval=10, landmarks=False, mode='train'):
    if mode == 'train':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    dataset = Data(data_root, category, input_size, transform, interval, landmarks, mode)
    print("total dataset:", len(dataset))

    shuffle = False
    if mode == 'train':
        shuffle = True
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return data_loader 


 