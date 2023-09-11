import os
import torch
import numpy as np
import torchvision
from matplotlib import pyplot as plt

import os
from PIL import Image


class BlenderDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        # root_dir: 图片文件夹的路径
        # labels_file: 标签文件的路径，每一行是图片文件名和标签之间用空格隔开
        # transform: 可选的图像变换操作
        self.root_dir = root_dir
        self.transform = transform
        # get the list of image files and annotation files
        self.image_names = sorted(os.listdir(os.path.join(root_dir)))
        for i in range(4):
            self.image_names = self.image_names + self.image_names
        print(self.image_names)
        self.labels = []
        for image_name in self.image_names:
            # print(image_name[:-6])
            self.labels.append(image_name[:-6])
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.image_names[idx]
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]
        label = int(label)
        # ret_label = torch.zeros(360)
        # ret_label[label] = 1
        return image, label

class NeRFDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        # root_dir: 图片文件夹的路径
        # labels_file: 标签文件的路径，每一行是图片文件名和标签之间用空格隔开
        # transform: 可选的图像变换操作
        self.root_dir = root_dir
        self.transform = transform
        # get the list of image files and annotation files
        self.image_names = []
        for folder in os.listdir(root_dir):
            names = os.listdir(os.path.join(root_dir, folder))
            for i in range(len(names)):
                names[i] = os.path.join(folder, names[i])
            self.image_names += names
        print(len(self.image_names))
        self.labels = []
        for image_name in self.image_names:
            label = image_name[:-4].split('_')[1]
            self.labels.append(label)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.image_names[idx]
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]
        label = int(label)
        return image, label
