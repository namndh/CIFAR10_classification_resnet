import os
import sys
import time
import math
import pickle

import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

class myDataSet(Dataset):
    """Custom dataset loader"""
    def __init__(self, train, root_dir, transform=None):
        assert os.path.isdir(root_dir), 'ERROR: Data directory is not found!'
        self.train = train
        self.root_dir = root_dir
        self.transform = transform
        if train:
            train_set_path = os.path.join(self.root_dir, 'train_set.b') 
            assert os.path.isfile(train_set_path), 'ERROR: Train dataset is not found!'
            with open(train_set_path, 'rb') as train_set_bin:
                self.dataset = pickle.load(train_set_bin)
        else:
            val_set_path = os.path.join(self.root_dir, 'val_set.b')
            assert os.path.isfile(val_set_path), 'ERROR: Val dataset is not found!'
            with open(val_set_path, 'rb') as val_set_bin:
                self.dataset = pickle.load(val_set_bin)
        self.images, self.labels = zip(*self.dataset)
        self.images = list(self.images)
        self.labels = list(self.labels)
                
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label