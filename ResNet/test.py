import os
import torch 
import torch.nn as NN 
from model import ResNet18 

CHECKPOINT_PATH = './checkpoint/ckpt.t7'

checkpoint = torch.load(CHECKPOINT_PATH) 
print(checkpoint['epoch'])