import torch 
import torch.nn as nn
import torch.nn.functional as F 


class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,12,3)
        self.conv2 = nn.Conv2d(12,16,3)
        self.conv3 = nn.Conv2d(16,24,3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(24*3*3, 200)
        self.fc2 = nn.Linear(200, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # print(len(x))
        x = x.view(-1, 24*3*3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    