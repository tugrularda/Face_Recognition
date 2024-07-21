import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    def __init__(self, in_channels, in_size):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        size = in_size/8
        size = int((size**2)*32)
        print(f'size: {size}')
        self.fc1 = nn.Linear(size, 1024)
        self.fc2 = nn.Linear(1024, 128)

    def forward_one(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        #x = x.view(x.size(0), -1)
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def forward(self, anchor, positive, negative):
        anchor_output = self.forward_one(anchor)
        positive_output = self.forward_one(positive)
        negative_output = self.forward_one(negative)
        return anchor_output, positive_output, negative_output