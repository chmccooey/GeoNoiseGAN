# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:25:11 2020

@author: ccmcc
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

# DEBUG: generator in forward: x.shape=torch.Size([10, 100])
# DEBUG: generator before view forward: x.shape=torch.Size([10, 1048576])
# DEBUG: generator after view forward: x.shape=torch.Size([10, 1024, 32, 32])
# DEBUG: generator post conv1: x.shape=torch.Size([10, 512, 64, 64])
# DEBUG: generator post conv2: x.shape=torch.Size([10, 256, 128, 128])
# DEBUG: generator post conv3: x.shape=torch.Size([10, 1, 256, 256])
# DEBUG: generator in forward: x.shape=torch.Size([10, 100])
# DEBUG: generator before view forward: x.shape=torch.Size([10, 1048576])
# DEBUG: generator after view forward: x.shape=torch.Size([10, 1024, 32, 32])
# DEBUG: generator post conv1: x.shape=torch.Size([10, 512, 64, 64])
# DEBUG: generator post conv2: x.shape=torch.Size([10, 256, 128, 128])
# DEBUG: generator post conv3: x.shape=torch.Size([10, 1, 256, 256])
# DEBUG: generator in forward: x.shape=torch.Size([16, 100])
# DEBUG: generator before view forward: x.shape=torch.Size([16, 1048576])
# DEBUG: generator after view forward: x.shape=torch.Size([16, 1024, 32, 32])
# DEBUG: generator post conv1: x.shape=torch.Size([16, 512, 64, 64])
# DEBUG: generator post conv2: x.shape=torch.Size([16, 256, 128, 128])
# DEBUG: generator post conv3: x.shape=torch.Size([16, 1, 256, 256])

class GenerativeNet(torch.nn.Module):
    
    def __init__(self):
        super(GenerativeNet, self).__init__()
        
        self.linear = torch.nn.Linear(100, 1600*5*5)
        
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1600, out_channels=800, kernel_size=10,
                stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(800),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=800, out_channels=400, kernel_size=10,
                stride=2, padding=0, bias=False
            ),
            nn.BatchNorm2d(400),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=400, out_channels=200, kernel_size=4,
                stride=2, padding=0, bias=False
            ),
            nn.BatchNorm2d(200),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=200, out_channels=1, kernel_size=4,
                stride=2, padding=0, bias=False
            )
        )
        self.out = torch.nn.Tanh()

    def forward(self, x):
        # Project and reshape
        print("DEBUG: generator in forward: x.shape={}".format(x.shape))
        x = self.linear(x)
        print("DEBUG: generator before view forward: x.shape={}".format(x.shape))
        #x = x.view(x.shape[0], 1024, 4, 4) # 1024 = 64 * 16
        x = x.view(x.shape[0], 1600, 5, 5) # 1024 = 64 * 16
        print("DEBUG: generator after view forward: x.shape={}".format(x.shape))
        
        # Convolutional layers
        x = self.conv1(x)
        print("DEBUG: generator post conv1: x.shape={}".format(x.shape))
        x = self.conv2(x)
        print("DEBUG: generator post conv2: x.shape={}".format(x.shape))
        x = self.conv3(x)
        print("DEBUG: generator post conv3: x.shape={}".format(x.shape))
        x = self.conv4(x)
        print("DEBUG: generator post conv4: x.shape={}".format(x.shape))
        # Apply Tanh
        print("DEBUG: generator out forward: x.shape={}".format(x.shape))
        return self.out(x)
    
# Noise
def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda()
    return n