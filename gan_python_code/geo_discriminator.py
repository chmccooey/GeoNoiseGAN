# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:25:17 2020

@author: ccmcc
"""

import torch
import torch.nn as nn

class DiscriminativeNet(torch.nn.Module):
    
    def __init__(self):
        super(DiscriminativeNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=100, kernel_size=4,
                stride=2, padding=7, bias=False
            ),
            nn.BatchNorm2d(100),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=100, out_channels=200, kernel_size=4,
                stride=2, padding=3, bias=False
            ),
            nn.BatchNorm2d(200),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=200, out_channels=400, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(400),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=400, out_channels=800, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(800),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(800*32*32, 1),
            #nn.Linear(8611840, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Convolutional layers
        print("DEBUG: discriminator in forward: x.shape={}".format(x.shape))
        x = self.conv1(x)
        print("DEBUG: discriminator post conv1: x.shape={}".format(x.shape))
        x = self.conv2(x)
        print("DEBUG: discriminator post conv2: x.shape={}".format(x.shape))
        x = self.conv3(x)
        print("DEBUG: discriminator post conv3: x.shape={}".format(x.shape))
        x = self.conv4(x)
        print("DEBUG: discriminator post conv4: x.shape={}".format(x.shape))
        # Flatten and apply sigmoid
        x = x.view(-1, 800*32*32)
        #x = x.view(-1, 8611840)
        x = self.out(x)
        print("DEBUG: discriminator out forward: x.shape={}".format(x.shape))
        return x
