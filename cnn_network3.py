#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 15:48:02 2022

@author: domini4
"""

import torch.nn as nn

class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(1, 64, 5, 1, 2),                              
            nn.ReLU(),                      
            nn.MaxPool2d(2),    
        )
        self.fc = nn.Linear(64 * 14 * 14, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)       
        output = self.fc(x)
        return output