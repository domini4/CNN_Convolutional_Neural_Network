#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 13:09:35 2022

@author: domini4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
import random

from cnn_network import CNN
from cnn_network2 import CNN2
from cnn_network3 import CNN3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print(device)

loss_m1 = []
accuracy_m1 = []

loss_m2 = []
accuracy_m2 = []

loss_m3 = []
accuracy_m3 = []

model1 = CNN().to(device)
model2 = CNN2().to(device)
model3 = CNN3().to(device)

pytorch_total_params = sum(p.numel() for p in model1.parameters() if p.requires_grad)
print(pytorch_total_params)

pytorch_total_params = sum(p.numel() for p in model2.parameters() if p.requires_grad)
print(pytorch_total_params)

pytorch_total_params = sum(p.numel() for p in model3.parameters() if p.requires_grad)
print(pytorch_total_params)

loss_func = nn.CrossEntropyLoss()  

batch_size = 100

# download data
train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True           
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor(),
    download = True
)


# load data
train_load = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_load = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

num_epochs = 80
def train(num_epochs, model, train_load):
    
    loss_m = []
    accuracy_m = []
    
    optimizer = optim.Adam(model.parameters(), lr = 0.0001) 
    
    model.train()
        
    # Train the model
    total_step = len(train_load)
        
    # in each epoch, train for all images in a batch
    for epoch in range(num_epochs):
        accuracy = 0
        for i, (images, labels) in enumerate(train_load):
            
            # gives batchs
            # images: x
            # labels: y
            # load the batch into gpu if available
            x = images.to(device)   # batch images
            y = labels.to(device)   # batch labels
            
            # perform forward pass
            output = model(x)
            
            # calculate accuracy
            accuracy += (torch.max(output, 1)[1].data.squeeze() == y).sum()
            
            # determine loss
            loss = loss_func(output, y)
            
            # back propagation 
            optimizer.zero_grad()           
            loss.backward()    
            optimizer.step()                
            
        # log values at end of each epoch
        loss_m.append(loss.item())
        accuracy_m.append(float(accuracy)/float(batch_size*(i+1)))
        
        print ('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}' 
                    .format(epoch + 1, num_epochs, loss.item(), float(accuracy*100)/float(batch_size*(i+1))))
    return(loss_m, accuracy_m)

loss_m1, accuracy_m1 = train(num_epochs, model1, train_load)
loss_m2, accuracy_m2 = train(num_epochs, model2, train_load)
loss_m3, accuracy_m3 = train(num_epochs, model3, train_load)

plt.cla()
plt.plot(range(len(loss_m1)), loss_m1, 'r', label="model 1")
plt.plot(range(len(loss_m2)), loss_m2, 'g', label="model 2")
plt.plot(range(len(loss_m3)), loss_m3, 'b', label="model 3")
plt.title("MNIST CNN Training Loss")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('mnist_cnn_loss.png', dpi=500)
plt.show()

plt.plot(range(len(accuracy_m1)), accuracy_m1, 'r', label="model 1")
plt.plot(range(len(accuracy_m2)), accuracy_m2, 'g', label="model 2")
plt.plot(range(len(accuracy_m3)), accuracy_m3, 'b', label="model 3")
plt.title("MNIST CNN Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('mnist_cnn_accuracy.png', dpi=500)
plt.show()