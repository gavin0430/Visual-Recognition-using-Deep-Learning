#!/usr/bin/env python
# coding: utf-8

# In[1]:



import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms
import torchvision.models as models

import matplotlib.pyplot as plt
from PIL import Image

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# In[2]:


NUM_CLASSES = 13
BATCH_SIZE = 20
LEARNING_RATE = 0.001
NUM_EPOCHS = 300
DEVICE = 'cuda:1'
device = torch.device('cuda')


# In[3]:
#####Load Data###########

data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
training_data = datasets.ImageFolder(root='./train',transform=data_transform)
train_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    #print(images)
    #images = images.cuda()
    break


# In[ ]:

###Using different model

model = models.resnet50(pretrained=True, progress=True)
model.fc = nn.Linear(2048, NUM_CLASSES)
#print(torch.device('cuda'))
#model.to(device)
model = model.cuda()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# In[4]:


model = models.resnext101_32x8d(pretrained=True, progress=True)
model.fc = nn.Linear(2048, NUM_CLASSES)
print(model)
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# In[ ]:


model = models.wide_resnet101_2(pretrained=True, progress=False)
model.fc = nn.Linear(2048, NUM_CLASSES)
print(model)
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# In[5]:


#model = torch.load('./resnext101_32x8d')
def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        #features = features.to(device)
        #targets = targets.to(device)
        features = features.cuda()
        targets = targets.cuda()
        
        #logits, probas = model(features)
        logits = model(features)
        probas = F.softmax(logits, dim=1)
        
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = LEARNING_RATE * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    
best_acc = 0.93
start_time = time.time()
for epoch in range(NUM_EPOCHS):
    
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        #features = features.to(DEVICE)
        #targets = targets.to(DEVICE)
        features = features.cuda()
        targets = targets.cuda()
        
        ### FORWARD AND BACK PROP
        #logits, probas = model(features)
        logits = model(features)
        probas = F.softmax(logits, dim=1)
        #print(logits)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        adjust_learning_rate(optimizer, epoch)
        
        ### LOGGING
        if not batch_idx % 30:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                   %(epoch+1, NUM_EPOCHS, batch_idx, 
                     len(train_loader), cost))

        

    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        acc = compute_accuracy(model, train_loader, device=DEVICE)
        print('Epoch: %03d/%03d | Train: %.3f%%' % (
              epoch+1, NUM_EPOCHS, acc))
        if acc > best_acc:
            torch.save(model, './resnext101_32x8d_300')
            best_acc = acc
            print("model save at acc = ",acc)
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))


# In[8]:

###Testing##############
import csv
model = torch.load('./resnext101_32x8d_300')
testing_data = datasets.ImageFolder(root='./test1',transform=data_transform)
test_loader = torch.utils.data.DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=4)
label = ['bedroom','coast','forest','highway','insidecity','kitchen','livingroom','mountain','office','opencountry','street','suburb','tallbuilding']
#print(testing_data[1039])
k = 0
with open('output5.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id','label'])
    for i, (images, labels) in enumerate(test_loader):
        images = images.cuda()
        model.eval()
        logits = model(images)
        probas = F.softmax(logits, dim=1)
        _, predicted_labels = torch.max(probas, 1)
        #print(predicted_labels)
        image_id = 'image_' + "%04d" % k
        #print(image_id)
        print(image_id,label[predicted_labels[0].item()])
        writer.writerow([image_id,label[predicted_labels[0].item()]])
        
        #break
        k+=1

'''
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id','label'])
    for i, (images, labels) in enumerate(test_loader):
        images = images.cuda()
        model.eval()
        logits = model(images)
        probas = F.softmax(logits, dim=1)
        _, predicted_labels = torch.max(probas, 1)

        #print(predicted_labels[0].item())
        image_id = 'image_' + "%04d" % i
        print()
        for j in range(10):
            writer.writerow([image_id,label[]])
        break
''' 


# In[7]:


torch.save(model, './resnext101_32x8d_300_last')
#rr = torch.load('./resnet50_first_try')
#rr.eval()
#print(rr)

