'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='/data/shared/shanbhag/sakr2/cifar_granular/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/data/shared/shanbhag/sakr2/cifar_granular/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#load numpy files
def load_param():
    model_dictionary = {}
    folder_name = 'extracted_params/'
    name = 'conv1.'
    model_dictionary[name+'weight'] = torch.from_numpy(np.load(folder_name+name+'weight.npy')).cuda().requires_grad_(True)
    model_dictionary[name+'bias']= torch.from_numpy(np.load(folder_name+name+'bias.npy')).cuda().requires_grad_(False)
    print('done with '+name)

    for l in ['1','2','3','4']:
        for s in ['0','1']:
            for c in ['1','2']:
                name = 'layer'+l+'.'+s+'.conv'+c
                model_dictionary[name+'.weight']=torch.from_numpy(np.load(folder_name+name+'.weight.npy')).cuda().requires_grad_(True)
                model_dictionary[name+'.bias']=torch.from_numpy(np.load(folder_name+name+'.bias.npy')).cuda().requires_grad_(False)
                print('done with '+name)
            if (l!='1') and (s=='0'):
                name = 'layer'+l+'.'+s+'.shortcut'
                model_dictionary[name+'.weight']=torch.from_numpy(np.load(folder_name+name+'.weight.npy')).cuda().requires_grad_(True)
                model_dictionary[name+'.bias']=torch.from_numpy(np.load(folder_name+name+'.bias.npy')).cuda().requires_grad_(False)
                print('done with '+name)
    name = 'linear.'
    model_dictionary[name+'weight'] = torch.from_numpy(np.load(folder_name+name+'weight.npy')).cuda().requires_grad_(True)
    #print(model_dictionary[name+'weight'].size())
    model_dictionary[name+'bias']= torch.from_numpy(np.load(folder_name+name+'bias.npy')).cuda().requires_grad_(False)
    print('done with '+name)
    return model_dictionary

def feedforward(x,model_dictionary):
    activation_dictionary = {}
    activation_dictionary['input.activation'] = x.requires_grad_(True)
    x.retain_grad()
    block_input = F.hardtanh(F.conv2d(x,model_dictionary['conv1.weight'],model_dictionary['conv1.bias'],padding=1),0,2).requires_grad_(True)
    block_input.retain_grad()
    activation_dictionary['conv1.activation'] = block_input
    for l in ['1','2','3','4']:
        for s in ['0','1']:
            name = 'layer'+l+'.'+s
            stride = 2 if ((l!='1') and (s=='0')) else 1
            intermediate_activation = F.hardtanh(F.conv2d(block_input,model_dictionary[name+'.conv1.weight'],model_dictionary[name+'.conv1.bias'],stride=stride,padding=1) ,0,2).requires_grad_(True)
            intermediate_activation.retain_grad()
            activation_dictionary[name+'.inside.activation'] = intermediate_activation
            block_output = F.conv2d(intermediate_activation,model_dictionary[name+'.conv2.weight'],model_dictionary[name+'.conv2.bias'],padding=1)
            shortcut = F.conv2d(block_input,model_dictionary[name+'.shortcut.weight'],model_dictionary[name+'.shortcut.bias'],stride=stride) if ((l!='1') and (s=='0')) else block_input
            block_input = F.hardtanh(block_output+shortcut,0,2).requires_grad_(True)
            block_input.retain_grad()
            activation_dictionary[name+'.outside.activation'] = block_input
    linear_input = F.avg_pool2d(block_input,4)
    linear_input = linear_input.view(linear_input.size(0),-1)
    Z = torch.matmul(model_dictionary['linear.weight'],linear_input.transpose(0,1))+model_dictionary['linear.bias'][:,None]
    #result is 10x BS
    return Z.sum(1), activation_dictionary

def compute_gains():
    model_dictionary = load_param()
    activation_gains = {}
    weight_gains = {}
    first_time=True
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        Z, activation_dictionary = feedforward(inputs,model_dictionary)
        Z_fl,Y_fl = Z.max(0)
        for i in range(Z.size(0)):
            if i!=Y_fl:
                #print(i)
                #print(Y_fl)
                output_difference = Z_fl-Z[i]
                output_difference.backward(retain_graph=True)
                with torch.no_grad():
                    denominator = 24*(output_difference**2)
                    for name in model_dictionary:
                        if 'weight' in name:
                            #print(name)
                            grad = model_dictionary[name].grad
                            #print(grad)
                            if first_time:
                                weight_gains[name] = (grad**2).sum()/denominator
                            else:
                                weight_gains[name].add_((grad**2).sum()/denominator)
                            model_dictionary[name].grad.zero_()
                    for name in activation_dictionary:
                        #print(name)
                        grad = activation_dictionary[name].grad
                        #print(grad)
                        if first_time:
                            activation_gains[name] = (grad**2).sum()/denominator
                        else:
                            activation_gains[name].add_((grad**2).sum()/denominator)
                        activation_dictionary[name].grad.zero_()
                    first_time=False
        progress_bar(batch_idx,len(trainloader))
    return activation_gains, weight_gains, len(trainloader)

activation_gains, weight_gains, data_size = compute_gains()
folder_name='gain_dump/'
for name in activation_gains:
    np.save(folder_name+name+'.npy',activation_gains[name].cpu().numpy()/data_size)
    print('done with '+name)
    print(np.load(folder_name+name+'.npy'))
for name in weight_gains:
    np.save(folder_name+name+'.npy',weight_gains[name].cpu().numpy()/data_size)
    print('done with '+name)
    print(np.load(folder_name+name+'.npy'))

