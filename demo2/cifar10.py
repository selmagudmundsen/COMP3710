
"""
Classify the CIFAR10 dataset using ResNet
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Hyperparameters
num_epochs = 35
learning_rate = 0.1
channels = 10

# Path
model_name = "resnet"

# Data
transform_train = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # these two lines will improve the accuracy of the network
        # these last two lines will increase the training data by flipping and cropping the images
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

#Downloading the CIFAR10 dataset into train and test sets
trainset = torchvision.datasets.CIFAR10(
    root="./CIFAR10/train", train=True,
    transform=transform_train,
    download=True)
    
testset = torchvision.datasets.CIFAR10(
    root="./CIFAR10/test", train=False,
    transform=transform_test,
    download=True)


train_loader = DataLoader(trainset, batch_size=128, shuffle=True)

test_loader = DataLoader(testset, batch_size=100, shuffle=False)