"""
Classify the CIFAR10 dataset using ResNet
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import ResNet
import torchvision.transforms as transforms
import torch.nn.functional as F

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
path = "/Selma/dev/COMP3710/"

# Data
transform_train = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # tranforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    ]
)

tranform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root=path + "cifar10", train=True, transform=transform_train
)
train_loader = torch.utils.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root=path + "cifar10", train=False, transform=transform_train
)
train_loader = torch.utils.DataLoader(testset, batch_size=100, shuffle=False)

