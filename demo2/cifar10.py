
"""
Classify the CIFAR10 dataset using ResNet
"""
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
num_epochs = 10
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


# -----------
# Model
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != (self.expansion*planes):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    # ties all the blocks together (almost identical to pytorch)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        # flattens it
        out = out.view(out.size(0), -1)
        # turns it into a linear layer
        out = self.linear(out)
        return out
    
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


model = ResNet18()
model = model.to(device)

# model info
print("Model No. of Parameters: ", sum([param.nelement() for param in model.parameters()]))
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

# learning rate schedular (piecewise linear schedule)
total_step = len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_step, epochs=num_epochs)

#----------------
# Train the model
model.train()
print("> Training")
start = time.time() # time generation
for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}"
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    scheduler.step()

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in training")

#----------------
# Test the model
print("> Testing")
start = time.time() # time generation
model.eval()
while torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test accuracy: {} %'.format(100 * correct/total))

end = time.time()
elapsed = end - start
print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in training")

print("END")

