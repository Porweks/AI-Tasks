import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*20*20, 128)
        self.fc2 = nn.Linear(128, 42)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 64 * 20 * 20)
        x = nn.functional.relu (self.fc1(x))
        x = self.fc2(x)
        return x

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
test_set =  datasets.ImageFolder("testset", transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 32)

model = Net().cuda()
model.load_state_dict(torch.load("/content/drive/MyDrive/model_data.pt"))
criterion = nn.CrossEntropyLoss().cuda()

accuracy = np.zeros(42)
all = np.zeros(42)
running_loss = 0.0
for i,data in enumerate(test_loader, 0):
    x, label = data
    x = x.cuda()
    all[label]+=1
    label = label.cuda()
    y = model(x)
    loss = criterion(y, label)
    running_loss += loss.item()


    buff=label.to('cpu')[(y.argmax(dim=1)==label).to('cpu').numpy()]
    accuracy[buff] += 1

print(accuracy,all)
print(sum(accuracy)/sum(all))
