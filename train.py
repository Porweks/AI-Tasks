import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.fc1 = nn.Linear(64*124*124, 128)
        self.fc2 = nn.Linear(128, 42)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(-1, 64 * 124 * 124)
        x = nn.functional.relu (self.fc1(x))
        x = self.fc2(x)
        return x

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
data_set =  datasets.ImageFolder("simpsons_dataset", transform=transform)

valid_count = int(len(data_set)*0.2)

train_set, valid_set = random_split(data_set,[len(data_set)-valid_count,valid_count])
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 32, shuffle = True)
validate_loader = torch.utils.data.DataLoader(valid_set, batch_size = 32, shuffle = True)

model = Net().cuda()
criterion = nn.CrossEntropyLoss().cuda()


accuracy_per_epoch = []
recall_per_epoch = []
precision_per_epoch = []


running_loss_per_epoch = []
validation_loss_per_epoch = []
validation_accuracy =[]
validation_recall = []
validation_precision = []

epochs = 40
for epoch in range(epochs):

    optimizer = torch.optim.SGD(model.parameters(), lr=0.02/(epoch+1), momentum=0,weight_decay=1e-04)
    running_loss = 0.0
    model.train()
    buff = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i,data in enumerate(train_loader, 0):
        x, label = data
        x = x.cuda()
        label=label.cuda()
        optimizer.zero_grad()
        y = model(x)
        loss = criterion(y, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        real = sum(y.argmax(dim=1)==label).item()
        TP += ((predicted == real) & (real == 1)).sum().item()
        TN += ((predicted == real) & (real == 0)).sum().item()
        FN += ((predicted != real) & (real == 1)).sum().item()
        FP += ((predicted != real) & (real == 0)).sum().item()

    accuracy_per_epoch.append((TP+TN)/(TP+TN+FP+FN))
    running_loss_per_epoch.append(running_loss/32/len(train_loader))
    recall_per_epoch.append(TP/(TP+FN))
    precision_per_epoch.append(TP/(TP+FP))
    total =0
    correct = 0
    # validation
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i,data in enumerate(validate_loader, 0):
            x, label = data
            x = x.cuda()
            label = label.cuda()
            y = model(x)
            loss = criterion(y, label)
            val_loss += loss.item()
            real = sum(y.argmax(dim=1)==label).to('cpu').item()
            TP += ((predicted == real) & (real == 1)).sum().item()
            TN += ((predicted == real) & (real == 0)).sum().item()
            FN += ((predicted != real) & (real == 1)).sum().item()
            FP += ((predicted != real) & (real == 0)).sum().item()
            _, predicted = torch.max(y,1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            
    validation_accuracy.append((TP+TN)/(TP+TN+FP+FN))
    validation_recall.append(TP/(TP+FN))
    validation_precision.append(TP/(TP+FP))
    print(f'Epoch {epoch+1}/{epochs}, Precision: {TP/(TP+FP)}, Train Loss: {running_loss/len(train_loader)}, Train Accuracy: {accuracy_per_epoch}, Val Loss: {val_loss/len(validate_loader)}, Val Accuracy: {100*correct/total}%')





