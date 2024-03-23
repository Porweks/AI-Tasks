import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from sklearn import metrics
import statistics

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



transform = transforms.Compose([transforms.Resize((128, 128)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), transforms.ToTensor()])
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
buff = 0
for epoch in range(epochs):
    labels = []
    reals = []
    labels_val = []
    reals_val = []
    precision_per_batch = []
    val_accuracy_per_batch = []

    optimizer = torch.optim.SGD(model.parameters(), lr=0.02/(epoch+1), momentum=0.4,weight_decay=1e-03)
    running_loss = 0.0
    model.train()
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
        labels.append(label.cpu().numpy())
        reals.append((y.argmax(dim=1)).cpu().numpy())
        precision = metrics.precision_score(reals[i], labels[i], average='micro')
        precision_per_batch.append(precision)

    total =0
    correct = 0
    # validation
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
            labels_val.append(label.cpu().numpy())
            reals_val.append((y.argmax(dim=1)).cpu().numpy())
            val_accuracy = metrics.accuracy_score(reals_val[i], labels_val[i])
            val_accuracy_per_batch.append(val_accuracy)

    precision = statistics.mean(precision_per_batch)
    val_accuracy = statistics.mean(val_accuracy_per_batch)
    print(f'Epoch {epoch+1}/{epochs}, Precision: {precision}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(validate_loader)}, Val Accuracy: {val_accuracy}%')





