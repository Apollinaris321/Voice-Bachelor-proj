import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(4, 4)

        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)

        #self.fc1 = nn.Linear(64 * 7 * 12, 1000)
        self.fc1 = nn.Linear(32 * 9 * 14, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        #x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = (self.fc2(x))
        return x


#30 epochs -> 92% ->  64 out of 110
class Net2(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(4, 4)

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        #self.conv3 = nn.Conv2d(32, 64, 5)

        # self.fc1 = nn.Linear(64 * 7 * 12, 1000)
        self.fc1 = nn.Linear(64 * 9 * 14, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = (self.fc2(x))
        return x


#30 e -> 92 -> 90 out of 110
# das ist gut
class Net3(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(4, 4)

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)

        # self.fc1 = nn.Linear(64 * 7 * 12, 1000)
        self.fc1 = nn.Linear(128 * 4 * 7, 1000)
        #self.fc1 = nn.Linear(64 * 8 * 13, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = (self.fc2(x))
        return x