from torch import nn
import torch


class SimpleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(128 * 87, 10000)  # Fully connected layer with 100 hidden neurons
        self.fc2 = nn.Linear(10000, 7000)  # Fully connected layer with 100 hidden neurons
        self.fc3 = nn.Linear(7000, 3000)  # Fully connected layer with 100 hidden neurons
        self.fc4 = nn.Linear(3000, num_classes)  # Fully connected layer with num_classes outputs

    def forward(self, x):
        x = x.view(-1, 128 * 87)  # reshape the input tensor
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x