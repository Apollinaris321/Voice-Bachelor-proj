import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# Define a convolution neural network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24 * 29 * 21, 3)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output = output.view(-1, 24 * 29 * 21)
        output = self.fc1(output)

        return output


if __name__ == "__main__":
    batch_size = 10
    number_of_labels = 10

    cnn = Network()
    # summary(cnn.cuda(), (1, 128, 44))
    summary(cnn.cuda(), (1, 496, 369))
    # expected 64, 44
    # mel shape: 128, 44
    # summary(cnn, (1, 128, 88))
    # ihc brauch 496 x 369 pixel
    # formula cnn :
