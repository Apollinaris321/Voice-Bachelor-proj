from torch import nn
from torchsummary import summary


class CNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten = nn.Flatten()
        # this is the shape of data outputted from the last conv layer
        # 10 is the amount of classes
        # 128 kann man ausrechnen anhand der vorherigen layer
        #self.linear = nn.Linear(128 * 5 * 4, 10)
        #self.linear = nn.Linear(128 * 9 * 4, 10)
        self.linear = nn.Linear(64 * 32 * 24, 4)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        return logits




if __name__ == "__main__":
    batch_size = 10
    number_of_labels = 4

    cnn = CNNNetwork()
    summary(cnn.cuda(), (1, 227, 227))
    summary(cnn.cuda(), (1, 496, 369))
    summary(cnn.cuda(), (1, 369, 496))
    # expected 64, 44
    #mel shape: 128, 44
    #summary(cnn, (1, 128, 88))
    #ihc brauch 496 x 369 pixel
    #formula cnn :