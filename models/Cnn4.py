from torch import nn
from torchsummary import summary
import torch


class Cnn4(nn.Module):

    def __init__(self, num_classes=4):
        super(Cnn4, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64*9*9,out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024,out_features=512),
            nn.ReLU(inplace=True),
            # 10 -> 10 classes
            nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    batch_size = 10
    number_of_labels = 4

    cnn = Cnn4(10)
    summary(cnn.cuda(), (1, 128, 128))
    #summary(cnn.cuda(), (1, 227, 227))
    #summary(cnn.cuda(), (1, 496, 369))
