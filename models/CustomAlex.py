from torch import nn
from torchsummary import summary
import torch


class CustomAlexNet(nn.Module):

    def __init__(self, num_classes=4):
        super(CustomAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=4),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64*14*14,out_features=1024),
            nn.ReLU(inplace=True),
            # 10 -> 10 classes
            nn.Linear(in_features=1024, out_features=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    batch_size = 10
    number_of_labels = 4

    cnn = CustomAlexNet(10)
    # summary(cnn.cuda(), (1, 128, 44))
    summary(cnn.cuda(), (1, 227, 227))
    #summary(cnn.cuda(), (1, 496, 369))
