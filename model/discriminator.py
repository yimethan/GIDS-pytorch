import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv0 = nn.Conv2d(1, 1, kernel_size=(4, 3), stride=(2, 1), padding=1, bias=False)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(4, 3), stride=(2, 1), padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(1, 1, (16, 48), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)

        return x