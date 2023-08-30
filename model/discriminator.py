import torch.nn as nn
import torch

device = torch.device('cuda')


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv0 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.relu0 = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(16, 1, kernel_size=(48, 1), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # batch * 1 * 64 * 48

        x = x.permute(0, 2, 3, 1)  # batch * 64 * 48 * 1

        x = self.conv0(x)
        x = self.relu0(x)

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)

        x = self.sigmoid(x)  # batch * 1 * 1 * 1

        return x.view(-1)
