import torch.nn as nn
import torch

device = torch.device('cuda')

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()   # batch * 1 * 48 * 64

        self.conv0 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.relu0 = nn.ReLU(inplace=True)  # batch * 1 * 48 * 32

        self.conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)  # batch * 1 * 48 * 16

        self.conv2 = nn.Conv2d(16, 1, kernel_size=(3, 48), bias=False)
        self.sigmoid = nn.Sigmoid()  # batch * 1 * 1 * 1
        # 1 = 48 - k + 2p) / s + 1, 48 = k

    def forward(self, x):

        x = x.permute(2, 1, 0)  # batch * 1 * 48 * 64

        x = self.conv0(x)
        x = self.relu0(x)

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)

        x = self.sigmoid(x)

        return x.view(-1)
