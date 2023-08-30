import torch.nn as nn

from config import Config

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.deconv0 = nn.ConvTranspose2d(256, 512, kernel_size=(4, 3), bias=False)
        self.relu0 = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=False)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False)
        self.relu2 = nn.ReLU(inplace=True)

        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False)
        self.relu3 = nn.ReLU(inplace=True)

        self.deconv4 = nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # batch * 256 * 1 * 1

        x = self.deconv0(x)  # batch * 512 * 4 * 3
        x = self.relu0(x)

        x = self.deconv1(x)  # batch * 256 * 8 * 6
        x = self.relu1(x)

        x = self.deconv2(x)  # batch * 128 * 16 * 12
        x = self.relu2(x)

        x = self.deconv3(x)  # batch * 64 * 32 * 24
        x = self.relu3(x)

        x = self.deconv4(x)  # batch * 1 * 64 * 48
        x = self.tanh(x)

        return x
