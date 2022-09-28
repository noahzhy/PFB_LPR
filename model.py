import os
import random

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# 3x3 depthwise separable convolution
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class TinyLPR(nn.Module):
    def __init__(self):
        super(TinyLPR, self).__init__()
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    model = TinyLPR()
    img = torch.randn(1, 3, 24, 94)
    label = torch.randint(0, 10, (1, 7))
    print(label)
    output = model(img, label)
    print(model)
