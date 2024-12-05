import torch
from torch import nn
from torch.nn import functional as F

class Residual2d1(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block2d1(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual2d1(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual2d1(num_channels, num_channels))
    return blk


class Residual1d1(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv1d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block1d1(input_channels, num_channels, num_residuals,
                    first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual1d1(input_channels, num_channels,
                                   use_1x1conv=True, strides=2))
        else:
            blk.append(Residual1d1(num_channels, num_channels))
    return blk

class Residual1d2(nn.Module):
    def __init__(self, input_channels, inter_channels, num_channels,
                 use_1x1conv=False, strides=1, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, inter_channels,
                               kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(inter_channels, inter_channels,
                               kernel_size=kernel_size, padding=kernel_size//2, stride=strides)
        self.conv3 = nn.Conv1d(inter_channels, num_channels,
                               kernel_size=1, padding=0)
        if use_1x1conv:
            self.conv4 = nn.Conv1d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm1d(inter_channels)
        self.bn2 = nn.BatchNorm1d(inter_channels)
        self.bn3 = nn.BatchNorm1d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.conv4(X)
        Y += X
        return F.relu(Y)


def resnet_block1d2(input_channels, inter_channels, num_channels, num_residuals,
                 first_block=False, kernel_size=3):
    blk = []
    for i in range(num_residuals):
        if first_block and i==0:
            blk.append(Residual1d2(input_channels, input_channels, num_channels,
                                use_1x1conv=True, strides=1, kernel_size=kernel_size))
        elif i == 0:
            blk.append(Residual1d2(input_channels, inter_channels, num_channels,
                                use_1x1conv=True, strides=2, kernel_size=kernel_size))
        else:
            blk.append(Residual1d2(num_channels, inter_channels, num_channels, kernel_size=kernel_size))
    return blk


class Residual1d3(nn.Module):
    def __init__(self, input_channels, num_channels, strides=1):
        super().__init__()
        self.use_1x1conv = input_channels != num_channels
        self.conv1 = nn.Conv1d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv1d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if self.use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, num_channels,
                                   kernel_size=1, stride=1)
            self.conv4 = nn.Conv1d(input_channels, num_channels,
                                   kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.use_1x1conv:
            res1 = self.conv3(X[:, :, ::2])
            res2 = self.conv4(X[:, :, 1::2])
            X = res1+res2
        else:
            X = X[:, :, ::2] + X[:, :, 1::2]
        Y += X
        return F.relu(Y)


def resnet_block1d3(input_channels, num_channels, num_residuals,
                    first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual1d3(input_channels, num_channels, strides=2))
        else:
            blk.append(Residual1d3(num_channels, num_channels))
    return blk



class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        # 2*4096
        b1 = nn.Sequential(nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3),            # 64*2048
                           nn.BatchNorm1d(64), nn.ReLU(),                                                       # 64*2048
                           nn.MaxPool1d(kernel_size=3, stride=2, padding=1))                                    # 64*1024
        b2 = nn.Sequential(*resnet_block(64, 64, 256, 3, first_block=True)) # 256*1024
        b3 = nn.Sequential(*resnet_block(256, 128, 512, 4))         # 512*512
        b4 = nn.Sequential(*resnet_block(512, 256, 1024, 2))        # 1024*256
        b5 = nn.Sequential(*resnet_block(1024, 512, 2048, 2))       # 2048*128

        self.net = nn.Sequential(b1, b2, b3, b4, b5,
                                    nn.AdaptiveAvgPool1d(1),
                                    nn.Flatten()
                                 )
        self.fc = nn.Linear(2048, 53)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = self.fc(x)
        return x


