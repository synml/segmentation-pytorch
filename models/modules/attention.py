import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """ Squeeze and Excitation"""
    def __init__(self, in_channels: int, reduction_ratio=4, activation: nn.Module = None, multiplication=True):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1)
        if activation is not None:
            self.activation = activation
        else:
            self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.multiplication = multiplication

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ca = self.gap(x)
        ca = self.conv1(ca)
        ca = self.activation(ca)
        ca = self.conv2(ca)
        ca = self.sigmoid(ca)
        if self.multiplication:
            return x * ca
        else:
            return ca


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int, dilation=1, multiplication=True):
        super(SpatialAttention, self).__init__()
        if dilation > 1:
            padding = dilation
        else:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, 1, padding, dilation, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
        self.multiplication = multiplication

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa = torch.cat((max_pool, avg_pool), dim=1)
        sa = self.conv(sa)
        sa = self.bn(sa)
        sa = self.sigmoid(sa)
        if self.multiplication:
            return x * sa
        else:
            return sa
