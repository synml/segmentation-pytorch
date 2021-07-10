import torch.nn as nn


class SeparableConv2d(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, dilation=1,
                 bias=True, activation: nn.Module = None):
        modules = [
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, in_channels, bias),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels)
        ]
        if activation is not None:
            modules.insert(2, activation)
        super(SeparableConv2d, self).__init__(*modules)
