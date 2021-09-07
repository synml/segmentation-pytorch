import torch.nn as nn


class SeparableConv2d(nn.Sequential):
    """ When using channel attention, use activation also """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, dilation=1,
                 activation: nn.Module = None, channel_attention: nn.Module = None):
        modules = [
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            # activation
            # channel_attention
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        if activation is not None:
            modules.insert(2, activation)
        if channel_attention is not None:
            modules.insert(3, channel_attention)
        super(SeparableConv2d, self).__init__(*modules)
