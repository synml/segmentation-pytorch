import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
import torchvision
import torchsummary


class ModifiedAlignedXception(nn.Module):
    def __init__(self, output_stride, pretrained):
        super(ModifiedAlignedXception, self).__init__()
        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError

        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.block1 = Block(64, 128, reps=2, stride=2, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, start_with_relu=False, grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride,
                            start_with_relu=True, grow_first=True, is_last=True)


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1,
                 start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, )


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 0, dilation,
                              groups=in_channels, bias=bias)
        self.bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x
