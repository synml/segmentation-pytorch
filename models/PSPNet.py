import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import models.backbone
from models.GCMNet import Global_ContextBlock


class PSPNet(nn.Module):
    def __init__(self, backbone: str, output_stride: int, num_classes: int, mode: bool):
        super(PSPNet, self).__init__()

        # Backbone
        if backbone == 'ResNet101':
            self.backbone = models.backbone.resnet101.ResNet101(output_stride)

        elif backbone == 'Xception':
            self.backbone = models.backbone.xception.xception(output_stride, pretrained=True)
        else:
            raise NotImplementedError('Wrong backbone.')
        self.decoder = Decoder(num_classes, mode=mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size()[2:]
        x = self.backbone(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x


class PPM(nn.Module):
    def __init__(self, in_channels: int):
        super(PPM, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        self.conv_1 = nn.Conv2d(in_channels, 256, 1, 1, 1 // 2, bias=False)
        self.conv_2 = nn.Conv2d(in_channels, 1024, 1, 1, 1 // 2, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.act = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_size = self.upsample(x)
        x_size1 = x_size.size()
        x_size2 = self.conv_1(x_size)
        x8 = self.conv_2(x_size)

        x1 = self.pool1(x_size2)
        x1 = self.conv1(x1)
        x1 = self.bn(x1)
        x1 = self.act(x1)
        x1 = F.interpolate(x1, x_size1[2:], mode='bilinear', align_corners=False)

        x2 = self.pool2(x_size2)
        x2 = self.conv2(x2)
        x2 = self.bn(x2)
        x2 = self.act(x2)
        x2 = F.interpolate(x2, x_size1[2:], mode='bilinear', align_corners=False)

        x3 = self.pool3(x_size2)
        x3 = self.conv3(x3)
        x3 = self.bn(x3)
        x3 = self.act(x3)
        x3 = F.interpolate(x3, x_size1[2:], mode='bilinear', align_corners=False)

        x4 = self.pool4(x_size2)
        x4 = self.conv4(x4)
        x4 = self.bn(x4)
        x4 = self.act(x4)
        x4 = F.interpolate(x4, x_size1[2:], mode='bilinear', align_corners=False)

        x5 = torch.cat([x1, x2, x3, x4], dim=1)
        x6 = torch.cat([x5, x8], dim=1)
        x6 = self.conv_2(x6)
        return x6


class Decoder(nn.Module):
    def __init__(self, num_classes: int, mode=True):
        super(Decoder, self).__init__()
        self.mode = mode
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        if mode == True:
            self.gcm = Global_ContextBlock(2048, [256, 256, 256], 256, [9, 7, 6])
            self.block1 = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Conv2d(256, num_classes, kernel_size=1)
            )
        else:
            self.ppm = PPM(2048)
            self.block = nn.Sequential(
                nn.Conv2d(1024, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Conv2d(256, num_classes, kernel_size=1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == True:
            x = self.gcm(x)
            x = self.upsample1(x)
            x = self.block1(x)
        else:
            x = self.ppm(x)
            x = self.upsample1(x)
            x = self.block(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PSPNet('ResNet101', output_stride=16, num_classes=19, mode=True).to(device)
    models.test.test_model(model, (1, 3, 512, 1024), '../runs', device)
