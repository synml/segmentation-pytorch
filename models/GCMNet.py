from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import models.backbone
import utils


class GCMNet(nn.Module):
    def __init__(self, backbone: str, output_stride: int, num_classes: int):
        super(GCMNet, self).__init__()
        self.low_level_feature = []

        # Backbone
        if backbone == 'ResNet101':
            self.backbone = models.backbone.resnet101.ResNet101(output_stride)
            self.backbone.layer1.register_forward_hook(utils.hooks.get_feature_maps(self.low_level_feature))
        elif backbone == 'Xception':
            self.backbone = models.backbone.xception.xception(output_stride, pretrained=True)
            self.backbone.block2.sepconv2.register_forward_hook(utils.hooks.get_feature_maps(self.low_level_feature))
        else:
            raise NotImplementedError('Wrong backbone.')

        self.global_contextblack = Global_ContextBlock(2048, [256, 256, 256], 256, [24, 12, 6])

        # Decoder
        self.decoder = Decoder(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        size = x.size()[2:]
        x = self.backbone(x)
        x = self.global_contextblack(x)
        x = self.decoder(x, self.low_level_feature)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


class Global_ContextBlock(nn.Module):
    def __init__(self, in_channels: int, ctb_channels: list, out_channels: int, rate: list):
        super(Global_ContextBlock, self).__init__()
        self.tf1 = nn.Conv2d(in_channels, 3 * out_channels, 1, 1, 1//2, bias=False)
        self.tf2 = nn.Conv2d(in_channels, out_channels, 1, 1, padding=1//2, bias=False)
        out_ch = ctb_channels[0] + ctb_channels[1] + ctb_channels[2]
        self.down_sample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.contextblock1 = ContextBlock(out_channels, ctb_channels[0], rate[0])
        self.contextblock2 = ContextBlock(out_channels, ctb_channels[1], rate[1])
        self.contextblock3 = ContextBlock(out_channels, ctb_channels[2], rate[2])
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.Conv = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.tf3 = nn.Conv2d(out_ch, out_channels, 1, 1, 1//2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x1 = self.tf1(x)
        x2 = self.tf2(x)

        ctb1 = self.contextblock1(x2)
        ctb1 = self.GAP(ctb1)

        x2 = self.down_sample(x2)
        ctb2 = self.contextblock2(x2)
        ctb2 = self.GAP(ctb2)

        x2 = self.down_sample(x2)
        ctb3 = self.contextblock3(x2)
        ctb3 = self.GAP(ctb3)

        x2 = torch.cat([ctb1, ctb2, ctb3], dim=1)
        x2 = torch.mul(x1, x2)
        x2 = torch.add(x1, x2)

        x2 = self.Conv(x2)
        x2 = self.Conv(x2)
        x2 = self.tf3(x2)
        return x2


class ContextBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rate: int):
        super(ContextBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=rate, groups=in_channels, dilation=rate, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_classes: int):
        super(Decoder, self).__init__()

        self.compress_low_level_feature = self.make_compressor(256, 48)
        self.decode1 = self.make_decoder(256 + 48, 256)
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, low_level_feature: List[torch.Tensor]) -> torch.Tensor:
        low_level_feature = self.compress_low_level_feature(low_level_feature.pop())

        x = F.interpolate(x, size=low_level_feature.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, low_level_feature), dim=1)
        x = self.decode1(x)
        x = self.classifier(x)
        return x

    def make_compressor(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def make_decoder(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCMNet('Xception', output_stride=16, num_classes=19).to(device)
    models.test.test_model(model, (1, 3, 512, 1024), '../runs', device)  # base: 324.788977664, params: 46984611



