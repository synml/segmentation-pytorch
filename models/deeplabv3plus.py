from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import models
import models.backbone
import utils


class DeepLabv3plus(nn.Module):
    def __init__(self, backbone: str, output_stride: int, num_classes: int):
        super(DeepLabv3plus, self).__init__()
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

        # ASPP
        if output_stride == 16:
            atrous_rates = (6, 12, 18)
        elif output_stride == 8:
            atrous_rates = (12, 24, 36)
        else:
            raise NotImplementedError('Wrong output_stride.')
        self.aspp = torchvision.models.segmentation.deeplabv3.ASPP(2048, atrous_rates, 256)

        # Decoder
        self.decoder = Decoder(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size()[2:]

        x = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, self.low_level_feature)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


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
    model = DeepLabv3plus('Xception', output_stride=16, num_classes=19).to(device)
    models.test.test_model(model, (3, 512, 1024), device)
