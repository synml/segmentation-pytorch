import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import models
import models.modules


class EAR_Net(nn.Module):
    def __init__(self, num_classes: int):
        super(EAR_Net, self).__init__()
        # Backbone
        backbone = torchvision.models.resnext50_32x4d(pretrained=True)
        self.stem_block = self.make_stem_block(3, 64)
        self.encode1 = backbone.layer1
        self.encode2 = backbone.layer2
        self.encode3 = backbone.layer3
        self.encode4 = backbone.layer4

        # ASPP
        atrous_rates = (3, 6, 9)
        self.aspp = models.modules.aspp.ASPPwDSConv(2048, atrous_rates, 256)

        # Decoder
        self.compress_low_level_feature3 = self.make_compressor(1024, 64)
        self.compress_low_level_feature2 = self.make_compressor(512, 32)
        self.compress_low_level_feature1 = self.make_compressor(256, 16)
        self.decode3 = self.make_decoder(320, 256)
        self.decode2 = self.make_decoder(288, 128)
        self.decode1 = self.make_decoder(144, 64)

        # Classifier
        self.classifier = models.modules.conv.SeparableConv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.size()

        # Encoder
        x = self.stem_block(x)
        encode1 = self.encode1(x)
        encode2 = self.encode2(encode1)
        encode3 = self.encode3(encode2)
        x = self.encode4(encode3)
        x = self.aspp(x)

        # Decoder
        x = F.interpolate(x, encode3.size()[2:], mode='bilinear', align_corners=False)
        encode3 = self.compress_low_level_feature3(encode3)
        x = self.decode3(torch.cat((x, encode3), dim=1))

        x = F.interpolate(x, encode2.size()[2:], mode='bilinear', align_corners=False)
        encode2 = self.compress_low_level_feature2(encode2)
        x = self.decode2(torch.cat((x, encode2), dim=1))

        x = F.interpolate(x, encode1.size()[2:], mode='bilinear', align_corners=False)
        encode1 = self.compress_low_level_feature1(encode1)
        x = self.decode1(torch.cat((x, encode1), dim=1))

        # Classifier
        x = F.interpolate(x, input_size[2:], mode='bilinear', align_corners=False)
        x = self.classifier(x)
        return x

    def make_stem_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            models.modules.conv.SeparableConv2d(in_channels, out_channels // 2, kernel_size=3,
                                                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            models.modules.conv.SeparableConv2d(out_channels // 2, out_channels, kernel_size=3,
                                                stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def make_compressor(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            models.modules.conv.SeparableConv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def make_decoder(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            models.modules.conv.SeparableConv2d(in_channels, out_channels, kernel_size=3,
                                                stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            models.modules.conv.SeparableConv2d(out_channels, out_channels, kernel_size=3,
                                                stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EAR_Net(19).to(device)
    models.test.test_model(model, (3, 400, 240), device)
