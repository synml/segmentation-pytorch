import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import models


class AR_UNet(nn.Module):
    def __init__(self, num_classes: int):
        super(AR_UNet, self).__init__()
        # Backbone
        backbone = torchvision.models.resnet34(pretrained=True)
        self.initial_conv = self.make_initial_conv(3, 64)
        self.encode1 = backbone.layer1  # 64
        self.encode2 = backbone.layer2  # 128, 1/2
        self.encode3 = backbone.layer3  # 256, 1/4
        self.encode4 = backbone.layer4  # 512, 1/8

        # ASPP
        self.aspp = ASPP(512, 512)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decode3 = self.make_decoder(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decode2 = self.make_decoder(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decode1 = self.make_decoder(128, 64)

        # Classifier
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        encode1 = self.encode1(self.initial_conv(x))
        encode2 = self.encode2(encode1)
        encode3 = self.encode3(encode2)
        encode_end = self.aspp(self.encode4(encode3))

        # Decoder
        out = self.decode3(torch.cat([self.upconv3(encode_end), encode3], dim=1))
        out = self.decode2(torch.cat([self.upconv2(out), encode2], dim=1))
        out = self.decode1(torch.cat([self.upconv1(out), encode1], dim=1))

        # Classifier
        out = self.classifier(out)
        return out

    def make_decoder(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def make_initial_conv(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


# ASPP(Atrous Spatial Pyramid Pooling) Module
class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ASPP, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=9, dilation=9),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        branch5 = F.interpolate(self.branch5(x), size=tuple(x.size()[2:]), mode="bilinear", align_corners=False)

        out = self.final_conv(torch.cat([branch1, branch2, branch3, branch4, branch5], dim=1))
        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AR_UNet(20).to(device)
    models.test.test_model(model, (3, 400, 800), device)
