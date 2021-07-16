import torch
import torch.nn as nn
import torchvision

import models


class AR_UNet(nn.Module):
    def __init__(self, num_classes: int):
        super(AR_UNet, self).__init__()
        # Backbone
        backbone = torchvision.models.resnet34(pretrained=True)
        self.stem_block = self.make_double_conv(3, 64)
        self.encode1 = backbone.layer1  # 64
        self.encode2 = backbone.layer2  # 128, 1/2
        self.encode3 = backbone.layer3  # 256, 1/4
        self.encode4 = backbone.layer4  # 512, 1/8

        # ASPP
        self.aspp = torchvision.models.segmentation.deeplabv3.ASPP(512, (3, 6, 9), 512)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decode3 = self.make_double_conv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decode2 = self.make_double_conv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decode1 = self.make_double_conv(128, 64)

        # Classifier
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x = self.stem_block(x)
        encode1 = self.encode1(x)
        encode2 = self.encode2(encode1)
        encode3 = self.encode3(encode2)
        x = self.encode4(encode3)
        x = self.aspp(x)

        # Decoder
        x = self.decode3(torch.cat([self.upconv3(x), encode3], dim=1))
        x = self.decode2(torch.cat([self.upconv2(x), encode2], dim=1))
        x = self.decode1(torch.cat([self.upconv1(x), encode1], dim=1))

        # Classifier
        x = self.classifier(x)
        return x

    def make_double_conv(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AR_UNet(20).to(device)
    models.test.test_model(model, (3, 400, 800), device)
