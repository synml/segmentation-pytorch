import torch
import torch.nn as nn

import models


class UNet(nn.Module):
    def __init__(self, num_classes: int):
        super(UNet, self).__init__()
        # Encoder
        self.encode1 = self.make_double_conv(3, 64)
        self.encode2 = self.make_double_conv(64, 128)
        self.encode3 = self.make_double_conv(128, 256)
        self.encode4 = self.make_double_conv(256, 512)
        self.encode5 = self.make_double_conv(512, 1024)
        self.max_pool = nn.MaxPool2d(2)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decode4 = self.make_double_conv(1024, 512)

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
        encode1 = self.encode1(x)
        encode2 = self.encode2(self.max_pool(encode1))
        encode3 = self.encode3(self.max_pool(encode2))
        encode4 = self.encode4(self.max_pool(encode3))
        x = self.encode5(self.max_pool(encode4))

        # Decoder
        x = self.decode4(torch.cat([self.upconv4(x), encode4], dim=1))
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
    model = UNet(19).to(device)
    models.test.test_model(model, (1, 3, 512, 1024), '../runs')
