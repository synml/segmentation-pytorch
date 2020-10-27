import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard

import utils.utils


def double_conv(in_channels, out_channels, batch_normalization=False):
    if batch_normalization:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()

        self.down1 = double_conv(n_channels, 64)
        self.down2 = double_conv(64, 128)
        self.down3 = double_conv(128, 256)
        self.down4 = double_conv(256, 512)
        self.bottom = double_conv(512, 1024)

        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.up4 = double_conv(1024, 512)
        self.up3 = double_conv(512, 256)
        self.up2 = double_conv(256, 128)
        self.up1 = double_conv(128, 64)

        self.classifier = nn.Conv2d(64, n_classes, kernel_size=1)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        down1 = self.down1(x)

        x = F.max_pool2d(down1, 2)
        down2 = self.down2(x)

        x = F.max_pool2d(down2, 2)
        down3 = self.down3(x)

        x = F.max_pool2d(down3, 2)
        down4 = self.down4(x)

        x = F.max_pool2d(down4, 2)
        x = self.bottom(x)

        # --------------------------------------

        x = self.upconv1(x)
        x = torch.cat([x, down4], dim=1)
        x = self.up4(x)

        x = self.upconv2(x)
        x = torch.cat([x, down3], dim=1)
        x = self.up3(x)

        x = self.upconv3(x)
        x = torch.cat([x, down2], dim=1)
        x = self.up2(x)

        x = self.upconv4(x)
        x = torch.cat([x, down1], dim=1)
        x = self.up1(x)

        x = self.classifier(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(3, 19).to(device)
    model.apply(utils.utils.init_weights)

    input_image = torch.rand(1, 3, 256, 256).to(device)
    out = model(input_image)

    writer = torch.utils.tensorboard.SummaryWriter('../runs')
    writer.add_graph(model, input_image)
    writer.close()
