import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
import torchsummary


class UNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(UNet, self).__init__()

        self.encode1 = self.double_conv(num_channels, 64)
        self.encode2 = self.double_conv(64, 128)
        self.encode3 = self.double_conv(128, 256)
        self.encode4 = self.double_conv(256, 512)
        self.encode_end = self.double_conv(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decode4 = self.double_conv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decode3 = self.double_conv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decode2 = self.double_conv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decode1 = self.double_conv(128, 64)

        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        encode1 = self.encode1(x)
        encode2 = self.encode2(F.max_pool2d(encode1, 2))
        encode3 = self.encode3(F.max_pool2d(encode2, 2))
        encode4 = self.encode4(F.max_pool2d(encode3, 2))
        encode_end = self.encode_end(F.max_pool2d(encode4, 2))

        # Decoder
        out = self.decode4(torch.cat([self.upconv4(encode_end), encode4], dim=1))
        out = self.decode3(torch.cat([self.upconv3(out), encode3], dim=1))
        out = self.decode2(torch.cat([self.upconv2(out), encode2], dim=1))
        out = self.decode1(torch.cat([self.upconv1(out), encode1], dim=1))

        # Classifier
        out = self.classifier(out)
        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(3, 20).to(device)
    model.eval()

    torchsummary.torchsummary.summary(model, (3, 256, 512))

    input_image = torch.rand(1, 3, 256, 512).to(device)
    model(input_image)

    writer = torch.utils.tensorboard.SummaryWriter('../runs')
    writer.add_graph(model, input_image)
    writer.close()
