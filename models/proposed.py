import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
import torchvision
import torchsummary


# ASPP(Atrous Spatial Pyramid Pooling) Module
class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ASPP, self).__init__()

        # 1번 branch = 1x1 convolution → BatchNorm → ReLu
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 2번 branch = 3x3 atrous convolution → BatchNorm → ReLu
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 3번 branch = 3x3 atrous convolution → BatchNorm → ReLu
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 4번 branch = 3x3 atrous convolution → BatchNorm → ReLu
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=9, dilation=9),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 5번 branch = AdaptiveAvgPool2d → 1x1 convolution → BatchNorm → ReLu
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 최종 출력 convolution
        self.outconv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        branch5 = F.interpolate(self.branch5(x), size=(x.size()[2], x.size()[3]), mode="bilinear", align_corners=False)

        out = self.outconv(torch.cat([branch1, branch2, branch3, branch4, branch5], dim=1))
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)

        out += identity
        out = self.relu2(out)

        return out


class Proposed(nn.Module):
    def __init__(self, num_classes: int):
        super(Proposed, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)

        self.encode1 = self.double_conv(3, 64)
        self.encode2 = resnet50.layer1  # 256
        self.encode3 = resnet50.layer2  # 512
        self.encode4 = resnet50.layer3  # 1024
        self.encode_end = resnet50.layer4  # 2048
        self.aspp = ASPP(2048, 2048)

        self.upconv4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.decode4 = self.double_conv(2048, 1024)

        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decode3 = self.double_conv(1024, 512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decode2 = self.double_conv(512, 256)

        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decode1 = self.double_conv(128 + 64, 64)

        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def double_conv(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def make_layer(self, in_channels, out_channels, num_blocks):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)]

        for _ in range(num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        encode1 = self.encode1(x)
        encode2 = self.encode2(F.max_pool2d(encode1, 2))
        encode3 = self.encode3(encode2)
        encode4 = self.encode4(encode3)
        encode_end = self.encode_end(encode4)
        encode_end = self.aspp(encode_end)

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
    model = Proposed(8).to(device)
    model.eval()

    torchsummary.torchsummary.summary(model, (3, 256, 512))

    writer = torch.utils.tensorboard.SummaryWriter('../runs')
    writer.add_graph(model, torch.rand(1, 3, 256, 512).to(device))
    writer.close()
