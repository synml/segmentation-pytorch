import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
import torchsummary

import utils.utils


# ASPP(Atrous Spatial Pyramid Pooling) Module
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        # 1번 branch = 1x1 convolution → BatchNorm → ReLu
        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)

        # 2번 branch = 3x3 convolution w/ rate=6 (or 12) → BatchNorm → ReLu
        self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(out_channels)

        # 3번 branch = 3x3 convolution w/ rate=12 (or 24) → BatchNorm → ReLu
        self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(out_channels)

        # 4번 branch = 3x3 convolution w/ rate=18 (or 36) → BatchNorm → ReLu
        self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(out_channels)

        # 5번 branch = AdaptiveAvgPool2d → 1x1 convolution → BatchNorm → ReLu
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)

        self.conv_1x1_3 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)  # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # feature map의 shape은 (batch_size, in_channels, height/output_stride, width/output_stride)
        feature_map_h = x.size()[2]  # (== h/16)
        feature_map_w = x.size()[3]  # (== w/16)

        # 1번 branch = 1x1 convolution → BatchNorm → ReLu
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x)))
        # 2번 branch = 3x3 convolution w/ rate=6 (or 12) → BatchNorm → ReLu
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(x)))
        # 3번 branch = 3x3 convolution w/ rate=12 (or 24) → BatchNorm → ReLu
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(x)))
        # 4번 branch = 3x3 convolution w/ rate=18 (or 36) → BatchNorm → ReLu
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(x)))

        # 5번 branch = AdaptiveAvgPool2d → 1x1 convolution → BatchNorm → ReLu
        # shape: (batch_size, in_channels, 1, 1)
        out_img = self.avg_pool(x)
        # shape: (batch_size, out_channels, 1, 1)
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), mode="bilinear", align_corners=False)

        # shape: (batch_size, out_channels * 5, height/output_stride, width/output_stride)
        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
        # shape: (batch_size, out_channels, height/output_stride, width/output_stride)
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))

        return out


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


def double_atrous_conv(in_channels, out_channels, batch_normalization=False):
    if batch_normalization:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )


class Proposed(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(Proposed, self).__init__()

        self.encode1 = double_atrous_conv(num_channels, 64)
        self.encode2 = double_atrous_conv(64, 128)
        self.encode3 = double_atrous_conv(128, 256)
        self.encode4 = double_atrous_conv(256, 512)
        self.encode5 = double_atrous_conv(512, 512)
        self.aspp = ASPP(512, 512)

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.decode4 = double_atrous_conv(768, 512)
        self.decode3 = double_atrous_conv(512, 256)
        self.decode2 = double_atrous_conv(256, 128)
        self.decode1 = double_atrous_conv(128, 64)

        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        encode1 = self.encode1(x)
        encode2 = self.encode2(F.max_pool2d(encode1, 2))
        encode3 = self.encode3(F.max_pool2d(encode2, 2))
        encode4 = self.encode4(F.max_pool2d(encode3, 2))
        encode_end = self.aspp(self.encode5(F.max_pool2d(encode4, 2)))

        # Decoder
        x = self.decode4(torch.cat([self.upconv4(encode_end), encode4], dim=1))
        x = self.decode3(torch.cat([self.upconv3(x), encode3], dim=1))
        x = self.decode2(torch.cat([self.upconv2(x), encode2], dim=1))
        x = self.decode1(torch.cat([self.upconv1(x), encode1], dim=1))

        # Classifier
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Proposed(3, 8).to(device)
    model.apply(utils.utils.init_weights)
    model.eval()

    torchsummary.torchsummary.summary(model, (3, 256, 512))

    input_image = torch.rand(1, 3, 256, 512).to(device)
    out = model(input_image)

    writer = torch.utils.tensorboard.SummaryWriter('../runs')
    writer.add_graph(model, input_image)
    writer.close()
