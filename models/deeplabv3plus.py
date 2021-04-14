import torch
import torch.nn as nn
import torch.utils.tensorboard
import torchvision
import torchsummary

import models.backbone.resnet101
import models.backbone.xception


class DeepLabV3plus(nn.Module):
    def __init__(self, backbone: str, output_stride: int, num_classes: int) -> None:
        super(DeepLabV3plus, self).__init__()
        # Backbone
        if backbone == 'resnet101':
            self.backbone = models.backbone.resnet101.load_resnet101(output_stride, pretrained=True)
        elif backbone == 'xception':
            self.backbone = models.backbone.xception.load_xception(output_stride, pretrained=True)
        else:
            raise NotImplementedError('Wrong backbone.')

        self.aspp = ASPP(2048, output_stride, 256)
        self.decoder = Decoder(backbone, output_stride, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, self.backbone.low_level_feature[0])
        return x


class ASPP(torchvision.models.segmentation.deeplabv3.ASPP):
    def __init__(self, in_channels: int, output_stride: int, out_channels=256) -> None:
        if output_stride == 16:
            atrous_rates = [6, 12, 18]
        elif output_stride == 8:
            atrous_rates = [12, 24, 36]
        else:
            raise NotImplementedError('Wrong output_stride.')

        super(ASPP, self).__init__(in_channels, atrous_rates, out_channels)


class Decoder(nn.Module):
    def __init__(self, backbone: str, output_stride: int, num_classes: int) -> None:
        super(Decoder, self).__init__()
        if backbone == 'resnet101':
            in_channels = 256
        elif backbone == 'xception':
            in_channels = 128
        else:
            raise NotImplementedError('Wrong backbone.')

        if output_stride == 16:
            scale_factor = 4
        elif output_stride == 8:
            scale_factor = 2
        else:
            raise NotImplementedError('Wrong output_stride.')

        low_level_feature_channels = 48
        self.compress_low_level_feature = nn.Sequential(
            nn.Conv2d(in_channels, low_level_feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_feature_channels),
            nn.ReLU(inplace=True)
        )
        self.upsample1 = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.decode1 = self.make_decoder(in_channels + low_level_feature_channels, 256)
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x: torch.Tensor, low_level_feature: torch.Tensor) -> torch.Tensor:
        x = torch.cat((self.upsample1(x), self.compress_low_level_feature(low_level_feature)), dim=1)
        x = self.decode1(x)
        x = self.classifier(x)
        x = self.upsample2(x)
        return x

    def make_decoder(self, in_channels: int, out_channels: int) -> nn.Sequential:
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
    model = DeepLabV3plus('resnet101', output_stride=16, num_classes=20).to(device)
    model.eval()

    torchsummary.torchsummary.summary(model, (3, 400, 800))

    writer = torch.utils.tensorboard.SummaryWriter('../runs')
    writer.add_graph(model, torch.rand(1, 3, 400, 800).to(device))
    writer.close()
