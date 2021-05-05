import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
import torchvision
import torchsummary
import ptflops

import models


class DeepLabV3plus(nn.Module):
    def __init__(self, backbone: str, output_stride: int, num_classes: int) -> None:
        super(DeepLabV3plus, self).__init__()
        # Backbone
        if backbone == 'resnet101':
            self.backbone = models.backbone.resnet101.ResNet101(output_stride)
        elif backbone == 'xception':
            self.backbone = models.backbone.xception.Xception(output_stride)
            self.backbone.load_state_dict(torch.load('weights/xception_65_imagenet.pth'))
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
        x = self.decoder(x, self.backbone.low_level_feature)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x


class Decoder(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(Decoder, self).__init__()
        low_level_feature_channels = 48
        self.compress_low_level_feature = nn.Sequential(
            nn.Conv2d(256, low_level_feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_feature_channels),
            nn.ReLU(inplace=True)
        )
        self.decode1 = self.make_decoder(256 + low_level_feature_channels, 256)
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, low_level_feature: list[torch.Tensor]) -> torch.Tensor:
        low_level_feature = self.compress_low_level_feature(low_level_feature.pop(0))

        x = F.interpolate(x, size=low_level_feature.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, low_level_feature), dim=1)
        x = self.decode1(x)
        x = self.classifier(x)
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
    macs, params = ptflops.get_model_complexity_info(model, (3, 400, 800), print_per_layer_stat=False, as_strings=False)
    print(f'GFLOPs: {macs / 1000000000 * 2}, params: {params}')

    writer = torch.utils.tensorboard.SummaryWriter('../runs')
    writer.add_graph(model, torch.rand(1, 3, 400, 800).to(device))
    writer.close()
