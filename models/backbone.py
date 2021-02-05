import os

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
import torchvision
import torchsummary


class Backbone(nn.Module):
    def __init__(self, num_classes: int):
        super(Backbone, self).__init__()
        # Backbone
        resnet34 = torchvision.models.resnet34(pretrained=True)
        self.initial_conv = self.make_initial_conv(3, 64)
        self.layer1 = resnet34.layer1  # 64
        self.layer2 = resnet34.layer2  # 128, 1/2
        self.layer3 = resnet34.layer3  # 256, 1/4
        self.layer4 = resnet34.layer4  # 512, 1/8

        # Classifier
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classifier
        x = self.classifier(x)

        # Upsample
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        return x

    def make_initial_conv(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


def load_backbone(num_classes: int, pretrained=False):
    model = Backbone(num_classes)
    if pretrained:
        if os.path.exists('weights/Backbone_best.pth'):
            model.load_state_dict(torch.load('weights/Backbone_best.pth'))
        else:
            print('FileNotFound: pretrained_weights (Backbone)')
    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Backbone(20).to(device)
    model.eval()

    torchsummary.torchsummary.summary(model, (3, 400, 800))

    writer = torch.utils.tensorboard.SummaryWriter('../runs')
    writer.add_graph(model, torch.rand(1, 3, 400, 800).to(device))
    writer.close()
