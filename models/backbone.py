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
        self.classifier_2s = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifier_4s = nn.Conv2d(256, num_classes, kernel_size=1)
        self.classifier_8s = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        initial_conv = self.initial_conv(x)
        layer1 = self.layer1(initial_conv)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # Classifier
        classifier_8s = self.classifier_8s(layer4)
        classifier_4s = self.classifier_4s(layer3)
        classifier_2s = self.classifier_2s(layer2)

        # FCN
        classifier_4s += F.interpolate(classifier_8s, scale_factor=2, mode='bilinear', align_corners=False)
        classifier_2s += F.interpolate(classifier_4s, scale_factor=2, mode='bilinear', align_corners=False)
        out = F.interpolate(classifier_2s, scale_factor=2, mode='bilinear', align_corners=False)
        return out

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
    model = Backbone(8).to(device)
    model.eval()

    torchsummary.torchsummary.summary(model, (3, 256, 512))

    writer = torch.utils.tensorboard.SummaryWriter('../runs')
    writer.add_graph(model, torch.rand(1, 3, 256, 512).to(device))
    writer.close()
