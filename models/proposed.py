import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import models
import models.backbone
import models.modules


class Proposed(nn.Module):
    def __init__(self, num_classes: int):
        super(Proposed, self).__init__()
        # Backbone
        efficientnetv2 = models.backbone.efficientnet.efficientnetv2('small', 16, pretrained=True)
        return_nodes = {
            'stages.0.1.add': 'stage1',
            'stages.1.3.add': 'stage2',
            'stages.2.3.add': 'stage3',
            'stages.5.14.add': 'stage6',
        }
        self.backbone = torchvision.models.feature_extraction.create_feature_extractor(efficientnetv2, return_nodes)

        # DAPPM
        self.dappm = models.modules.aspp.DAPPM(256, (6, 12, 18), 256)

        # Decoder
        self.decoder = Decoder()

        # Classifier
        self.classifier = nn.Conv2d(128, num_classes, 1)
        self.upsample = nn.Upsample(mode='bilinear', align_corners=False)

        # Auxiliary classifier
        self.aux_classifier1 = nn.Conv2d(256, num_classes, 1)
        self.aux_classifier2 = nn.Conv2d(256, num_classes, 1)

    def forward(self, x: torch.Tensor):
        self.upsample.size = x.size()[-2:]

        if self.training:
            features = self.backbone(x)
            x = features.pop('stage6')
            aux1 = self.aux_classifier1(x)
            aux1 = self.upsample(aux1)
            x = self.dappm(x)
            aux2 = self.aux_classifier2(x)
            aux2 = self.upsample(aux2)
            x = self.decoder(x, features)
            x = self.classifier(x)
            x = self.upsample(x)
            return x, (aux1, aux2)
        else:
            features = self.backbone(x)
            x = features.pop('stage6')
            x = self.dappm(x)
            x = self.decoder(x, features)
            x = self.classifier(x)
            x = self.upsample(x)
            return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.feature_refinement_module1 = self.make_feature_refinement_module(64, 64)
        self.feature_refinement_module2 = self.make_feature_refinement_module(48, 32)
        self.feature_refinement_module3 = self.make_feature_refinement_module(24, 16)

        self.decoding_block1 = self.make_decoding_block(256 + 64, 256, 256)
        self.decoding_block2 = self.make_decoding_block(256 + 32, 256, 128)
        self.decoding_block3 = self.make_decoding_block(128 + 16, 128, 128)

    def forward(self, x: torch.Tensor, low_level_feature: dict) -> torch.Tensor:
        low_level_feature1 = self.feature_refinement_module1(low_level_feature.pop('stage3'))
        x = F.interpolate(x, size=low_level_feature1.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, low_level_feature1), dim=1)
        x = self.decoding_block1(x)

        low_level_feature2 = self.feature_refinement_module2(low_level_feature.pop('stage2'))
        x = F.interpolate(x, size=low_level_feature2.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, low_level_feature2), dim=1)
        x = self.decoding_block2(x)

        low_level_feature3 = self.feature_refinement_module3(low_level_feature.pop('stage1'))
        x = F.interpolate(x, size=low_level_feature3.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, low_level_feature3), dim=1)
        x = self.decoding_block3(x)
        return x

    def make_feature_refinement_module(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def make_decoding_block(self, in_channels: int, mid_channels: int, out_channels: int):
        return nn.Sequential(
            models.modules.conv.SeparableConv2d(in_channels, mid_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            models.modules.conv.SeparableConv2d(mid_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
        )


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Proposed(num_classes=19).to(device)
    models.test.test_model(model, (1, 3, 1024, 2048), '../runs')
