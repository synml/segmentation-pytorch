import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import models.backbone
import models.modules
import utils


class Proposed(nn.Module):
    def __init__(self, num_classes: int):
        super(Proposed, self).__init__()
        self.low_level_feature = []

        # Backbone
        self.backbone = models.backbone.efficientnet.efficientnetv2('small', 16, pretrained=True)
        self.backbone.stages[0].register_forward_hook(utils.hooks.get_feature_maps(self.low_level_feature))
        self.backbone.stages[1].register_forward_hook(utils.hooks.get_feature_maps(self.low_level_feature))
        self.backbone.stages[2].register_forward_hook(utils.hooks.get_feature_maps(self.low_level_feature))

        # ASPP
        self.aspp = models.modules.aspp.ASPPwDSConv(256, (6, 12, 18), 256)

        # Decoder
        self.decoder = Decoder(num_classes)
        self.upsample = nn.Upsample(mode='bilinear', align_corners=False)

        # Auxiliary classifier
        self.aux_classifier1 = nn.Conv2d(256, num_classes, 1)
        self.aux_classifier2 = nn.Conv2d(256, num_classes, 1)

        """
        aux1 = 32x64@256 ( effv2s backbone last )loss_factor =0.2
        aux2 = 128x256 ( aspp 종단 concat-conv-bn-ac-upsamling4x ) loss_factor =0.5 
        original 종단 아웃풋 loss_factor=1
        
        def classifier(x, upper_factor=4, use_aux):
            if use_aux:
                x = SepConv(256, 256)(x)
            
            x = Conv2D(256, 19, kernel_size=(1, 1), padding='same', use_bias=True)(x)
            x = Upsampling2D((upper_factor, upper_factor))(x)
        
        loss = CE
        GLOBAL_BATCH_SIZE = 16
        Optimizer = Adam
        LearningRateScheduler = PolynomialDecay
        
        그리고, 중요한거 
        ignore index를 제외한 모든 class 번호의 빈도를 계산하여 class_weight를 계산
        
        self.class_weight = (0.8373, 0.918, 0.866, 1.0345,
                        1.0166, 0.9969, 0.9754, 1.0489,
                        0.8786, 1.0023, 0.9539, 0.9843,
                        1.1116, 0.9037, 1.0865, 1.0955,
                        1.0865, 1.1529, 1.0507)
        
        for i in range(len(output)):
            output[i] = (ce * class_weight) * loss_factor
        """

    def forward(self, x: torch.Tensor):
        self.upsample.size = x.size()[-2:]

        if self.training:
            x = self.backbone(x)
            aux1 = self.aux_classifier1(x)
            aux1 = self.upsample(aux1)
            x = self.aspp(x)
            aux2 = self.aux_classifier2(x)
            aux2 = self.upsample(aux2)
            x = self.decoder(x, self.low_level_feature)
            x = self.upsample(x)
            return x, (aux1, aux2)
        else:
            x = self.backbone(x)
            x = self.aspp(x)
            x = self.decoder(x, self.low_level_feature)
            x = self.upsample(x)
            return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


class Decoder(nn.Module):
    def __init__(self, num_classes: int):
        super(Decoder, self).__init__()
        self.feature_refinement_module1 = self.make_feature_refinement_module(64, 64)
        self.feature_refinement_module2 = self.make_feature_refinement_module(48, 32)
        self.feature_refinement_module3 = self.make_feature_refinement_module(24, 16)

        self.decoding_block1 = self.make_decoding_block(256 + 64, 256, 256)
        self.decoding_block2 = self.make_decoding_block(256 + 32, 256, 128)
        self.decoding_block3 = self.make_decoding_block(128 + 16, 128, 128)

        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, low_level_feature: list[torch.Tensor]) -> torch.Tensor:
        low_level_feature1 = self.feature_refinement_module1(low_level_feature.pop())
        x = F.interpolate(x, size=low_level_feature1.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, low_level_feature1), dim=1)
        x = self.decoding_block1(x)

        low_level_feature2 = self.feature_refinement_module2(low_level_feature.pop())
        x = F.interpolate(x, size=low_level_feature2.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, low_level_feature2), dim=1)
        x = self.decoding_block2(x)

        low_level_feature3 = self.feature_refinement_module3(low_level_feature.pop())
        x = F.interpolate(x, size=low_level_feature3.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, low_level_feature3), dim=1)
        x = self.decoding_block3(x)

        x = self.classifier(x)
        return x

    def make_feature_refinement_module(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def make_decoding_block(self, in_channels: int, mid_channels: int, out_channels: int):
        return nn.Sequential(
            models.modules.conv.SeparableConv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            models.modules.conv.SeparableConv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Proposed(num_classes=19).to(device)
    models.test.test_model(model, (1, 3, 1024, 2048), '../runs')
