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
        self.aux_classifier = nn.Conv2d(256, num_classes, 1)

        """
        aux1 = 32x64@256 ( effv2s backbone last )loss_factor =0.2
        aux2 = 128x256 ( aspp 종단 concat-conv-bn-ac-upsamling4x ) loss_factor =0.5 
        aux3 = 128x256 ( dlv3+ skip connection이랑 concat(32channel)한거 이후 ) loss_factor=0.8
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
        
        for i in range(len(output)):
            output[i] = (ce * class_weight) * loss_factor
        """

    def forward(self, x: torch.Tensor):
        self.upsample.size = x.size()[-2:]

        x = self.backbone(x)
        if self.training:
            aux = self.aux_classifier(x)
            aux = self.upsample(aux)
        else:
            aux = None
        x = self.aspp(x)
        x = self.decoder(x, self.low_level_feature)
        x = self.upsample(x)
        if self.training:
            return x, (aux,)
        else:
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
        self.compress_low_level_feature1 = self.make_compressor(64, 64)
        self.compress_low_level_feature2 = self.make_compressor(48, 32)
        self.compress_low_level_feature3 = self.make_compressor(24, 16)

        self.decode1 = self.make_decoder(256 + 64, 256, 256)
        self.decode2 = self.make_decoder(256 + 32, 256, 128)
        self.decode3 = self.make_decoder(128 + 16, 128, 128)

        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, low_level_feature: list[torch.Tensor]) -> torch.Tensor:
        low_level_feature1 = self.compress_low_level_feature1(low_level_feature.pop())
        x = F.interpolate(x, size=low_level_feature1.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, low_level_feature1), dim=1)
        x = self.decode1(x)

        low_level_feature2 = self.compress_low_level_feature2(low_level_feature.pop())
        x = F.interpolate(x, size=low_level_feature2.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, low_level_feature2), dim=1)
        x = self.decode2(x)

        low_level_feature3 = self.compress_low_level_feature3(low_level_feature.pop())
        x = F.interpolate(x, size=low_level_feature3.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, low_level_feature3), dim=1)
        x = self.decode3(x)

        x = self.classifier(x)
        return x

    def make_compressor(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def make_decoder(self, in_channels: int, mid_channels: int, out_channels: int):
        return nn.Sequential(
            models.modules.conv.SeparableConv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            models.modules.conv.SeparableConv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Proposed(num_classes=19, aux_loss=False).to(device)
    models.test.test_model(model, (1, 3, 1024, 2048), '../runs')
