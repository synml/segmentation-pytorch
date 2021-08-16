import torch
import torch.nn as nn

import models
from models.backbone.efficientnet_builder import EfficientNetBuilder, decode_arch_def


state_dict_url = {
    'small': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_s-eb54923e.pth',
    'medium': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_m-cc09e0cd.pth',
    'large': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_l-d664b728.pth',
}


class EfficientNet(nn.Module):
    def __init__(self, block_args, stem_out_channels, output_stride=32):
        super(EfficientNet, self).__init__()
        act_layer = nn.SiLU
        norm_layer = nn.BatchNorm2d
        se_layer = models.backbone.efficientnet_blocks.SqueezeExcite

        # Stem
        self.conv_stem = nn.Conv2d(3, stem_out_channels, 3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(stem_out_channels)
        self.act1 = act_layer(inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(output_stride, act_layer=act_layer, norm_layer=norm_layer, se_layer=se_layer)
        self.blocks = nn.Sequential(*builder(stem_out_channels, block_args))

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        return x


def efficientnetv2_small():
    arch_def = [
        ['cn_r2_k3_s1_e1_c24_skip'],
        ['er_r4_k3_s2_e4_c48'],
        ['er_r4_k3_s2_e4_c64'],
        ['ir_r6_k3_s2_e4_c128_se0.25'],
        ['ir_r9_k3_s1_e6_c160_se0.25'],
        ['ir_r15_k3_s2_e6_c256_se0.25'],
    ]
    model = EfficientNet(decode_arch_def(arch_def), stem_out_channels=24)
    return model


def efficientnetv2_medium():
    arch_def = [
        ['cn_r3_k3_s1_e1_c24_skip'],
        ['er_r5_k3_s2_e4_c48'],
        ['er_r5_k3_s2_e4_c80'],
        ['ir_r7_k3_s2_e4_c160_se0.25'],
        ['ir_r14_k3_s1_e6_c176_se0.25'],
        ['ir_r18_k3_s2_e6_c304_se0.25'],
        ['ir_r5_k3_s1_e6_c512_se0.25'],
    ]
    model = EfficientNet(decode_arch_def(arch_def), stem_out_channels=24)
    return model


def efficientnetv2_large():
    arch_def = [
        ['cn_r4_k3_s1_e1_c32_skip'],
        ['er_r7_k3_s2_e4_c64'],
        ['er_r7_k3_s2_e4_c96'],
        ['ir_r10_k3_s2_e4_c192_se0.25'],
        ['ir_r19_k3_s1_e6_c224_se0.25'],
        ['ir_r25_k3_s2_e6_c384_se0.25'],
        ['ir_r7_k3_s1_e6_c640_se0.25'],
    ]
    model = EfficientNet(decode_arch_def(arch_def), stem_out_channels=32)
    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = efficientnetv2_small().to(device)
    models.test.test_model(model, (3, 512, 1024), device)
