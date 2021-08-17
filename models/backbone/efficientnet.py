import torch
import torch.nn as nn

import models


state_dict_urls = {
    'small': 'https://github.com/synml/segmentation-pytorch/releases/download/v1.5.0-weights/efficientnetv2_small.pth',
    'medium': 'https://github.com/synml/segmentation-pytorch/releases/download/v1.5.0-weights/efficientnetv2_medium.pth',
    'large': 'https://github.com/synml/segmentation-pytorch/releases/download/v1.5.0-weights/efficientnetv2_large.pth',
}
arch_defs = {
    'small': [
        ['cn_r2_k3_s1_e1_c24_skip'],
        ['er_r4_k3_s2_e4_c48'],
        ['er_r4_k3_s2_e4_c64'],
        ['ir_r6_k3_s2_e4_c128_se0.25'],
        ['ir_r9_k3_s1_e6_c160_se0.25'],
        ['ir_r15_k3_s2_e6_c256_se0.25'],
    ],
    'medium': [
        ['cn_r3_k3_s1_e1_c24_skip'],
        ['er_r5_k3_s2_e4_c48'],
        ['er_r5_k3_s2_e4_c80'],
        ['ir_r7_k3_s2_e4_c160_se0.25'],
        ['ir_r14_k3_s1_e6_c176_se0.25'],
        ['ir_r18_k3_s2_e6_c304_se0.25'],
        ['ir_r5_k3_s1_e6_c512_se0.25'],
    ],
    'large': [
        ['cn_r4_k3_s1_e1_c32_skip'],
        ['er_r7_k3_s2_e4_c64'],
        ['er_r7_k3_s2_e4_c96'],
        ['ir_r10_k3_s2_e4_c192_se0.25'],
        ['ir_r19_k3_s1_e6_c224_se0.25'],
        ['ir_r25_k3_s2_e6_c384_se0.25'],
        ['ir_r7_k3_s1_e6_c640_se0.25'],
    ],
}
stem_out_channels = {
    'small': 24,
    'medium': 24,
    'large': 32,
}


class EfficientNet(nn.Module):
    def __init__(self, block_args, stem_out_channels: int, output_stride: int):
        super(EfficientNet, self).__init__()
        # Stem
        self.conv_stem = nn.Conv2d(3, stem_out_channels, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stem_out_channels)
        self.act1 = nn.SiLU(inplace=True)

        # Stages
        builder = models.backbone.efficientnet_builder.EfficientNetBuilder(output_stride)
        blocks = builder(stem_out_channels, block_args)
        self.stages = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        for stage in self.stages:
            x = stage(x)
        return x


def efficientnetv2(model_type: str, output_stride: int, pretrained: bool = False):
    assert model_type in ('small', 'medium', 'large')
    block_args = models.backbone.efficientnet_builder.decode_arch_def(arch_defs[model_type])
    model = EfficientNet(block_args, stem_out_channels[model_type], output_stride)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(state_dict_urls[model_type])
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = efficientnetv2('small', output_stride=32, pretrained=True).to(device)
    models.test.test_model(model, (3, 512, 1024), device)
