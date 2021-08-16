from functools import partial

import torch
import torch.nn as nn

import models
from models.backbone.efficientnet_blocks import SqueezeExcite
from models.backbone.efficientnet_builder import EfficientNetBuilder, decode_arch_def, round_channels


state_dict_url = {
    'efficientnetv2_s': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_s-eb54923e.pth',
    'efficientnetv2_m': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_m-cc09e0cd.pth',
    'efficientnetv2_l': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_l-d664b728.pth',
}


class EfficientNet(nn.Module):
    def __init__(self, block_args, stem_size=32, output_stride=32, round_chs_fn=round_channels):
        super(EfficientNet, self).__init__()
        act_layer = nn.SiLU
        norm_layer = nn.BatchNorm2d
        se_layer = SqueezeExcite

        # Stem
        self.conv_stem = nn.Conv2d(3, stem_size, 3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(stem_size)
        self.act1 = act_layer(inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=output_stride, pad_type='', round_chs_fn=round_chs_fn,
            act_layer=act_layer, norm_layer=norm_layer, se_layer=se_layer, drop_path_rate=0.)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        return x


def _gen_efficientnetv2_s(channel_multiplier=1.0, depth_multiplier=1.0):
    arch_def = [
        ['cn_r2_k3_s1_e1_c24_skip'],
        ['er_r4_k3_s2_e4_c48'],
        ['er_r4_k3_s2_e4_c64'],
        ['ir_r6_k3_s2_e4_c128_se0.25'],
        ['ir_r9_k3_s1_e6_c160_se0.25'],
        ['ir_r15_k3_s2_e6_c256_se0.25'],
    ]
    num_features = 1280
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_chs_fn(num_features),
        stem_size=24,
        round_chs_fn=round_chs_fn,
    )
    model = EfficientNet(**model_kwargs)
    return model


def _gen_efficientnetv2_m(channel_multiplier=1.0, depth_multiplier=1.0):
    arch_def = [
        ['cn_r3_k3_s1_e1_c24_skip'],
        ['er_r5_k3_s2_e4_c48'],
        ['er_r5_k3_s2_e4_c80'],
        ['ir_r7_k3_s2_e4_c160_se0.25'],
        ['ir_r14_k3_s1_e6_c176_se0.25'],
        ['ir_r18_k3_s2_e6_c304_se0.25'],
        ['ir_r5_k3_s1_e6_c512_se0.25'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=1280,
        stem_size=24,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
    )
    model = EfficientNet(**model_kwargs)
    return model


def _gen_efficientnetv2_l(channel_multiplier=1.0, depth_multiplier=1.0):
    arch_def = [
        ['cn_r4_k3_s1_e1_c32_skip'],
        ['er_r7_k3_s2_e4_c64'],
        ['er_r7_k3_s2_e4_c96'],
        ['ir_r10_k3_s2_e4_c192_se0.25'],
        ['ir_r19_k3_s1_e6_c224_se0.25'],
        ['ir_r25_k3_s2_e6_c384_se0.25'],
        ['ir_r7_k3_s1_e6_c640_se0.25'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=1280,
        stem_size=32,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
    )
    model = EfficientNet(**model_kwargs)
    return model


def efficientnetv2_s():
    """ EfficientNet-V2 Small. """
    model = _gen_efficientnetv2_s()
    return model


def efficientnetv2_m():
    """ EfficientNet-V2 Medium. """
    model = _gen_efficientnetv2_m()
    return model


def efficientnetv2_l():
    """ EfficientNet-V2 Large. """
    model = _gen_efficientnetv2_l()
    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = efficientnetv2_s().to(device)
    models.test.test_model(model, (3, 512, 1024), device)
