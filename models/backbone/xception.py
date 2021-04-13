"""
Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)
@author: tstandley
Adapted by cadene
Creates an Xception Model as defined in:
Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
This weights ported from the Keras implementation. Achieves the following performance on the validation set:
Loss:0.9173 Prec@1:78.892 Prec@5:94.292
REMEMBER to set your image size to 3x299x299 for both test and validation
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import torch
import torch.nn as nn
import torch.utils.tensorboard
import torchsummary

model_urls = {
    'xception': '/home/wangyude/.torch/models/xception_pytorch_imagenet.pth'
    # 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth'
}


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False,
                 activate_first=True, inplace=True) -> None:
        super(SeparableConv2d, self).__init__()
        self.relu0 = nn.ReLU(inplace=inplace)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                   bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.activate_first = activate_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activate_first:
            x = self.relu0(x)
        x = self.depthwise(x)
        x = self.bn1(x)
        if not self.activate_first:
            x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        if not self.activate_first:
            x = self.relu2(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, atrous: int = None,
                 grow_first=True, activate_first=True) -> None:
        super(Block, self).__init__()
        if atrous is None:
            atrous = [1, 1, 1]
        else:
            atrous = [atrous, atrous, atrous]

        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        self.hook_layer = None

        if grow_first:
            filters = out_channels
        else:
            filters = in_channels
        self.sepconv1 = SeparableConv2d(in_channels, filters, 3, stride=1, padding=1 * atrous[0], dilation=atrous[0],
                                        bias=False, activate_first=activate_first, inplace=False)
        self.sepconv2 = SeparableConv2d(filters, out_channels, 3, stride=1, padding=1 * atrous[1], dilation=atrous[1],
                                        bias=False, activate_first=activate_first)
        self.sepconv3 = SeparableConv2d(out_channels, out_channels, 3, stride=stride, padding=1 * atrous[2],
                                        dilation=atrous[2], bias=False, activate_first=activate_first)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skipbn(skip)
        else:
            skip = x

        out = self.sepconv1(x)
        out = self.sepconv2(out)
        self.hook_layer = out
        out = self.sepconv3(out)

        out += skip
        return out


class Xception(nn.Module):
    def __init__(self, output_stride: int) -> None:
        super(Xception, self).__init__()
        if output_stride == 8:
            stride_list = [2, 1, 1]
            rate = 2
        elif output_stride == 16:
            stride_list = [2, 2, 1]
            rate = 1
        else:
            raise NotImplementedError('Wrong output_stride.')

        self.layers = []

        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.block1 = Block(64, 128, 2)
        self.block2 = Block(128, 256, stride_list[0])
        self.block3 = Block(256, 728, stride_list[1])

        # Middle flow
        self.block4 = Block(728, 728, 1, atrous=rate)
        self.block5 = Block(728, 728, 1, atrous=rate)
        self.block6 = Block(728, 728, 1, atrous=rate)
        self.block7 = Block(728, 728, 1, atrous=rate)
        self.block8 = Block(728, 728, 1, atrous=rate)
        self.block9 = Block(728, 728, 1, atrous=rate)
        self.block10 = Block(728, 728, 1, atrous=rate)
        self.block11 = Block(728, 728, 1, atrous=rate)
        self.block12 = Block(728, 728, 1, atrous=rate)
        self.block13 = Block(728, 728, 1, atrous=rate)
        self.block14 = Block(728, 728, 1, atrous=rate)
        self.block15 = Block(728, 728, 1, atrous=rate)
        self.block16 = Block(728, 728, 1, atrous=rate)
        self.block17 = Block(728, 728, 1, atrous=rate)
        self.block18 = Block(728, 728, 1, atrous=rate)
        self.block19 = Block(728, 728, 1, atrous=rate)

        # Exit flow
        self.block20 = Block(728, 1024, stride_list[2], atrous=rate, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1 * rate, dilation=rate, activate_first=False)
        self.conv4 = SeparableConv2d(1536, 1536, 3, 1, 1 * rate, dilation=rate, activate_first=False)
        self.conv5 = SeparableConv2d(1536, 2048, 3, 1, 1 * rate, dilation=rate, activate_first=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.block1(x)
        x = self.block2(x)
        self.layers.append(self.block2.hook_layer)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def get_layers(self):
        return self.layers


def xception(output_stride: int, pretrained: bool) -> Xception:
    model = Xception(output_stride)
    if pretrained:
        old_dict = torch.load(model_urls['xception'])
        old_dict = {k: v for k, v in old_dict.items() if ('itr' not in k and 'tmp' not in k and 'track' not in k)}
        model_dict = model.state_dict()
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = xception(output_stride=16, pretrained=False).to(device)
    model.eval()

    torchsummary.torchsummary.summary(model, (3, 200, 400))

    writer = torch.utils.tensorboard.SummaryWriter('../../runs')
    writer.add_graph(model, torch.rand(1, 3, 400, 800).to(device))
    writer.close()
