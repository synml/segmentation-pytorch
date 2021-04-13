import torch
import torch.nn as nn
import torch.utils.tensorboard
import torchsummary


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, dilation=1,
                 bias=True, activation_first=True, inplace=True) -> None:
        super(SeparableConv2d, self).__init__()
        self.relu0 = nn.ReLU(inplace=inplace)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                                   groups=in_channels, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.activation_first = activation_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_first:
            x = self.relu0(x)
        x = self.depthwise(x)
        x = self.bn1(x)
        if not self.activation_first:
            x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        if not self.activation_first:
            x = self.relu2(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dilation: int,
                 skip_connection_type: str, grow_first=True) -> None:
        super(Block, self).__init__()

        if skip_connection_type == 'conv':
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        elif skip_connection_type == 'sum':
            self.skip = None
            self.skipbn = None
        else:
            raise NotImplementedError('Wrong skip_connection_type.')

        self.hook_layer = None

        if grow_first:
            mid_channels = out_channels
        else:
            mid_channels = in_channels
        self.sepconv1 = SeparableConv2d(in_channels, mid_channels, 3, 1, padding=dilation, dilation=dilation,
                                        bias=False, inplace=False)
        self.sepconv2 = SeparableConv2d(mid_channels, out_channels, 3, 1, padding=dilation, dilation=dilation,
                                        bias=False)
        self.sepconv3 = SeparableConv2d(out_channels, out_channels, 3, stride, padding=dilation, dilation=dilation,
                                        bias=False)

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
        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError('Wrong output_stride.')

        self.low_level_features = []

        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.block1 = Block(64, 128, 2, dilation=1, skip_connection_type='conv')
        self.block2 = Block(128, 256, 2, dilation=1, skip_connection_type='conv')
        self.block3 = Block(256, 728, entry_block3_stride, dilation=1, skip_connection_type='conv')

        # Middle flow
        self.block4 = Block(728, 728, 1, middle_block_dilation, skip_connection_type='sum')
        self.block5 = Block(728, 728, 1, middle_block_dilation, skip_connection_type='sum')
        self.block6 = Block(728, 728, 1, middle_block_dilation, skip_connection_type='sum')
        self.block7 = Block(728, 728, 1, middle_block_dilation, skip_connection_type='sum')
        self.block8 = Block(728, 728, 1, middle_block_dilation, skip_connection_type='sum')
        self.block9 = Block(728, 728, 1, middle_block_dilation, skip_connection_type='sum')
        self.block10 = Block(728, 728, 1, middle_block_dilation, skip_connection_type='sum')
        self.block11 = Block(728, 728, 1, middle_block_dilation, skip_connection_type='sum')
        self.block12 = Block(728, 728, 1, middle_block_dilation, skip_connection_type='sum')
        self.block13 = Block(728, 728, 1, middle_block_dilation, skip_connection_type='sum')
        self.block14 = Block(728, 728, 1, middle_block_dilation, skip_connection_type='sum')
        self.block15 = Block(728, 728, 1, middle_block_dilation, skip_connection_type='sum')
        self.block16 = Block(728, 728, 1, middle_block_dilation, skip_connection_type='sum')
        self.block17 = Block(728, 728, 1, middle_block_dilation, skip_connection_type='sum')
        self.block18 = Block(728, 728, 1, middle_block_dilation, skip_connection_type='sum')
        self.block19 = Block(728, 728, 1, middle_block_dilation, skip_connection_type='sum')

        # Exit flow
        self.block20 = Block(728, 1024, 1, exit_block_dilations[0], skip_connection_type='conv', grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, exit_block_dilations[1], exit_block_dilations[1],
                                     bias=False, activation_first=False)
        self.conv4 = SeparableConv2d(1536, 1536, 3, 1, exit_block_dilations[1], exit_block_dilations[1],
                                     bias=False, activation_first=False)
        self.conv5 = SeparableConv2d(1536, 2048, 3, 1, exit_block_dilations[1], exit_block_dilations[1],
                                     bias=False, activation_first=False)

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
        self.low_level_features.append(self.block2.hook_layer)
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


def load_xception(output_stride: int, pretrained: bool) -> Xception:
    model = Xception(output_stride)
    if pretrained:
        old_dict = torch.load('../../weights/xception_pytorch_imagenet.pth')
        model_dict = model.state_dict()
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_xception(output_stride=16, pretrained=True).to(device)
    model.eval()

    torchsummary.torchsummary.summary(model, (3, 200, 400))

    writer = torch.utils.tensorboard.SummaryWriter('../../runs')
    writer.add_graph(model, torch.rand(1, 3, 400, 800).to(device))
    writer.close()
