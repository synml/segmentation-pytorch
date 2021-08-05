import torch
import torch.nn as nn

import models
import models.modules.conv


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dilation: int,
                 skip_connection_type: str, grow_first=True):
        super(Block, self).__init__()
        if skip_connection_type == 'conv':
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        elif skip_connection_type == 'sum':
            self.shortcut = None
        else:
            raise ValueError('Wrong skip_connection_type.')

        if grow_first:
            mid_channels = out_channels
        else:
            mid_channels = in_channels
        self.relu1 = nn.ReLU()
        self.sepconv1 = models.modules.conv.SeparableConv2d(in_channels, mid_channels, 3, 1,
                                                            dilation, dilation, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.sepconv2 = models.modules.conv.SeparableConv2d(mid_channels, out_channels, 3, 1,
                                                            dilation, dilation, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.sepconv3 = models.modules.conv.SeparableConv2d(out_channels, out_channels, 3, stride,
                                                            dilation, dilation, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        x = self.relu1(x)
        x = self.sepconv1(x)
        x = self.relu2(x)
        x = self.sepconv2(x)  # forward hook
        x = self.relu3(x)
        x = self.sepconv3(x)

        x += shortcut
        return x


class Xception(nn.Module):
    def __init__(self, output_stride: int):
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

        # Entry flow
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.block1 = Block(64, 128, 2, dilation=1, skip_connection_type='conv')
        self.block2 = Block(128, 256, 2, dilation=1, skip_connection_type='conv')
        self.block3 = Block(256, 728, entry_block3_stride, dilation=1, skip_connection_type='conv')

        # Middle flow
        layers = []
        for _ in range(16):
            layers.append(Block(728, 728, 1, middle_block_dilation, skip_connection_type='sum'))
        self.middle_flow = nn.Sequential(*layers)

        # Exit flow
        self.exit_flow = nn.Sequential(
            Block(728, 1024, 1, exit_block_dilations[0], skip_connection_type='conv', grow_first=False),
            models.modules.conv.SeparableConv2d(1024, 1536, 3, 1, exit_block_dilations[1], exit_block_dilations[1],
                                                bias=False, activation=nn.ReLU(inplace=True)),
            nn.ReLU(inplace=True),
            models.modules.conv.SeparableConv2d(1536, 1536, 3, 1, exit_block_dilations[1], exit_block_dilations[1],
                                                bias=False, activation=nn.ReLU(inplace=True)),
            nn.ReLU(inplace=True),
            models.modules.conv.SeparableConv2d(1536, 2048, 3, 1, exit_block_dilations[1], exit_block_dilations[1],
                                                bias=False, activation=nn.ReLU(inplace=True)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Entry flow
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.middle_flow(x)
        x = self.exit_flow(x)
        return x


def xception(output_stride: int, pretrained: bool = False):
    model = Xception(output_stride)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url('https://github.com/synml/segmentation-pytorch'
                                                        '/releases/download/v1.4.3-weights/xception_65_imagenet.pth')
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = xception(output_stride=16, pretrained=True).to(device)
    models.test.test_model(model, (3, 512, 1024), device)
