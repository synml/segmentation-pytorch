import torch.nn as nn
import torch.utils.tensorboard
import torchvision
import torchsummary
import ptflops


class ResNet101(nn.Module):
    def __init__(self, output_stride: int) -> None:
        super(ResNet101, self).__init__()
        if output_stride == 16:
            replace_stride_with_dilation = [False, False, True]
        elif output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
        else:
            raise NotImplementedError('Wrong output_stride.')

        resnet101 = torchvision.models.resnet101(pretrained=True,
                                                 replace_stride_with_dilation=replace_stride_with_dilation)
        self.conv1 = resnet101.conv1
        self.bn1 = resnet101.bn1
        self.relu = resnet101.relu
        self.maxpool = resnet101.maxpool
        self.layer1 = resnet101.layer1
        self.layer2 = resnet101.layer2
        self.layer3 = resnet101.layer3
        self.layer4 = resnet101.layer4

        self.low_level_feature = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        self.low_level_feature.append(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet101(output_stride=16).to(device)
    model.eval()

    torchsummary.torchsummary.summary(model, (3, 400, 800))
    macs, params = ptflops.get_model_complexity_info(model, (3, 400, 800), print_per_layer_stat=False, as_strings=False)
    print(f'GFLOPs: {macs / 1000000000 * 2}, params: {params}')

    writer = torch.utils.tensorboard.SummaryWriter('../../runs')
    writer.add_graph(model, torch.rand(1, 3, 400, 800).to(device))
    writer.close()
