import torch
import torch.utils.tensorboard
import torchvision
import torchsummary


class ResNet101(torchvision.models.resnet.ResNet):
    def __init__(self, output_stride: int) -> None:
        if output_stride == 16:
            replace_stride_with_dilation = [False, False, True]
        elif output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
        else:
            raise NotImplementedError('Wrong output_stride.')

        super(ResNet101, self).__init__(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3],
                                        replace_stride_with_dilation=replace_stride_with_dilation)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feature = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feature


def load_resnet101(output_stride: int, pretrained: bool):
    model = ResNet101(output_stride)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(torchvision.models.resnet.model_urls['resnet101'])
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_resnet101(output_stride=8, pretrained=True).to(device)
    model.eval()

    torchsummary.torchsummary.summary(model, (3, 400, 800))

    writer = torch.utils.tensorboard.SummaryWriter('../../runs')
    writer.add_graph(model, torch.rand(1, 3, 400, 800).to(device))
    writer.close()
