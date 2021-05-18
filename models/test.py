from typing import Tuple

import ptflops
import torch
import torch.nn as nn
import torch.utils.tensorboard
import torchsummary


def test_model(model: nn.Module, input_size: Tuple[int, int, int], device):
    model.eval()

    torchsummary.torchsummary.summary(model, input_size)
    macs, params = ptflops.get_model_complexity_info(model, input_size, print_per_layer_stat=False, as_strings=False)
    print(f'GFLOPs: {macs / 1000000000 * 2}, params: {params}')

    writer = torch.utils.tensorboard.SummaryWriter('../runs')
    writer.add_graph(model, torch.unsqueeze(torch.rand(input_size), 0).to(device))
    writer.close()
