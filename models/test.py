import torch
import torch.nn as nn
import torch.utils.tensorboard
import torchinfo


def test_model(model: nn.Module, input_size: tuple[int, int, int, int], graph_dir: str, device: torch.device):
    model.eval()

    model_statistics = torchinfo.summary(model, input_size, depth=10,
                                         col_names=('kernel_size', 'output_size', 'num_params', 'mult_adds'),
                                         row_settings=('depth', 'var_names'))
    print(f'Total GFLOPs: {model_statistics.total_mult_adds * 2 / 1e9:.4f}')

    writer = torch.utils.tensorboard.SummaryWriter(graph_dir)
    writer.add_graph(model, torch.rand(input_size).to(device))
    writer.close()
