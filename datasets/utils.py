from typing import Union

import matplotlib.pyplot as plt
import torch
import torchvision


def decode_segmap_to_color_image(masks: torch.Tensor,
                                 colormap: Union[list, tuple],
                                 num_classes: int,
                                 ignore_index: int = None,
                                 ignore_color: Union[list, tuple] = None):
    # 각 채널 별로 디코딩하기 위해 복사
    r = masks.clone()
    g = masks.clone()
    b = masks.clone()

    # Assign colors according to class for each channel (각 채널 별로 class에 따라 색상 대입)
    for i in range(num_classes):
        r[masks == i] = colormap[i][0]
        g[masks == i] = colormap[i][1]
        b[masks == i] = colormap[i][2]
    if ignore_index and ignore_color is not None:
        r[masks == ignore_index] = ignore_color[0]
        g[masks == ignore_index] = ignore_color[1]
        b[masks == ignore_index] = ignore_color[2]

    decoded_masks = (r.unsqueeze(dim=1), g.unsqueeze(dim=1), b.unsqueeze(dim=1))
    decoded_masks = torch.cat(decoded_masks, dim=1).to(torch.float32)
    decoded_masks /= 255
    return decoded_masks


# Validate dataset loading code
def show_dataset(images: torch.Tensor, targets: torch.Tensor):
    to_pil_image = torchvision.transforms.ToPILImage()
    plt.rcParams['figure.figsize'] = (17, 6)
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.labelleft'] = False

    assert images.shape[0] == targets.shape[0]
    for i in range(images.shape[0]):
        fig, axs = plt.subplots(1, 2)
        axs[0].set_title('Input image')
        axs[0].imshow(to_pil_image(images[i].cpu()))
        axs[1].set_title('Groundtruth')
        axs[1].imshow(targets[i].cpu())
        plt.show()
