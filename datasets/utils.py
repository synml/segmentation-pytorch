import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def decode_segmap_to_color_image(
        masks: torch.Tensor,
        colormap: np.ndarray,
        num_classes: int,
        ignore_index: int = None,
        ignore_color: np.ndarray = None):
    masks = masks.cpu().numpy()

    decoded_masks = []
    for mask in masks:
        r = mask.copy()
        g = mask.copy()
        b = mask.copy()

        # Assign colors according to class for each channel (각 채널 별로 class에 따라 색상 대입)
        for i in range(num_classes):
            r[mask == i] = colormap[i, 0]
            g[mask == i] = colormap[i, 1]
            b[mask == i] = colormap[i, 2]
        if ignore_index is not None:
            r[mask == ignore_index] = ignore_color[0]
            g[mask == ignore_index] = ignore_color[1]
            b[mask == ignore_index] = ignore_color[2]

        rgb = np.zeros((3, mask.shape[0], mask.shape[1]))
        rgb[0, :, :] = r / 255.0
        rgb[1, :, :] = g / 255.0
        rgb[2, :, :] = b / 255.0
        decoded_masks.append(rgb)
    decoded_masks = torch.from_numpy(np.array(decoded_masks))
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
