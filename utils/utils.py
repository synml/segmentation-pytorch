import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision


def make_plt_subplot(nrows: int, ncols: int, index: int, title: str, image):
    plt.subplot(nrows, ncols, index)
    plt.title(title)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])


def show_dataset(images: torch.Tensor, masks: torch.Tensor):
    to_pil_image = torchvision.transforms.ToPILImage()

    for i in range(images.shape[0]):
        make_plt_subplot(1, 2, 1, 'Input image', to_pil_image(images[i].squeeze().cpu()))
        make_plt_subplot(1, 2, 2, 'Groundtruth', to_pil_image(masks[i].squeeze().cpu()))
        plt.show()


def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.normal_(m.bias)
    elif type(m) == nn.BatchNorm2d:
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
