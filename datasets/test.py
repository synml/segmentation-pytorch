import matplotlib.pyplot as plt
import torch
import torchvision


# 데이터셋 불러오는 코드 검증 (Shape: [batch, channel, height, width])
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
