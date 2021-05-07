import matplotlib.pyplot as plt
import torch
import torchvision


# 데이터셋 불러오는 코드 검증 (Shape: [batch, channel, height, width])
def show_dataset(images: torch.Tensor, targets: torch.Tensor):
    def make_plt_subplot(nrows: int, ncols: int, index: int, title: str, image):
        plt.subplot(nrows, ncols, index)
        plt.title(title)
        plt.imshow(image)

    to_pil_image = torchvision.transforms.ToPILImage()
    plt.rcParams['figure.figsize'] = (17, 6)
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.labelleft'] = False

    assert images.shape[0] == targets.shape[0]
    for i in range(images.shape[0]):
        make_plt_subplot(1, 2, 1, 'Input image', to_pil_image(images[i].cpu()))
        make_plt_subplot(1, 2, 2, 'Groundtruth', targets[i].cpu())
        plt.show()
