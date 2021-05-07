import matplotlib.pyplot as plt
import torch
import torchvision


# 데이터셋 불러오는 코드 검증 (Shape: [batch, channel, height, width])
def show_dataset(images: torch.Tensor, targets: torch.Tensor):
    def make_plt_subplot(nrows: int, ncols: int, index: int, title: str, image):
        plt.subplot(nrows, ncols, index)
        plt.title(title)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])

    to_pil_image = torchvision.transforms.ToPILImage()

    assert images.shape[0] == targets.shape[0]
    for i in range(images.shape[0]):
        make_plt_subplot(1, 2, 1, 'Input image', to_pil_image(images[i].squeeze().cpu()))
        make_plt_subplot(1, 2, 2, 'Groundtruth', to_pil_image(targets[i].cpu()))
        plt.show()
