import matplotlib.pyplot as plt
import torch
import torchvision


# 데이터셋 불러오는 코드 검증 (Shape: [batch, channel, height, width])
def show_dataset(image: torch.Tensor, target: torch.Tensor):
    def make_plt_subplot(nrows: int, ncols: int, index: int, title: str, image):
        plt.subplot(nrows, ncols, index)
        plt.title(title)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])

    to_pil_image = torchvision.transforms.ToPILImage()

    assert image.shape[0] == target.shape[0]
    for i in range(image.shape[0]):
        make_plt_subplot(1, 2, 1, 'Input image', to_pil_image(image[i].squeeze().cpu()))
        make_plt_subplot(1, 2, 2, 'Groundtruth', to_pil_image(target[i].cpu()))
        plt.show()
