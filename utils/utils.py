import matplotlib.pyplot as plt
import torchvision


def make_plt_subplot(nrows: int, ncols: int, index: int, title: str, image):
    plt.subplot(nrows, ncols, index)
    plt.title(title)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])


def show_dataset(images, masks):
    # make subplot
    to_pil_image = torchvision.transforms.ToPILImage()
    make_plt_subplot(1, 2, 1, 'Input image', to_pil_image(images.cpu()))
    make_plt_subplot(1, 2, 2, 'Groundtruth', to_pil_image(masks.cpu()))

    # show plot
    plt.show()
