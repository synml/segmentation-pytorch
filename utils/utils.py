import configparser
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torchvision

import models.unet
import models.proposed
import utils.datasets


# 설정 불러오기
def load_config():
    config_file = 'config.ini'
    parser = configparser.ConfigParser()
    parser.read(config_file, encoding='utf-8')

    model_name = parser['model']['activate_model']
    config = {
        'dataset_root': parser['dataset']['root'],
        'model_name': model_name,
        'batch_size': parser.getint(model_name, 'batch_size'),
        'epoch': parser.getint(model_name, 'epoch'),
        'image_size': (int(parser[model_name]['image_size'].split('x')[1]),
                       int(parser[model_name]['image_size'].split('x')[0])),
        'lr': parser.getfloat(model_name, 'lr'),
        'num_classes': parser.getint(model_name, 'num_classes'),
        'num_workers': parser.getint(model_name, 'num_workers'),
        'pretrained_weights': parser[model_name]['pretrained_weights'],
    }
    return config


# 모델 불러오기
def get_model(model_name: str, num_classes: int, pretrained: str = None) -> torch.nn.Module:
    if model_name == 'UNet':
        model = models.unet.UNet(num_classes)
    elif model_name == 'Proposed':
        model = models.proposed.Proposed(num_classes)
    else:
        raise NameError('Wrong model_name.')

    if pretrained is not None:
        if os.path.exists(pretrained):
            model.load_state_dict(torch.load(pretrained))
        else:
            print('FileNotFound: pretrained_weights')
    return model


# Cityscapes 데이터셋 라벨 색상 불러오기
def get_cityscapes_label_colormap(short=False):
    if not short:
        colormap = np.zeros((20, 3), dtype=np.uint8)
        colormap[0] = [0, 0, 0]
        colormap[1] = [128, 64, 128]
        colormap[2] = [244, 35, 232]
        colormap[3] = [70, 70, 70]
        colormap[4] = [102, 102, 156]
        colormap[5] = [190, 153, 153]
        colormap[6] = [153, 153, 153]
        colormap[7] = [250, 170, 30]
        colormap[8] = [220, 220, 0]
        colormap[9] = [107, 142, 35]
        colormap[10] = [152, 251, 152]
        colormap[11] = [70, 130, 180]
        colormap[12] = [220, 20, 60]
        colormap[13] = [255, 0, 0]
        colormap[14] = [0, 0, 142]
        colormap[15] = [0, 0, 70]
        colormap[16] = [0, 60, 100]
        colormap[17] = [0, 80, 100]
        colormap[18] = [0, 0, 230]
        colormap[19] = [119, 11, 32]
    else:
        colormap = np.zeros((8, 3), dtype=np.uint8)
        colormap[0] = [0, 0, 0]
        colormap[1] = [128, 64, 128]
        colormap[2] = [70, 70, 70]
        colormap[3] = [250, 170, 30]
        colormap[4] = [107, 142, 35]
        colormap[5] = [70, 130, 180]
        colormap[6] = [220, 20, 60]
        colormap[7] = [0, 0, 142]

    return np.divide(colormap, 255).tolist()


# Cityscapes 데이터셋 설정
def init_cityscapes_dataset(config: dict):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(config['image_size']),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(config['image_size'], interpolation=0),
        torchvision.transforms.ToTensor(),
    ])
    trainset = utils.datasets.Cityscapes(root=config['dataset_root'],
                                         split='train',
                                         mode='fine',
                                         target_type='semantic',
                                         transform=transform,
                                         target_transform=target_transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=config['batch_size'],
                                              shuffle=True,
                                              num_workers=config['num_workers'],
                                              pin_memory=True)
    testset = utils.datasets.Cityscapes(root=config['dataset_root'],
                                        split='val',
                                        mode='fine',
                                        target_type='semantic',
                                        transform=transform,
                                        target_transform=target_transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=config['batch_size'],
                                             shuffle=False,
                                             num_workers=config['num_workers'])

    return trainset, trainloader, testset, testloader


# 데이터셋 불러오는 코드 검증
def show_dataset(images: torch.Tensor, masks: torch.Tensor):
    def make_plt_subplot(nrows: int, ncols: int, index: int, title: str, image):
        plt.subplot(nrows, ncols, index)
        plt.title(title)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])

    to_pil_image = torchvision.transforms.ToPILImage()

    for i in range(images.shape[0]):
        make_plt_subplot(1, 2, 1, 'Input image', to_pil_image(images[i].squeeze().cpu()))
        make_plt_subplot(1, 2, 2, 'Groundtruth', to_pil_image(masks[i].squeeze().cpu()))
        plt.show()
