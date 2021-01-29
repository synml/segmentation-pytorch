import os
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms.functional
import yaml

import models.backbone
import models.proposed
import models.unet


def load_config():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    config['dataset']['image_size'] = (int(config['dataset']['image_size'].split('x')[1]),
                                       int(config['dataset']['image_size'].split('x')[0]))

    return config


def get_model(config: dict, pretrained=False) -> torch.nn.Module:
    assert isinstance(pretrained, bool)

    if config['model'] == 'UNet':
        model = models.unet.UNet(config[config['model']]['num_classes'])
    elif config['model'] == 'Backbone':
        model = models.backbone.Backbone(config[config['model']]['num_classes'])
    elif config['model'] == 'Proposed':
        model = models.proposed.Proposed(config[config['model']]['num_classes'])
    else:
        raise NameError('Wrong model name.')

    if pretrained:
        if os.path.exists(config[config['model']]['pretrained_weights']):
            model.load_state_dict(torch.load(config[config['model']]['pretrained_weights']))
        else:
            print('FileNotFound: pretrained_weights (' + config['model'] + ')')
    return model


def get_optimizer(config: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    cfg_optim: dict = config[config['model']]['optimizer']

    if cfg_optim['name'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg_optim['lr'],
                                    momentum=cfg_optim['momentum'], weight_decay=cfg_optim['weight_decay'])
    elif cfg_optim['name'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg_optim['lr'], weight_decay=cfg_optim['weight_decay'])
    else:
        raise NameError('Wrong optimizer name.')

    return optimizer


def get_scheduler(config: dict, optimizer: torch.optim.Optimizer):
    cfg_scheduler: dict = config[config['model']]['scheduler']

    if cfg_scheduler['name'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg_scheduler['patience'],
                                                               min_lr=cfg_scheduler['min_lr'])
    else:
        raise NameError('Wrong scheduler name.')

    return scheduler


class Cityscapes:
    def __init__(self, config: dict):
        self.config = config
        self.class_names = ['unlabeled', 'road', 'sidewalk', 'building', 'wall',
                            'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation',
                            'terrain', 'sky', 'person', 'rider', 'car',
                            'truck', 'bus', 'train', 'motorcycle', 'bicycle']

        self.class_names_short = ['unlabeled', 'flat', 'construction', 'object',
                                  'nature', 'sky', 'human', 'vehicle']

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.config['dataset']['image_size']),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.target_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.config['dataset']['image_size'], interpolation=0),
            torchvision.transforms.ToTensor(),
        ])
        self.transforms = DataAugmentation(self.transform, self.target_transform)

    # Cityscapes 데이터셋 설정
    def set_cityscapes(self):
        trainset = torchvision.datasets.Cityscapes(root=self.config['dataset']['root'],
                                                   split='train',
                                                   mode='fine',
                                                   target_type='semantic',
                                                   transforms=self.transforms)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=self.config[self.config['model']]['batch_size'],
                                                  shuffle=True,
                                                  num_workers=self.config['dataset']['num_workers'],
                                                  pin_memory=True)
        testset = torchvision.datasets.Cityscapes(root=self.config['dataset']['root'],
                                                  split='val',
                                                  mode='fine',
                                                  target_type='semantic',
                                                  transform=self.transform,
                                                  target_transform=self.target_transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=self.config[self.config['model']]['batch_size'],
                                                 shuffle=False,
                                                 num_workers=self.config['dataset']['num_workers'])

        trainset.images.sort()
        trainset.targets.sort()
        testset.images.sort()
        testset.targets.sort()

        return trainset, trainloader, testset, testloader

    # Cityscapes 데이터셋 라벨 색상 불러오기
    def get_cityscapes_colormap(self, short=False):
        assert isinstance(short, bool)
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


class DataAugmentation:
    def __init__(self,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 p=0.5) -> None:
        self.transform = transform
        self.target_transform = target_transform
        self.p = p

    def __call__(self, image, target):
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # RandomHorizontalFlip
        if self.p < torch.rand(1):
            image = torchvision.transforms.functional.hflip(image)
            target = torchvision.transforms.functional.hflip(target)
        return image, target

    def __repr__(self):
        return self.__class__.__name__ + '()'


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
