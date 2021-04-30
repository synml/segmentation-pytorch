import math
import os
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms.functional
import yaml

import models.backbone.resnet34
import models.ar_unet
import models.deeplabv3plus
import models.unet


def load_config():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    config['dataset']['image_size'] = tuple([int(i) for i in config['dataset']['image_size'].split('x')])
    return config


def get_model(config: dict, pretrained=False, pretrained_backbone=False) -> torch.nn.Module:
    assert isinstance(pretrained, bool)
    assert config['dataset']['num_classes'] == 20 or config['dataset']['num_classes'] == 8

    if config['model'] == 'UNet':
        model = models.unet.UNet(config['dataset']['num_classes'])
    elif config['model'] == 'ResNet34':
        model = models.backbone.resnet34.ResNet34(config['dataset']['num_classes'])
    elif config['model'] == 'AR_UNet':
        if pretrained_backbone:
            model = models.ar_unet.AR_UNet(config['dataset']['num_classes'], config['Resnet34']['pretrained_weights'])
        else:
            model = models.ar_unet.AR_UNet(config['dataset']['num_classes'])
    elif config['model'] == 'DeepLabV3plus':
        model = models.deeplabv3plus.DeepLabV3plus('xception', 16, config['dataset']['num_classes'])
    else:
        raise NotImplementedError('Wrong model name.')

    if pretrained:
        if os.path.exists(config[config['model']]['pretrained_weights']):
            model.load_state_dict(torch.load(config[config['model']]['pretrained_weights']))
        else:
            print('FileNotFound: pretrained_weights (' + config['model'] + ')')
    return model


def get_criterion(config: dict) -> nn.Module:
    cfg_criterion: dict = config[config['model']]['criterion']

    if cfg_criterion['name'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError('Wrong criterion name.')

    return criterion


def get_optimizer(config: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    cfg_optim: dict = config[config['model']]['optimizer']

    if cfg_optim['name'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg_optim['lr'],
                                    momentum=cfg_optim['momentum'], weight_decay=cfg_optim['weight_decay'])
    elif cfg_optim['name'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg_optim['lr'], weight_decay=cfg_optim['weight_decay'])
    else:
        raise NotImplementedError('Wrong optimizer name.')

    return optimizer


def get_scheduler(config: dict, optimizer: torch.optim.Optimizer):
    cfg_scheduler: dict = config[config['model']]['scheduler']

    if cfg_scheduler['name'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg_scheduler['factor'],
                                                               patience=cfg_scheduler['patience'],
                                                               min_lr=cfg_scheduler['min_lr'])
    elif cfg_scheduler['name'] == 'WarmupPolyLR':
        scheduler = WarmupPolyLR(optimizer, config[config['model']]['iter'])
    else:
        raise NotImplementedError('Wrong scheduler name.')

    return scheduler


class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
        power: float = 0.9,
        constant_ending: float = 0.0,
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.power = power
        self.constant_ending = constant_ending
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = self._get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        if self.constant_ending > 0 and warmup_factor == 1.0:
            # Constant ending lr.
            if (
                math.pow((1.0 - self.last_epoch / self.max_iters), self.power) < self.constant_ending
            ):
                return [base_lr * self.constant_ending for base_lr in self.base_lrs]
        return [
            base_lr * warmup_factor * math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()

    def _get_warmup_factor_at_iter(self, method: str, iter: int, warmup_iters: int, warmup_factor: float) -> float:
        """
        Return the learning rate warmup factor at a specific iteration.
        See :paper:`ImageNet in 1h` for more details.
        Args:
            method (str): warmup method; either "constant" or "linear".
            iter (int): iteration at which to calculate the warmup factor.
            warmup_iters (int): the number of warmup iterations.
            warmup_factor (float): the base warmup factor (the meaning changes according
                to the method used).
        Returns:
            float: the effective warmup factor at the given iteration.
        """
        if iter >= warmup_iters:
            return 1.0

        if method == "constant":
            return warmup_factor
        elif method == "linear":
            alpha = iter / warmup_iters
            return warmup_factor * (1 - alpha) + alpha
        else:
            raise ValueError("Unknown warmup method: {}".format(method))


class Cityscapes:
    def __init__(self, config: dict):
        self.config = config
        self.num_classes = self.config['dataset']['num_classes']
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.config['dataset']['image_size']),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.target_transform = torchvision.transforms.Compose([
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

    # Cityscapes 데이터셋 클래스 이름 가져오기
    def get_class_names(self):
        if self.num_classes == 20:
            class_names = ['unlabeled', 'road', 'sidewalk', 'building', 'wall',
                           'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation',
                           'terrain', 'sky', 'person', 'rider', 'car',
                           'truck', 'bus', 'train', 'motorcycle', 'bicycle']
        elif self.num_classes == 8:
            class_names = ['unlabeled', 'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle']
        else:
            raise ValueError('Wrong num_classes.')

        return class_names

    # Cityscapes 데이터셋 라벨 색상 가져오기
    def get_cityscapes_colormap(self):
        if self.num_classes == 20:
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
        elif self.num_classes == 8:
            colormap = np.zeros((8, 3), dtype=np.uint8)
            colormap[0] = [0, 0, 0]
            colormap[1] = [128, 64, 128]
            colormap[2] = [70, 70, 70]
            colormap[3] = [250, 170, 30]
            colormap[4] = [107, 142, 35]
            colormap[5] = [70, 130, 180]
            colormap[6] = [220, 20, 60]
            colormap[7] = [0, 0, 142]
        else:
            raise ValueError('Wrong num_classes.')

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
        if torch.rand(1) < self.p:
            image = torchvision.transforms.functional.hflip(image)
            target = torchvision.transforms.functional.hflip(target)
        return image, target


def train_interupter():
    with open('train_interupter.ini', 'r', encoding='utf-8') as f:
        flag = f.read().strip()

    if flag == '0':
        return False
    elif flag == '1':
        with open('train_interupter.ini', 'w', encoding='utf-8') as f:
            f.write('0')
        return True
    else:
        raise ValueError('Wrong flag value.')


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
