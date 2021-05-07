import platform

import numpy as np
import torch
import torch.utils.data
import torchvision

import datasets


class Cityscapes:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.class_names = ['road', 'sidewalk', 'building', 'wall', 'fence',
                            'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                            'sky', 'person', 'rider', 'car', 'truck',
                            'bus', 'train', 'motorcycle', 'bicycle', 'unlabeled']
        self.ignore_index = 255
        self.num_classes = 19
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

    # Cityscapes 데이터셋 설정
    def get_dataloader(self, split: str):
        root = self.cfg['dataset']['root']
        batch_size = self.cfg[self.cfg['model']['name']]['batch_size']
        if platform.system() == 'Windows':
            num_workers = 0
        else:
            num_workers = self.cfg['dataset']['num_workers']

        if split == 'train':
            dataset = torchvision.datasets.Cityscapes(root=root,
                                                      split='train',
                                                      mode='fine',
                                                      target_type='semantic',
                                                      transforms=datasets.transforms.Transforms(self.cfg,
                                                                                                augmentation=True))
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     num_workers=num_workers,
                                                     pin_memory=True)
        elif split == 'val':
            dataset = torchvision.datasets.Cityscapes(root=root,
                                                      split='val',
                                                      mode='fine',
                                                      target_type='semantic',
                                                      transforms=datasets.transforms.Transforms(self.cfg))
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     num_workers=num_workers)
        elif split == 'test':
            dataset = torchvision.datasets.Cityscapes(root=root,
                                                      split='test',
                                                      mode='fine',
                                                      target_type='semantic',
                                                      transforms=datasets.transforms.Transforms(self.cfg))
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     num_workers=num_workers)
        else:
            raise ValueError('Wrong split.')

        dataset.images.sort()
        dataset.targets.sort()
        return dataset, dataloader

    # Testset의 segmentation map을 labelID로 인코딩
    def encode_test_segmap(self, mask: np.ndarray):
        for i in range(19):
            mask[mask == i] = self.valid_classes[i]
        return mask

    # Cityscapes 데이터셋 라벨 색상 가져오기
    def get_cityscapes_colormap(self) -> np.ndarray:
        colormap = np.array([
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [0, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32]])
        return np.divide(colormap, 255).tolist()
