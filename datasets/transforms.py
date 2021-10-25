from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF


class Transforms:
    def __init__(self, cfg: dict, augmentation=False):
        if not augmentation:
            self.augmentation = None
        else:
            cfg_augmentation: dict = cfg[cfg['model']['name']]['augmentation']
            compose_items = []
            for k, v in cfg_augmentation.items():
                if k == 'ColorJitter':
                    compose_items.append(ColorJitter(v['brightness'], v['contrast'], v['saturation'], v['hue']))
                elif k == 'GaussianBlur':
                    compose_items.append(GaussianBlur(v['kernel_size'], v['sigma']))
                elif k == 'RandomAdjustSharpness':
                    compose_items.append(RandomAdjustSharpness(v['sharpness_factor']))
                elif k == 'RandomCrop':
                    compose_items.append(RandomCrop(v['size']))
                elif k == 'RandomHorizontalFlip':
                    compose_items.append(RandomHorizontalFlip())
                elif k == 'RandomResize':
                    compose_items.append(RandomResize(v['min_scale'], v['max_scale']))
                elif k == 'RandomResizedCrop':
                    compose_items.append(RandomResizedCrop(v['size'], v['scale'], v['ratio']))
                elif k == 'Resize':
                    compose_items.append(Resize(v['size']))
                else:
                    raise NotImplementedError('Wrong augmentation.')
            self.augmentation = torchvision.transforms.Compose(compose_items)

        self.to_tensor = ToTensor()
        self.normalize = Normalize(cfg['dataset']['normalize_mean'], cfg['dataset']['normalize_std'])

    def __call__(self, image, target):
        data = {'image': image, 'target': target}

        data = self.to_tensor(data)
        if self.augmentation is not None:
            data = self.augmentation(data)
        data = self.normalize(data)

        return data['image'], data['target']


class ColorJitter(torchvision.transforms.ColorJitter):
    def __init__(self, brightness: float, contrast: float, saturation: float, hue: float):
        super().__init__(brightness, contrast, saturation, hue)

    def forward(self, data: dict):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                data['image'] = TF.adjust_brightness(data['image'], brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                data['image'] = TF.adjust_contrast(data['image'], contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                data['image'] = TF.adjust_saturation(data['image'], saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                data['image'] = TF.adjust_hue(data['image'], hue_factor)
        return data


class GaussianBlur(torchvision.transforms.GaussianBlur):
    def __init__(self, kernel_size: int, sigma=tuple[float, float]):
        super().__init__(kernel_size, sigma)

    def forward(self, data: dict):
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        data['image'] = TF.gaussian_blur(data['image'], self.kernel_size, [sigma, sigma])
        return data


class RandomAdjustSharpness(torchvision.transforms.RandomAdjustSharpness):
    def __init__(self, sharpness_factor: float):
        super(RandomAdjustSharpness, self).__init__(sharpness_factor)

    def forward(self, data: dict):
        if torch.rand(1).item() < self.p:
            data['image'] = TF.adjust_sharpness(data['image'], self.sharpness_factor)
        return data


class RandomCrop(torchvision.transforms.RandomCrop):
    def __init__(self, size: tuple[int, int]):
        super().__init__(size)

    def forward(self, data: dict):
        i, j, h, w = self.get_params(data['image'], self.size)
        data['image'] = TF.crop(data['image'], i, j, h, w)
        data['target'] = TF.crop(data['target'], i, j, h, w)
        return data


class RandomHorizontalFlip(torchvision.transforms.RandomHorizontalFlip):
    def __init__(self):
        super().__init__()

    def forward(self, data: dict):
        if torch.rand(1) < self.p:
            data['image'] = TF.hflip(data['image'])
            data['target'] = TF.hflip(data['target'])
        return data


# Equal to RandomScale
class RandomResize(nn.Module):
    def __init__(self, min_scale: float, max_scale: float):
        super(RandomResize, self).__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale

    def forward(self, data: dict):
        scale = torch.empty(1).uniform_(self.min_scale, self.max_scale)
        size = torch.round(torch.as_tensor(data['image'].shape[-2:]) * scale).to(dtype=torch.int).tolist()

        data['target'].unsqueeze_(dim=0)

        data['image'] = TF.resize(data['image'], size, TF.InterpolationMode.BILINEAR, antialias=True)
        data['target'] = TF.resize(data['target'], size, TF.InterpolationMode.NEAREST)

        data['target'].squeeze_(dim=0)
        return data


class RandomResizedCrop(torchvision.transforms.RandomResizedCrop):
    """
    1. ratio로 종횡비를 조절
    2. scale로 면적의 일정 비율을 자름 (비율 단위: 면적, 길이가 아님!)
    3. size로 최종 출력 크기 조절
    Args:
        size: 최종 출력 크기
        scale: 원본 이미지에서 자를 면적의 비율 (auto = (원본 이미지에 대한 size의 crop 비율, 1.0))
        ratio: 자를 영역의 종횡비 (auto = size의 종횡비로 일치시킴)
    """

    def __init__(
            self, size: Union[int, Sequence], scale: Union[tuple[float, float], str],
            ratio: Union[tuple[float, float], str]
    ):
        if ratio == 'auto':
            ratio = (size[1] / size[0], size[1] / size[0])

        super().__init__(size, scale, ratio)

    def forward(self, data: dict):
        if self.scale == 'auto':
            self.scale = ((self.size[0] * self.size[1]) / (data['target'].size()[-2] * data['target'].size()[-1]), 1.0)

        data['target'].unsqueeze_(dim=0)

        i, j, h, w = self.get_params(data['image'], self.scale, self.ratio)
        data['image'] = TF.resized_crop(data['image'], i, j, h, w, self.size, TF.InterpolationMode.BILINEAR)
        data['target'] = TF.resized_crop(data['target'], i, j, h, w, self.size, TF.InterpolationMode.NEAREST)

        data['target'].squeeze_(dim=0)
        return data


class Resize(torchvision.transforms.Resize):
    def __init__(self, size: tuple[int, int]):
        super().__init__(size)

    def forward(self, data: dict):
        data['target'].unsqueeze_(dim=0)

        data['image'] = TF.resize(data['image'], self.size, TF.InterpolationMode.BILINEAR, antialias=True)
        data['target'] = TF.resize(data['target'], self.size, TF.InterpolationMode.NEAREST)

        data['target'].squeeze_(dim=0)
        return data


class Normalize(torchvision.transforms.Normalize):
    def __init__(self, mean: Sequence, std: Sequence):
        super().__init__(mean, std)

    def forward(self, data: dict):
        data['image'] = TF.normalize(data['image'], self.mean, self.std)
        return data


class ToTensor(torchvision.transforms.ToTensor):
    def __call__(self, data: dict):
        data['image'] = TF.to_tensor(data['image'])
        data['target'] = torch.as_tensor(np.array(data['target']), dtype=torch.int64)
        return data
