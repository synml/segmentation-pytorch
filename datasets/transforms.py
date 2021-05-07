import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F


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
                elif k == 'RandomCrop':
                    compose_items.append(RandomCrop(v['size']))
                elif k == 'RandomHorizontalFlip':
                    compose_items.append(RandomHorizontalFlip())
                elif k == 'Resize':
                    compose_items.append(Resize(v['size']))
                else:
                    raise NotImplementedError('Wrong augmentation.')
            self.augmentation = torchvision.transforms.Compose(compose_items)

    def __call__(self, image, target):
        to_tensor_normalize = torchvision.transforms.Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        data = {'image': image, 'target': target}

        if self.augmentation is not None:
            data = self.augmentation(data)

        data = to_tensor_normalize(data)
        return data['image'], data['target']


class ColorJitter(torchvision.transforms.ColorJitter):
    def __init__(self, brightness: float, contrast: float, saturation: float, hue: float):
        super(ColorJitter, self).__init__(brightness, contrast, saturation, hue)

    def forward(self, data: dict):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                data['image'] = F.adjust_brightness(data['image'], brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                data['image'] = F.adjust_contrast(data['image'], contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                data['image'] = F.adjust_saturation(data['image'], saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                data['image'] = F.adjust_hue(data['image'], hue_factor)
        return data


class RandomCrop(torchvision.transforms.RandomCrop):
    def __init__(self, size: tuple[int, int]):
        super().__init__(size)

    def forward(self, data: dict):
        i, j, h, w = self.get_params(data['image'], self.size)
        data['image'] = F.crop(data['image'], i, j, h, w)
        data['target'] = F.crop(data['target'], i, j, h, w)
        return data


class RandomHorizontalFlip(torchvision.transforms.RandomHorizontalFlip):
    def __init__(self):
        super().__init__()

    def forward(self, data: dict):
        if torch.rand(1) < self.p:
            data['image'] = F.hflip(data['image'])
            data['target'] = F.hflip(data['target'])
        return data


class Resize(torchvision.transforms.Resize):
    def __init__(self, size: tuple[int, int]):
        super(Resize, self).__init__(size)

    def forward(self, data: dict):
        data['image'] = F.resize(data['image'], self.size, F.InterpolationMode.BILINEAR)
        data['target'] = F.resize(data['target'], self.size, F.InterpolationMode.NEAREST)
        return data


class Normalize(torchvision.transforms.Normalize):
    def __init__(self, mean, std):
        super(Normalize, self).__init__(mean, std)

    def forward(self, data: dict):
        data['image'] = F.normalize(data['image'], self.mean, self.std)
        return data


class ToTensor(torchvision.transforms.ToTensor):
    def __call__(self, data: dict):
        data['image'] = F.to_tensor(data['image'])
        data['target'] = torch.as_tensor(np.array(data['target']), dtype=torch.int64)
        return data
