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
            for key, value in cfg_augmentation.items():
                if key == 'RandomHorizontalFlip':
                    compose_items.append(RandomHorizontalFlip())
                elif key == 'RandomResizedCrop':
                    compose_items.append(RandomResizedCrop(value['size'], value['scale'], value['ratio']))
                elif key == 'Resize':
                    compose_items.append(Resize(value['size']))
                else:
                    raise NotImplementedError('Wrong augmentation.')
            self.augmentation = torchvision.transforms.Compose(compose_items)

    def __call__(self, image, target):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        target_transform = torchvision.transforms.ToTensor()

        image = transform(image)
        target = target_transform(target)

        if self.augmentation is not None:
            data = {'image': image, 'target': target}
            data = self.augmentation(data)
            return data['image'], data['target']
        else:
            return image, target


class RandomHorizontalFlip(torchvision.transforms.RandomHorizontalFlip):
    def __init__(self):
        super().__init__()

    def forward(self, data: dict):
        if torch.rand(1) < self.p:
            data['image'] = F.hflip(data['image'])
            data['target'] = F.hflip(data['target'])
        return data


class RandomResizedCrop(torchvision.transforms.RandomResizedCrop):
    def __init__(self, size: tuple[int, int], scale: tuple[float, float], ratio: tuple[float, float]):
        super().__init__(size, scale, ratio)

    def forward(self, data: dict):
        i, j, h, w = self.get_params(data['image'], self.scale, self.ratio)

        data['image'] = F.resized_crop(data['image'], i, j, h, w, self.size, F.InterpolationMode.BILINEAR)
        data['target'] = F.resized_crop(data['target'], i, j, h, w, self.size, F.InterpolationMode.NEAREST)
        return data


class Resize(torchvision.transforms.Resize):
    def __init__(self, size: tuple[int, int]):
        super(Resize, self).__init__(size)

    def forward(self, data: dict):
        data['image'] = F.resize(data['image'], self.size, F.InterpolationMode.BILINEAR)
        data['target'] = F.resize(data['target'], self.size, F.InterpolationMode.NEAREST)
        return data
