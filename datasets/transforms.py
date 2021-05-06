import torch
import torchvision
import torchvision.transforms.functional as F


class Transforms:
    def __init__(self, size: tuple[int, int], scale: tuple[float, float], ratio: tuple[float, float]):
        self.random_resized_crop = RandomResizedCrop(size, scale, ratio)

    def __call__(self, image, target):
        image = torchvision.transforms.ToTensor()(image)
        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        target = torchvision.transforms.ToTensor()(target)

        data = {'image': image, 'target': target}
        data = RandomHorizontalFlip()(data)
        data = self.random_resized_crop(data)
        return data['image'], data['target']


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
