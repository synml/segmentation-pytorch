from collections import namedtuple
import random
from typing import Any, Callable, List, Optional, Union, Tuple

from PIL import Image
import torchvision
import torchvision.transforms.functional


class Cityscapes(torchvision.datasets.Cityscapes):
    class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                   'traffic light', 'traffic sign', 'vegetation', 'terrain',
                   'sky', 'person', 'rider', 'car', 'truck',
                   'bus', 'train', 'motorcycle', 'bicycle']

    def __init__(self,
                 root: str,
                 split: str = "train",
                 mode: str = "fine",
                 target_type: Union[List[str], str] = "instance",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None) -> None:
        super().__init__(root, split, mode, target_type, transform, target_transform, transforms)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert('RGB')

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        # Random horizontal flip
        if self.split == 'train' and random.random() > 0.5:
            image = torchvision.transforms.functional.hflip(image)
            target = torchvision.transforms.functional.hflip(target)

        return image, target

    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_customLabelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            return '{}_polygons.json'.format(mode)


class VOCSegmentation(torchvision.datasets.VOCSegmentation):
    VOCClasses = namedtuple('VOCClasses', ['name', 'classId', 'color'])
    classes = [
        VOCClasses('__background__', 0, (0, 0, 0)),
        VOCClasses('aeroplane', 1, (128, 0, 0)),
        VOCClasses('bicycle', 2, (0, 128, 0)),
        VOCClasses('bird', 3, (128, 128, 0)),
        VOCClasses('boat', 4, (0, 0, 128)),
        VOCClasses('bottle', 5, (128, 0, 128)),
        VOCClasses('bus', 6, (0, 128, 128)),
        VOCClasses('car', 7, (128, 128, 128)),
        VOCClasses('cat', 8, (64, 0, 0)),
        VOCClasses('chair', 9, (192, 0, 0)),
        VOCClasses('cow', 10, (64, 128, 0)),
        VOCClasses('diningtable', 11, (192, 128, 0)),
        VOCClasses('dog', 12, (64, 0, 128)),
        VOCClasses('horse', 13, (192, 0, 128)),
        VOCClasses('motorbike', 14, (64, 128, 128)),
        VOCClasses('person', 15, (192, 128, 128)),
        VOCClasses('pottedplant', 16, (0, 64, 0)),
        VOCClasses('sheep', 17, (128, 64, 0)),
        VOCClasses('sofa', 18, (0, 192, 0)),
        VOCClasses('train', 19, (128, 192, 0)),
        VOCClasses('tvmonitor', 20, (0, 64, 128)),
    ]

    def __init__(
            self,
            root: str,
            year: str = "2012",
            image_set: str = "train",
            download: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ):
        super(VOCSegmentation, self).__init__(root, year, image_set, download, transform, target_transform, transforms)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # Random horizontal flip
        if self.image_set == 'trainval' and random.random() > 0.5:
            img = torchvision.transforms.functional.hflip(img)
            target = torchvision.transforms.functional.hflip(target)

        return img, target
