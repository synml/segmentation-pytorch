import random
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from PIL import Image

import torchvision
import torchvision.transforms.functional

class_names = ['flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle']


class Cityscapes(torchvision.datasets.Cityscapes):
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
