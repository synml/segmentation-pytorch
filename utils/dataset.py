import torchvision
from typing import Callable, List, Optional, Union

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

    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_customLabelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            return '{}_polygons.json'.format(mode)
