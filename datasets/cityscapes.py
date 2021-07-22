from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torchvision


class Cityscapes(torchvision.datasets.Cityscapes):
    def __init__(self,
                 root: str,
                 split: str = "train",
                 mode: str = "fine",
                 target_type: Union[List[str], str] = "instance",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None):
        super(Cityscapes, self).__init__(root, split, mode, target_type, transform, target_transform, transforms)
        self.class_names, self.test_ids, self.colors, \
            self.num_classes, self.ignore_index, self.ignore_color = self._get_class_info()
        self.images.sort()
        self.targets.sort()

    def _get_class_info(self):
        class_names = []
        test_ids = []
        colors = []
        num_classes = 0
        ignore_index = None
        ignore_color = None

        for cityscapes_class in self.classes:
            if cityscapes_class.name == 'unlabeled':
                ignore_index = cityscapes_class.train_id
                ignore_color = cityscapes_class.color
            if not cityscapes_class.ignore_in_eval:
                class_names.append(cityscapes_class.name)
                test_ids.append(cityscapes_class.id)
                colors.append(cityscapes_class.color)
                num_classes += 1

        colors = torch.as_tensor(colors)
        ignore_color = torch.as_tensor(ignore_color)
        return class_names, test_ids, colors, num_classes, ignore_index, ignore_color

    def decode_segmap_to_test_id(self, masks: torch.Tensor):
        for i in range(self.num_classes):
            masks[masks == i] = self.test_ids[i]
        masks = masks.to(torch.float32)
        masks /= 255
        return masks
