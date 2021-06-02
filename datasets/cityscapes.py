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
        self.class_names = ['road', 'sidewalk', 'building', 'wall', 'fence',
                            'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                            'sky', 'person', 'rider', 'car', 'truck',
                            'bus', 'train', 'motorcycle', 'bicycle', 'unlabeled']
        self.ignore_index = 255
        self.num_classes = 19
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.images.sort()
        self.targets.sort()

    def decode_segmap_to_test_id(self, masks: torch.Tensor):
        masks = masks.cpu().numpy()

        decoded_masks = []
        for mask in masks:
            for i in range(self.num_classes):
                mask[mask == i] = self.valid_classes[i]
            decoded_masks.append(mask)
        decoded_masks = torch.from_numpy(np.array(decoded_masks))
        return decoded_masks

    def get_colormap(self) -> np.ndarray:
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
            [70, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
            [0, 0, 0]   # ignore_index (255)
        ])
        return colormap
