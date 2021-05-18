from typing import Tuple

import numpy as np
import torch


class Evaluator:
    def __init__(self, num_classes: int):
        self.confusion_matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes

    def update_matrix(self, gt_batch: torch.Tensor, pred_batch: torch.Tensor):
        assert gt_batch.shape == pred_batch.shape
        self.confusion_matrix += self._generate_matrix(gt_batch.cpu().numpy(), pred_batch.cpu().numpy())

    def get_scores(self) -> Tuple[np.ndarray, np.float_]:
        iou = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) +
                                                self.confusion_matrix.sum(axis=1) -
                                                np.diag(self.confusion_matrix)) * 100
        miou = np.mean(iou)
        return iou, miou

    def _generate_matrix(self, gt_batch: torch.Tensor, pred_batch: torch.Tensor) -> np.ndarray:
        mask = (gt_batch >= 0) & (gt_batch < self.num_classes)
        label = self.num_classes * gt_batch[mask].astype('int') + pred_batch[mask]
        count = np.bincount(label, minlength=self.num_classes ** 2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix
