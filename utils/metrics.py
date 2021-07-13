from typing import Tuple

import torch


class Evaluator:
    def __init__(self, num_classes: int, device):
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
        self.num_classes = num_classes

    def update_matrix(self, gt_batch: torch.Tensor, pred_batch: torch.Tensor):
        assert gt_batch.shape == pred_batch.shape
        self.confusion_matrix += self._generate_matrix(gt_batch, pred_batch)

    def get_scores(self) -> Tuple[torch.Tensor, torch.Tensor]:
        iou = torch.diag(self.confusion_matrix) / (self.confusion_matrix.sum(dim=0) +
                                                   self.confusion_matrix.sum(dim=1) -
                                                   torch.diag(self.confusion_matrix)) * 100
        miou = torch.mean(iou)
        return iou, miou

    def _generate_matrix(self, gt_batch: torch.Tensor, pred_batch: torch.Tensor) -> torch.Tensor:
        mask = (gt_batch >= 0) & (gt_batch < self.num_classes)
        label = self.num_classes * gt_batch[mask] + pred_batch[mask]
        count = torch.bincount(label, minlength=self.num_classes ** 2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix
