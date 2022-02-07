import torch


class Evaluator:
    def __init__(self, num_classes: int, device: torch.device):
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
        self.num_classes = num_classes

    def pixel_accuracy(self) -> torch.Tensor:
        acc = torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def pixel_accuracy_class(self) -> torch.Tensor:
        acc_cls = torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum(dim=0)
        acc_cls = torch.nanmean(acc_cls)
        return acc_cls

    def intersection_over_union(self, percent=False) -> torch.Tensor:
        iou = torch.diag(self.confusion_matrix) / (self.confusion_matrix.sum(dim=0) +
                                                   self.confusion_matrix.sum(dim=1) -
                                                   torch.diag(self.confusion_matrix))
        if percent:
            iou *= 100
        return iou

    def mean_intersection_over_union(self, percent=False) -> tuple[torch.Tensor, torch.Tensor]:
        iou = self.intersection_over_union(percent)
        miou = torch.nanmean(iou)
        return iou, miou

    def frequency_weighted_intersection_over_union(self) -> torch.Tensor:
        freq = torch.sum(self.confusion_matrix, dim=0) / torch.sum(self.confusion_matrix)
        iou = self.intersection_over_union()
        fwiou = (freq[freq > 0] * iou[freq > 0]).sum()
        return fwiou

    def _generate_matrix(self, gt_batch: torch.Tensor, pred_batch: torch.Tensor) -> torch.Tensor:
        mask = (gt_batch >= 0) & (gt_batch < self.num_classes)
        label = self.num_classes * gt_batch[mask] + pred_batch[mask]
        count = torch.bincount(label, minlength=self.num_classes ** 2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def update_matrix(self, gt_batch: torch.Tensor, pred_batch: torch.Tensor):
        assert gt_batch.shape == pred_batch.shape
        self.confusion_matrix += self._generate_matrix(gt_batch, pred_batch)
