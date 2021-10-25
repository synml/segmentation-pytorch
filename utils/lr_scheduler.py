import torch


class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, max_iter: int, power=0.9, min_lr=1e-7, last_epoch=-1):
        assert max_iter != 0

        self.max_iter = max_iter
        self.power = power
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.max_iter) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]
