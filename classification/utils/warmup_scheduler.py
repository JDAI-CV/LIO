import torch
from bisect import bisect_right


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epoch, milestones, gamma=0.1, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = [warmup_epoch + e for e in milestones]
        self.gamma = gamma
        self.warmup_epoch = warmup_epoch
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epoch:
            return [base_lr * 0.1 for base_lr in self.base_lrs]
        else:
            return [base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                    for base_lr in self.base_lrs]

