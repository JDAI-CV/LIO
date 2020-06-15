import torch.nn as nn

from ..registry import LOSSES


@LOSSES.register_module
class BareLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
    
    def forward(self, pre_loss):
        loss = self.loss_weight * pre_loss.mean()
        return loss
