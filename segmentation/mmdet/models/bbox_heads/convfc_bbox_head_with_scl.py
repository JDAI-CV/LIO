import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from ..losses import accuracy

from ..registry import HEADS
from ..utils import ConvModule
from .bbox_head import BBoxHead
from .convfc_bbox_head import ConvFCBBoxHead

from ..builder import build_loss
from .scl_module import SCLModule
from mmdet.core import (auto_fp16, bbox_target, delta2bbox, force_fp32,
                        multiclass_nms)


@HEADS.register_module
class ConvFCBBoxHeadWithSCL(ConvFCBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 with_scl=True,
                 structure_dim=128,
                 loss_scl=dict(
                     type='BareLoss',
                     loss_weight=1.0),
                 *args,
                 **kwargs):
        super(ConvFCBBoxHeadWithSCL, self).__init__(*args, **kwargs)
        self.structure_dim = structure_dim
        self.scl = SCLModule(size=self.roi_feat_size,
                             feature_dim=self.in_channels,
                             structure_dim=self.structure_dim)
        self.loss_scl = build_loss(loss_scl)

    def forward(self, x):
        base_result = super().forward(x)
        scl_result = self.scl(x)
        return base_result, scl_result

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             scl_result,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            losses['loss_cls'] = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               4)[pos_inds, labels[pos_inds]]
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)

        losses['loss_scl'] = self.loss_scl(scl_result)
        return losses

@HEADS.register_module
class SharedFCBBoxHeadWithSCL(ConvFCBBoxHeadWithSCL):
    def __init__(self, num_fcs=2, fc_out_channels=1024, 
                 with_scl=True,
                 structure_dim=128,
                 loss_scl=dict(
                     type='BareLoss',
                     loss_weight=1.0),
                 *args, **kwargs):
        assert num_fcs >= 1
        super(SharedFCBBoxHeadWithSCL, self).__init__(
            with_scl=with_scl,
            structure_dim=structure_dim,
            loss_scl=loss_scl,
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
