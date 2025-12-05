# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union, Optional

import torch
from pathlib import Path
from torch import Tensor
import torch.nn as nn

from mmdet.registry import MODELS
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmcv.cnn import build_conv_layer, ConvModule
from .base import BaseDetector
from pytorch_wavelets import DWTForward, DWTInverse
from mmdet.models.losses.utils import weight_reduce_loss
from mmdet.models.utils import unpack_gt_instances
import torch.nn.functional as F

def kd_quality_focal_loss(pred, target, weight=None, beta=1, reduction='mean', avg_factor=None):
    num_classes = pred.size(1)
    if weight is not None:
        weight = weight[:, None].repeat(1, num_classes)
    target = target.detach().sigmoid()
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    focal_weight = torch.abs(pred.sigmoid() - target).pow(beta)
    loss = loss * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

@MODELS.register_module()
class KD_QualityFocalLoss(nn.Module):
    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(KD_QualityFocalLoss, self).__init__()
        self.beta, self.reduction, self.loss_weight = beta, reduction, loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * kd_quality_focal_loss(pred, target, weight, beta=self.beta, reduction=reduction, avg_factor=avg_factor)
        return loss

class DS2D2(nn.Module):
    def __init__(self, c, dist_cfg):
        super().__init__()
        self.conv = build_conv_layer(dict(type='Conv2d'), c, c, 1, padding=0, bias=False)
        self.xfm = DWTForward(J=1, mode='zero',wave='haar')
        self.loss_reg_kd = MODELS.build(dist_cfg['loss_reg_kd'])
        self.loss_cls_kd = MODELS.build(dist_cfg['loss_cls_kd'])

    def transform_bboxes(self, bboxes, src_size, dst_size, device):
        scale_h, scale_w = dst_size[2] / src_size[2], dst_size[3] / src_size[3]
        fg_mask = torch.zeros(dst_size, dtype=torch.float16)
        fg_weight = torch.ones(dst_size, dtype=torch.float16)
        for i in range(len(bboxes)):
            for bbox in bboxes[i]:
                x_min, y_min, x_max, y_max = bbox
                new_x_min = max(0, int(torch.floor(x_min * scale_w)))
                new_y_min = max(0, int(torch.floor(y_min * scale_h)))
                new_x_max = min(dst_size[3] - 1, int(torch.ceil(x_max * scale_w)))
                new_y_max = min(dst_size[2] - 1, int(torch.ceil(y_max * scale_h)))
                fg_mask[i, :, new_y_min:new_y_max+1, new_x_min:new_x_max+1] = 1.0
                area = 1.0 / (new_x_max - new_x_min + 1) / (new_y_max - new_y_min + 1)
                fg_weight[i, :, new_y_min:new_y_max+1, new_x_min:new_x_max+1] += area
        bg_mask = 1 - fg_mask
        return fg_mask.to(device), bg_mask.to(device), fg_weight.to(device)
    
    def high_freq(self, feats):
        cH_feats = []
        for feat in feats:
            _, cH = self.xfm(feat)
            cH_sum = cH[0].sum(dim=2)
            cH_feat = F.interpolate(cH_sum, size=feat.shape[2:], mode='bilinear', align_corners=False)
            cH_feats.append(cH_feat)
        return cH_feats
    
    def explicit(self, stu_feats, tea_feats, gt_bboxes, src_size):
        distill_cL_loss, distill_cH_loss = 0, 0
        for stu_feat, tea_feat in zip(stu_feats, tea_feats):
            adapt_stu_feat = self.conv(stu_feat)
            cL_t, cH_t = self.xfm(tea_feat)
            cL_s, cH_s = self.xfm(adapt_stu_feat)
            
            fgL_mask, bgL_mask, fgL_weight = self.transform_bboxes(gt_bboxes, src_size, cL_s.shape, cL_s.device)
            fgH_mask, bgH_mask, fgH_weight = fgL_mask.unsqueeze(2), bgL_mask.unsqueeze(2), fgL_weight.unsqueeze(2)

            distill_cL_loss += torch.mean((cL_s - cL_t)**2 * fgL_weight)
            distill_cH_loss += torch.mean((cH_s[0] - cH_t[0])**2 * fgH_weight)

        return distill_cL_loss, distill_cH_loss
    
    def implicit(self, stu_clses, stu_regs, tea_clses, tea_regs, bbox_head, batch_data_samples):
        cls_sizes = [stu_cls.size()[-2:] for stu_cls in stu_clses]
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = unpack_gt_instances(batch_data_samples)
        anchor_list, valid_flag_list = bbox_head.get_anchors(cls_sizes, batch_img_metas, device=stu_clses[0].device)
        results = bbox_head.get_targets(anchor_list, valid_flag_list, batch_gt_instances, batch_img_metas, batch_gt_instances_ignore=batch_gt_instances_ignore)
        _, label_weights_list, _, _, avg_factor = results
        multi_anchors = []
        for i in range(len(anchor_list[0])):
            multi_anchors.append(anchor_list[0][i].unsqueeze(0).repeat(stu_clses[0].shape[0], 1, 1))

        distill_reg, distill_cls = 0, 0
        for stu_cls, stu_reg, tea_cls, tea_reg, anchors, label_weights in zip(stu_clses, stu_regs, tea_clses, tea_regs, multi_anchors, label_weights_list):
            tea_cls = tea_cls.permute(0, 2, 3, 1).reshape(-1, bbox_head.cls_out_channels)
            stu_cls = stu_cls.permute(0, 2, 3, 1).reshape(-1, bbox_head.cls_out_channels)
            
            bbox_coder = bbox_head.bbox_coder
            tea_reg = tea_reg.permute(0, 2, 3, 1).reshape(-1, bbox_coder.encode_size)
            stu_reg = stu_reg.permute(0, 2, 3, 1).reshape(-1, bbox_coder.encode_size)
            anchors = anchors.reshape(-1, anchors.size(-1))
            tea_reg = bbox_coder.decode(anchors, tea_reg)
            stu_reg = bbox_coder.decode(anchors, stu_reg)
            reg_weights = tea_cls.max(dim=1)[0].sigmoid()
            label_weights = label_weights.reshape(-1)
            
            distill_reg += self.loss_reg_kd(stu_reg, tea_reg, reg_weights, avg_factor)
            distill_cls += self.loss_cls_kd(stu_cls, tea_cls, label_weights, avg_factor)
        return distill_reg + distill_cls

@MODELS.register_module()
class DS2D2_RetinaNet(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone: ConfigType,
                 teacher_config: Union[ConfigType, str, Path],
                 amplifier_config: Union[ConfigType, str, Path],
                 teacher_weights: Optional[str] = None,
                 amplifier_weights: Optional[str] = None,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 dist_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if isinstance(teacher_config, (str, Path)):
            teacher_config = Config.fromfile(teacher_config)
            self.teacher = MODELS.build(teacher_config['model'])
            if teacher_weights is not None:
                load_checkpoint(self.teacher, teacher_weights, map_location='cpu')
            self.freeze(self.teacher)
            self.teacher.is_init = True
        
        if isinstance(amplifier_config, (str, Path)):
            amplifier_config = Config.fromfile(amplifier_config)
            self.high_amplifier = MODELS.build(amplifier_config['model'])
            if amplifier_weights is not None:
                load_checkpoint(self.high_amplifier, amplifier_weights, map_location='cpu')
            self.high_amplifier = self.high_amplifier.bbox_head
            self.freeze(self.high_amplifier)
            self.high_amplifier.is_init = True

        if neck is not None:
            self.neck = MODELS.build(neck)
            self.distloss = DS2D2(neck['out_channels'], dist_cfg)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.dist_cfg = dist_cfg

    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading two-stage
        weights into single-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + \
                                rpn_head_key[len(rpn_head_prefix):]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)
        losses = dict()
        if hasattr(self, 'teacher'):
            distloss = 0
            tea_feats = self.teacher.extract_feat(batch_inputs)
            gt_bboxes = [x.gt_instances.bboxes for x in batch_data_samples]
            distloss_explicit_cL, distloss_explicit_cH = self.distloss.explicit(x, tea_feats, gt_bboxes, batch_inputs.shape)
            distloss += distloss_explicit_cL * self.dist_cfg['distloss_explicit_cL'] + distloss_explicit_cH * self.dist_cfg['distloss_explicit_cH']
            
            stu_cls, stu_reg = self.teacher.bbox_head(x)
            tea_cls, tea_reg = self.teacher.bbox_head(tea_feats)
            distloss_implicit_cL = self.distloss.implicit(stu_cls, stu_reg, tea_cls, tea_reg, self.bbox_head, batch_data_samples)
            distloss += distloss_implicit_cL * self.dist_cfg['distloss_implicit_cL']

            stu_cls, stu_reg = self.high_amplifier(self.distloss.high_freq(x))
            tea_cls, tea_reg = self.high_amplifier(self.distloss.high_freq(tea_feats))
            distloss_implicit_cH = self.distloss.implicit(stu_cls, stu_reg, tea_cls, tea_reg, self.bbox_head, batch_data_samples)
            distloss += distloss_implicit_cH * self.dist_cfg['distloss_implicit_cH']

            losses.update({'distloss': distloss})

        losses.update(self.bbox_head.loss(x, batch_data_samples))
        return losses
        
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return results

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x
