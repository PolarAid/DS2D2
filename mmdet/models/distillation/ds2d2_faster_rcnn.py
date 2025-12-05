# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from pathlib import Path
from typing import List, Tuple, Union, Optional

import torch
import torch.nn as nn
from torch import Tensor
from pytorch_wavelets import DWTForward, DWTInverse
from ..losses.utils import weighted_loss
from ..utils import unpack_gt_instances

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.models.losses.utils import weight_reduce_loss
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmcv.cnn import build_conv_layer
from .base import BaseDetector
import torch.nn.functional as F

@weighted_loss
def kd_kl_div_loss(pred, soft_label, T, class_reduction='mean', detach_target=True):
    assert pred.size() == soft_label.size()
    target = F.softmax(soft_label / T, dim=1)
    if detach_target:
        target = target.detach()

    kd_loss = F.kl_div(
        F.log_softmax(pred / T, dim=1), target, reduction='none')
    if class_reduction == 'mean':
        kd_loss = kd_loss.mean(1)
    elif class_reduction == 'sum':
        kd_loss = kd_loss.sum(1)
    else:
        raise NotImplementedError
    kd_loss = kd_loss * (T * T)
    return kd_loss

@MODELS.register_module()
class KD_KLDivLoss(nn.Module):
    def __init__(self,
                 class_reduction='mean',
                 reduction='mean',
                 loss_weight=1.0,
                 T=10):
        super(KD_KLDivLoss, self).__init__()
        assert T >= 1
        self.class_reduction = class_reduction
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T

    def forward(self,
                pred,
                soft_label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_kd = self.loss_weight * kd_kl_div_loss(
            pred,
            soft_label,
            weight,
            class_reduction=self.class_reduction,
            reduction=reduction,
            avg_factor=avg_factor,
            T=self.T)

        return loss_kd
    
class DS2D2(nn.Module):
    def __init__(self, c, dist_cfg):
        super().__init__()
        self.conv = build_conv_layer(dict(type='Conv2d'), c, c, 1, padding=0, bias=False)
        self.xfm = DWTForward(J=1, mode='zero',wave='haar')
        self.ixfm = DWTInverse(mode='zero',wave='haar')
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
            cL_t, cH_t = self.xfm(tea_feat)
            cL_s, cH_s = self.xfm(stu_feat)
            
            adapt_stu_feat = self.conv(stu_feat)
            cL_s, cH_s = self.xfm(adapt_stu_feat)

            fgL_mask, bgL_mask, fgL_weight = self.transform_bboxes(gt_bboxes, src_size, cL_s.shape, cL_s.device)
            fgH_mask, bgH_mask, fgH_weight = fgL_mask.unsqueeze(2), bgL_mask.unsqueeze(2), fgL_weight.unsqueeze(2)
            distill_cL_loss += torch.mean((cL_s - cL_t)**2 * fgL_weight)
            distill_cH_loss += torch.mean((cH_s[0] - cH_t[0])**2 * fgH_weight)
            
        return distill_cL_loss, distill_cH_loss
    
    def implicit(self, stu_feats, tea_feats, roi_head, tea_head, rpn_results_list, batch_data_samples):
        batch_gt_instances, batch_gt_instances_ignore, _ = unpack_gt_instances(batch_data_samples)

        sampling_results = []
        for i in range(len(batch_data_samples)):
            rpn_results = rpn_results_list[i]

            assign_result = roi_head.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = roi_head.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in stu_feats])
            sampling_results.append(sampling_result)

        rois = bbox2roi([res.priors for res in sampling_results])

        stu_results = tea_head._bbox_forward(stu_feats, rois)
        tea_results = tea_head._bbox_forward(tea_feats, rois)
        stu_cls, stu_reg, tea_cls, tea_reg = stu_results['cls_score'], stu_results['bbox_pred'], tea_results['cls_score'], tea_results['bbox_pred']

        avg_factor = sum([res.avg_factor for res in sampling_results])
        num_classes = roi_head.bbox_head.num_classes
        
        distill_reg, distill_cls = 0, 0
        distill_cls += self.loss_cls_kd(stu_cls, tea_cls, avg_factor)

        tea_cls = tea_cls.softmax(dim=1)[:, :num_classes]
        reg_weights, reg_distill_idx = tea_cls.max(dim=1)
        if not roi_head.bbox_head.reg_class_agnostic:
            reg_distill_idx = reg_distill_idx[:, None, None].repeat(1, 1, 4)
            stu_reg = stu_reg.reshape(-1, num_classes, 4)
            stu_reg = stu_reg.gather(dim=1, index=reg_distill_idx)
            stu_reg = stu_reg.squeeze(1)
            tea_reg = tea_reg.reshape(-1, num_classes, 4)
            tea_reg = tea_reg.gather(dim=1, index=reg_distill_idx)
            tea_reg = tea_reg.squeeze(1)
        distill_reg += self.loss_reg_kd(stu_reg, tea_reg, reg_weights[:, None], reg_weights.sum() * 4)
            
        return distill_reg + distill_cls
    
@MODELS.register_module()
class DS2D2_FasterRCNN(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone: ConfigType,
                 teacher_config: Union[ConfigType, str, Path],
                 amplifier_config: Union[ConfigType, str, Path],
                 teacher_weights: Optional[str] = None,
                 amplifier_weights: Optional[str] = None,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
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
            self.high_amplifier = self.high_amplifier.roi_head
            self.freeze(self.high_amplifier)
            self.high_amplifier.is_init = True
            
        if neck is not None:
            self.neck = MODELS.build(neck)
            self.distloss = DS2D2(neck['out_channels'], dist_cfg)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            rpn_head_num_classes = rpn_head_.get('num_classes', None)
            if rpn_head_num_classes is None:
                rpn_head_.update(num_classes=1)
            else:
                if rpn_head_num_classes != 1:
                    warnings.warn(
                        'The `num_classes` should be 1 in RPN, but get '
                        f'{rpn_head_num_classes}, please set '
                        'rpn_head.num_classes = 1 in your config file.')
                    rpn_head_.update(num_classes=1)
            self.rpn_head = MODELS.build(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = MODELS.build(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.dist_cfg = dist_cfg

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading single-stage
        weights into two-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) != 0 and len(rpn_head_keys) == 0:
            for bbox_head_key in bbox_head_keys:
                rpn_head_key = rpn_head_prefix + \
                               bbox_head_key[len(bbox_head_prefix):]
                state_dict[rpn_head_key] = state_dict.pop(bbox_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    @property
    def with_rpn(self) -> bool:
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self) -> bool:
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

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

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        results = ()
        x = self.extract_feat(batch_inputs)

        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        roi_outs = self.roi_head.forward(x, rpn_results_list,
                                         batch_data_samples)
        results = results + (roi_outs, )
        return results
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        x = self.extract_feat(batch_inputs)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

        # Distillation Loss
        if hasattr(self, 'teacher'):
            distloss = 0
            tea_feats = self.teacher.extract_feat(batch_inputs)
            gt_bboxes = [x.gt_instances.bboxes for x in batch_data_samples]

            distloss_explicit_cL, distloss_explicit_cH = self.distloss.explicit(x, tea_feats, gt_bboxes, batch_inputs.shape)
            distloss += distloss_explicit_cL * self.dist_cfg['distloss_explicit_cL'] + distloss_explicit_cH * self.dist_cfg['distloss_explicit_cH']
            
            distloss_implicit_cL = self.distloss.implicit(x, tea_feats, self.roi_head, self.teacher.roi_head, rpn_results_list, batch_data_samples)
            distloss += distloss_implicit_cL * self.dist_cfg['distloss_implicit_cL']
            
            high_stu_feats, high_tea_feats = self.distloss.high_freq(x), self.distloss.high_freq(tea_feats)
            distloss_implicit_cH = self.distloss.implicit(high_stu_feats, high_tea_feats, self.roi_head, self.high_amplifier, rpn_results_list, batch_data_samples)
            distloss += distloss_implicit_cH * self.dist_cfg['distloss_implicit_cH']

            losses.update({'distloss': distloss})

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
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(batch_inputs)
        
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
