# Copyright (c) GrokCV. All rights reserved.
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from mmcv.cnn import ConvModule, Scale
from mmengine.config import ConfigDict

from mmdet.structures import SampleList
from mmdet.utils import (ConfigType, InstanceList, PixelList, MultiConfig,
                         OptInstanceList, RangeType, reduce_mean)
from mmdet.models.dense_heads.fcos_head import FCOSHead
from mmdet.models.utils import multi_apply
from mmdet.models.utils import (select_single_mlvl)

from mmdet.registry import MODELS

@MODELS.register_module()
class FCOSDepthHead(FCOSHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (Sequence[int] or Sequence[Tuple[int, int]]): Strides of points
            in multiple feature levels. Defaults to (4, 8, 16, 32, 64).
        regress_ranges (Sequence[Tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling.
            Defaults to False.
        center_sample_radius (float): Radius of center sampling.
            Defaults to 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets with
            FPN strides. Defaults to False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Defaults to False.
        conv_bias (bool or str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Defaults to "auto".
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_centerness (:obj:`ConfigDict`, or dict): Config of centerness
            loss.
        norm_cfg (:obj:`ConfigDict` or dict): dictionary to construct and
            config norm layer.  Defaults to
            ``norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)``.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 seg_out_channels=1,
                 loss_seg: ConfigType = dict(
                     type='MSELoss',
                     loss_weight=1.0,
                     ignore_index=255),
                 **kwargs) -> None:
        self.seg_out_channels = seg_out_channels
        super().__init__(**kwargs)
        self.loss_seg = MODELS.build(loss_seg)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        super()._init_layers()
        self._init_seg_convs()
        self.conv_seg = nn.Conv2d(
            self.feat_channels, self.seg_out_channels, 3, padding=1)

    def _init_seg_convs(self) -> None:
        """Initialize classification conv layers of the head."""
        self.seg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.seg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        outs = self(x)

        outputs = unpack_gt_instances_sem_seg(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore, batch_gt_sem_seg,
         batch_img_metas) = outputs

        loss_inputs = outs + (batch_gt_instances, batch_gt_sem_seg,
                              batch_img_metas, batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def forward(
            self, x: Tuple[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of each level outputs.

            - cls_scores (list[Tensor]): Box scores for each scale level, \
            each is a 4D-tensor, the channel number is \
            num_points * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for each \
            scale level, each is a 4D-tensor, the channel number is \
            num_points * 4.
            - centernesses (list[Tensor]): centerness for each scale level, \
            each is a 4D-tensor, the channel number is num_points * 1.
            - seg_logits (list[Tensor]): segmentation logits for each scale level, \
            each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, x, self.scales, self.strides)

    def forward_single(self, x: Tensor, scale: Scale,
                       stride: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj:`mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions, centerness
            predictions, segmentation scores of input feature maps.
        """
        seg_feat = x
        for seg_layer in self.seg_convs:
            seg_feat = seg_layer(seg_feat)
        seg_score = self.conv_seg(seg_feat)

        cls_score, bbox_pred, centerness = super().forward_single(x, scale, stride)

        return cls_score, bbox_pred, centerness, seg_score

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        centernesses: List[Tensor],
        seg_scores: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_gt_sem_seg: PixelList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> Dict[str, Tensor]:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses) == len(seg_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        labels, bbox_targets = self.get_targets(all_level_points,
                                                batch_gt_instances)
        seg_targets = self.get_seg_targets(cls_scores, batch_gt_sem_seg)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds, centerness, and seg_scores
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_seg_scores = [
            seg_score.permute(0, 2, 3, 1).reshape(-1, self.seg_out_channels)
            for seg_score in seg_scores
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_seg_scores = torch.cat(flatten_seg_scores)

        # flatten labels, bbox_targets, and seg_targets
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_seg_targets = torch.cat(seg_targets)

        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        loss_seg = self.loss_seg(
            flatten_seg_scores, flatten_seg_targets)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_seg=loss_seg)

    def get_seg_targets(
        self,
        seg_scores: List[Tensor],
        batch_gt_sem_seg: InstanceList
    ) -> List[Tensor]:
        """
        Prepare segmentation targets.

        Args:
            seg_scores (List[Tensor]): List of segmentation scores of different
                levels in shape (batch_size, num_classes, h, w)
            batch_gt_instances (InstanceList): Ground truth instances.

        Returns:
            List[Tensor]: Segmentation targets of different levels.
        """
        lvls = len(seg_scores)
        batch_size = len(batch_gt_sem_seg)
        assert batch_size == seg_scores[0].size(0)
        seg_targets = []

        for lvl in range(lvls):
            _, _, h, w = seg_scores[lvl].shape
            lvl_seg_target = []
            for gt_sem_seg in batch_gt_sem_seg:
                lvl_seg_target.append(gt_sem_seg.sem_seg.data)
            lvl_seg_target = torch.stack(lvl_seg_target, dim=0)
            lvl_seg_target = F.interpolate(lvl_seg_target,
                                           size=(h, w), mode='nearest')
            seg_targets.append(lvl_seg_target)

        # flatten seg_targets
        flatten_seg_targets = [
            seg_target.permute(0, 2, 3, 1).reshape(-1, self.seg_out_channels)
            for seg_target in seg_targets
        ]

        return flatten_seg_targets

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        seg_scores: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds) == len(seg_scores)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]
            # seg_score_list = select_single_mlvl(
            #     seg_scores, img_id, detach=True)

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                score_factor_list=score_factor_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list


def unpack_gt_instances_sem_seg(batch_data_samples: SampleList) -> tuple:
    """Unpack ``gt_instances``, ``gt_instances_ignore``, ``img_metas``, and
    ``gt_sem_seg`` based on ``batch_data_samples``

    Args:
        batch_data_samples (List[:obj:`DetDataSample`]): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg`, `gt_sem_seg`, `gt_sem_seg`.

    Returns:
        tuple:

            - batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            - batch_gt_instances_ignore (list[:obj:`InstanceData`]):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            - batch_gt_sem_seg (list[:obj:`PixelData`]): Batch of gt_sem_seg.
                It includes ``semantic_seg`` and ``ignore_index`` attributes.
            - batch_img_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
    """
    batch_gt_instances = []
    batch_gt_instances_ignore = []
    batch_gt_sem_seg = []
    batch_img_metas = []
    for data_sample in batch_data_samples:
        batch_img_metas.append(data_sample.metainfo)
        batch_gt_instances.append(data_sample.gt_instances)
        if 'ignored_instances' in data_sample:
            batch_gt_instances_ignore.append(data_sample.ignored_instances)
        else:
            batch_gt_instances_ignore.append(None)
        batch_gt_sem_seg.append(data_sample.gt_sem_seg)

    return (batch_gt_instances, batch_gt_instances_ignore, batch_gt_sem_seg,
            batch_img_metas)