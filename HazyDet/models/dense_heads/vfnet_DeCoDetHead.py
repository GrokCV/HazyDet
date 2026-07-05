# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale
from mmcv.ops import DeformConv2d
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import bbox_overlaps
from mmdet.structures.det_data_sample import SampleList
from mmdet.utils import (ConfigType, InstanceList, MultiConfig, RangeType,PixelList, OptConfigType,
                         OptInstanceList, reduce_mean)
from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
from mmdet.models.task_modules.samplers import PseudoSampler
from mmdet.models.utils import multi_apply
from mmdet.models.dense_heads.atss_head import ATSSHead
from mmdet.models.dense_heads.fcos_head import FCOSHead
import torch.nn.functional as F

INF = 1e8


@MODELS.register_module()
class VFNetDeCoDetHead(ATSSHead, FCOSHead):
    """Head of `VarifocalNet (VFNet): An IoU-aware Dense Object
    Detector.<https://arxiv.org/abs/2008.13367>`_.

    The VFNet predicts IoU-aware classification scores which mix the
    object presence confidence and object localization accuracy as the
    detection score. It is built on the FCOS architecture and uses ATSS
    for defining positive/negative training examples. The VFNet is trained
    with Varifocal Loss and empolys star-shaped deformable convolution to
    extract features for a bbox.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        regress_ranges (Sequence[Tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Defaults to False.
        center_sample_radius (float): Radius of center sampling. Defaults to 1.5.
        sync_num_pos (bool): If true, synchronize the number of positive
            examples across GPUs. Defaults to True
        gradient_mul (float): The multiplier to gradients from bbox refinement
            and recognition. Defaults to 0.1.
        bbox_norm_type (str): The bbox normalization type, 'reg_denom' or
            'stride'. Defaults to reg_denom
        loss_cls_fl (:obj:`ConfigDict` or dict): Config of focal loss.
        use_vfl (bool): If true, use varifocal loss for training.
            Defaults to True.
        loss_cls (:obj:`ConfigDict` or dict): Config of varifocal loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss,
            GIoU Loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization
            refinement loss, GIoU Loss.
        norm_cfg (:obj:`ConfigDict` or dict): dictionary to construct and
            config norm layer. Defaults to norm_cfg=dict(type='GN',
            num_groups=32, requires_grad=True).
        use_atss (bool): If true, use ATSS to define positive/negative
            examples. Defaults to True.
        anchor_generator (:obj:`ConfigDict` or dict): Config of anchor
            generator for ATSS.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`]): Initialization config dict.

    Example:
        >>> self = VFNetHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, bbox_pred_refine= self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 regress_ranges: RangeType = ((-1, 64), (64, 128), (128, 256),
                                              (256, 512), (512, INF)),
                 center_sampling: bool = False,
                 center_sample_radius: float = 1.5,
                 sync_num_pos: bool = True,
                 gradient_mul: float = 0.1,
                 bbox_norm_type: str = 'reg_denom',
                 loss_cls_fl: ConfigType = dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 use_vfl: bool = True,
                 loss_cls: ConfigType = dict(
                     type='VarifocalLoss',
                     use_sigmoid=True,
                     alpha=0.75,
                     gamma=2.0,
                     iou_weighted=True,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='GIoULoss', loss_weight=1.5),
                 loss_bbox_refine: ConfigType = dict(
                     type='GIoULoss', loss_weight=2.0),
                 norm_cfg: ConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 use_atss: bool = True,
                 reg_decoded_bbox: bool = True,
                 anchor_generator: ConfigType = dict(
                     type='AnchorGenerator',
                     ratios=[1.0],
                     octave_base_scale=8,
                     scales_per_octave=1,
                     center_offset=0.0,
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='vfnet_cls',
                         std=0.01,
                         bias_prob=0.01)),                 
################################################                 
                 seg_out_channels=1,
                 loss_seg: ConfigType = dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0), 
                 condition_layers=1,
                 mapping=False,
                 con_kernel_size=7,###用于生成调制头的卷积核
                group_channels=16,
                con_reduction_ratio=4,
##################################################                                            
                 **kwargs) -> None:
        
        
        

        # dcn base offsets, adapted from reppoints_head.py
        self.num_dconv_points = 9
        self.dcn_kernel = int(np.sqrt(self.num_dconv_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        
#############################################################
        self.condition_layers = condition_layers
        self.mapping=mapping
        self.con_kernel_size = con_kernel_size
        self.seg_out_channels = seg_out_channels
        self.group_channels = group_channels   
        self.reduction_ratio = con_reduction_ratio
        
################################################################
        super(FCOSHead, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.regress_ranges = regress_ranges
        self.reg_denoms = [
            regress_range[-1] for regress_range in regress_ranges
        ]
        self.reg_denoms[-1] = self.reg_denoms[-2] * 2
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.sync_num_pos = sync_num_pos
        self.bbox_norm_type = bbox_norm_type
        self.gradient_mul = gradient_mul
        self.use_vfl = use_vfl
        if self.use_vfl:
            self.loss_cls = MODELS.build(loss_cls)
        else:
            self.loss_cls = MODELS.build(loss_cls_fl)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_bbox_refine = MODELS.build(loss_bbox_refine)

        # for getting ATSS targets
        self.use_atss = use_atss
        self.reg_decoded_bbox = reg_decoded_bbox
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)

        self.anchor_center_offset = anchor_generator['center_offset']

        self.num_base_priors = self.prior_generator.num_base_priors[0]

        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            if self.train_cfg.get('sampler', None) is not None:
                self.sampler = TASK_UTILS.build(
                    self.train_cfg['sampler'], default_args=dict(context=self))
            else:
                self.sampler = PseudoSampler()
        # only be used in `get_atss_targets` when `use_atss` is True
        self.atss_prior_generator = TASK_UTILS.build(anchor_generator)

        self.fcos_prior_generator = MlvlPointGenerator(
            anchor_generator['strides'],
            self.anchor_center_offset if self.use_atss else 0.5)

        # In order to reuse the `get_bboxes` in `BaseDenseHead.
        # Only be used in testing phase.
        self.prior_generator = self.fcos_prior_generator
        
#################################################
        # Initialize DCK modules
        self.cross_involution_cls = DCKModule(channels=self.feat_channels, 
                                              num_layers=self.condition_layers,
                                              kernel_size=self.con_kernel_size,
                                              group_channels=self.group_channels,
                                              reduction_ratio=self.reduction_ratio)
        self.cross_involution_reg = DCKModule(channels=self.feat_channels, 
                                              num_layers=self.condition_layers,
                                              kernel_size=self.con_kernel_size,
                                              group_channels=self.group_channels,
                                              reduction_ratio=self.reduction_ratio)
        self.loss_seg = MODELS.build(loss_seg)
        self.absolute = nn.Sigmoid()
        self.map = nn.ReLU()
#################################################  

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        super(FCOSHead, self)._init_cls_convs()
        super(FCOSHead, self)._init_reg_convs()
        self.relu = nn.ReLU()
        self.vfnet_reg_conv = ConvModule(
            self.feat_channels,
            self.feat_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            bias=self.conv_bias)
        self.vfnet_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

        self.vfnet_reg_refine_dconv = DeformConv2d(
            self.feat_channels,
            self.feat_channels,
            self.dcn_kernel,
            1,
            padding=self.dcn_pad)
        self.vfnet_reg_refine = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.scales_refine = nn.ModuleList([Scale(1.0) for _ in self.strides])

        self.vfnet_cls_dconv = DeformConv2d(
            self.feat_channels,
            self.feat_channels,
            self.dcn_kernel,
            1,
            padding=self.dcn_pad)
        self.vfnet_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        
        ##################################
        self.seg_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.seg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        ###########################
        self.vfnet_seg = nn.Conv2d(
            self.feat_channels,
            1,
            3,
            padding=1)
        ###################################  

        
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

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:

            - cls_scores (list[Tensor]): Box iou-aware scores for each scale
              level, each is a 4D-tensor, the channel number is
              num_points * num_classes.
            - bbox_preds (list[Tensor]): Box offsets for each
              scale level, each is a 4D-tensor, the channel number is
              num_points * 4.
            - bbox_preds_refine (list[Tensor]): Refined Box offsets for
              each scale level, each is a 4D-tensor, the channel
              number is num_points * 4.
        """
        return multi_apply(self.forward_single, x, self.scales,
                           self.scales_refine, self.strides, self.reg_denoms)

    def forward_single(self, x: Tensor, scale: Scale, scale_refine: Scale,
                       stride: int, reg_denom: int) -> tuple:
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            scale_refine (:obj: `mmcv.cnn.Scale`): Learnable scale module to
                resize the refined bbox prediction.
            stride (int): The corresponding stride for feature maps,
                used to normalize the bbox prediction when
                bbox_norm_type = 'stride'.
            reg_denom (int): The corresponding regression range for feature
                maps, only used to normalize the bbox prediction when
                bbox_norm_type = 'reg_denom'.

        Returns:
            tuple: iou-aware cls scores for each box, bbox predictions and
            refined bbox predictions of input feature maps.
        """
        cls_feat = x
        reg_feat = x
        seg_feat = x
        
        seg_feats = [seg_feat]  # 初始化保存 seg_feat 特征的列表

        # 处理分割特征
        for seg_layer in self.seg_convs:
            seg_feat = seg_layer(seg_feat)
            seg_feats.append(seg_feat)  # 每次经过 seg_layer 后保存 seg_feat 进入列表
            
            
        # 处理分类特征
        for i, cls_layer in enumerate(self.cls_convs):
            cls_feat_condi = self.cross_involution_cls(cls_feat, seg_feats[i+1])  # 用seg_feats进行调制
            cls_feat = cls_feat + cls_feat_condi  # 恒等映射，可以在此处添加其他操作
            cls_feat = cls_layer(cls_feat)
            
        # # 处理回归特征
        for i, reg_layer in enumerate(self.reg_convs):
            reg_feat_condi  = self.cross_involution_reg(reg_feat, seg_feats[i+1])  # 用seg_feats进行调制
            reg_feat = reg_feat_condi + reg_feat  # 恒等映射，可以在此处添加其他操作
            reg_feat = reg_layer(reg_feat)

        # predict the bbox_pred of different level
        reg_feat_init = self.vfnet_reg_conv(reg_feat)
        if self.bbox_norm_type == 'reg_denom':
            bbox_pred = scale(
                self.vfnet_reg(reg_feat_init)).float().exp() * reg_denom
        elif self.bbox_norm_type == 'stride':
            bbox_pred = scale(
                self.vfnet_reg(reg_feat_init)).float().exp() * stride
        else:
            raise NotImplementedError

        # compute star deformable convolution offsets
        # converting dcn_offset to reg_feat.dtype thus VFNet can be
        # trained with FP16
        dcn_offset = self.star_dcn_offset(bbox_pred, self.gradient_mul,
                                          stride).to(reg_feat.dtype)

        # refine the bbox_pred
        reg_feat = self.relu(self.vfnet_reg_refine_dconv(reg_feat, dcn_offset))
        bbox_pred_refine = scale_refine(
            self.vfnet_reg_refine(reg_feat)).float().exp()
        bbox_pred_refine = bbox_pred_refine * bbox_pred.detach()

        # predict the iou-aware cls score
        cls_feat = self.relu(self.vfnet_cls_dconv(cls_feat, dcn_offset))
        cls_score = self.vfnet_cls(cls_feat)
        
        # 使用最后一层 seg_feat 进行分割映射
        seg_score = self.vfnet_seg(seg_feats[-1])
        if self.mapping:
            seg_score = self.map(seg_score)
        else: 
            seg_score = self.absolute(seg_score)
        

        if self.training:
            return cls_score, bbox_pred, bbox_pred_refine, seg_score
        else:
            return cls_score, bbox_pred_refine

    def star_dcn_offset(self, bbox_pred: Tensor, gradient_mul: float,
                        stride: int) -> Tensor:
        """Compute the star deformable conv offsets.

        Args:
            bbox_pred (Tensor): Predicted bbox distance offsets (l, r, t, b).
            gradient_mul (float): Gradient multiplier.
            stride (int): The corresponding stride for feature maps,
                used to project the bbox onto the feature map.

        Returns:
            Tensor: The offsets for deformable convolution.
        """
        dcn_base_offset = self.dcn_base_offset.type_as(bbox_pred)
        bbox_pred_grad_mul = (1 - gradient_mul) * bbox_pred.detach() + \
            gradient_mul * bbox_pred
        # map to the feature map scale
        bbox_pred_grad_mul = bbox_pred_grad_mul / stride
        N, C, H, W = bbox_pred.size()

        x1 = bbox_pred_grad_mul[:, 0, :, :]
        y1 = bbox_pred_grad_mul[:, 1, :, :]
        x2 = bbox_pred_grad_mul[:, 2, :, :]
        y2 = bbox_pred_grad_mul[:, 3, :, :]
        bbox_pred_grad_mul_offset = bbox_pred.new_zeros(
            N, 2 * self.num_dconv_points, H, W)
        bbox_pred_grad_mul_offset[:, 0, :, :] = -1.0 * y1  # -y1
        bbox_pred_grad_mul_offset[:, 1, :, :] = -1.0 * x1  # -x1
        bbox_pred_grad_mul_offset[:, 2, :, :] = -1.0 * y1  # -y1
        bbox_pred_grad_mul_offset[:, 4, :, :] = -1.0 * y1  # -y1
        bbox_pred_grad_mul_offset[:, 5, :, :] = x2  # x2
        bbox_pred_grad_mul_offset[:, 7, :, :] = -1.0 * x1  # -x1
        bbox_pred_grad_mul_offset[:, 11, :, :] = x2  # x2
        bbox_pred_grad_mul_offset[:, 12, :, :] = y2  # y2
        bbox_pred_grad_mul_offset[:, 13, :, :] = -1.0 * x1  # -x1
        bbox_pred_grad_mul_offset[:, 14, :, :] = y2  # y2
        bbox_pred_grad_mul_offset[:, 16, :, :] = y2  # y2
        bbox_pred_grad_mul_offset[:, 17, :, :] = x2  # x2
        dcn_offset = bbox_pred_grad_mul_offset - dcn_base_offset

        return dcn_offset

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            bbox_preds_refine: List[Tensor],
            seg_scores: List[Tensor],         
            batch_gt_instances: InstanceList,
            batch_gt_sem_seg:PixelList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box offsets for each
                scale level, each is a 4D-tensor, the channel number is
                num_points * 4.
            bbox_preds_refine (list[Tensor]): Refined Box offsets for
                each scale level, each is a 4D-tensor, the channel
                number is num_points * 4.
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
        assert len(cls_scores) == len(bbox_preds) == len(bbox_preds_refine)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.fcos_prior_generator.grid_priors(
            featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)
        labels, label_weights, bbox_targets, bbox_weights = self.get_targets(
            cls_scores,
            all_level_points,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and bbox_preds_refine
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3,
                              1).reshape(-1,
                                         self.cls_out_channels).contiguous()
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4).contiguous()
            for bbox_pred in bbox_preds
        ]
        flatten_bbox_preds_refine = [
            bbox_pred_refine.permute(0, 2, 3, 1).reshape(-1, 4).contiguous()
            for bbox_pred_refine in bbox_preds_refine
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_bbox_preds_refine = torch.cat(flatten_bbox_preds_refine)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes - 1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = torch.where(
            ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)) > 0)[0]
        num_pos = len(pos_inds)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_bbox_preds_refine = flatten_bbox_preds_refine[pos_inds]
        pos_labels = flatten_labels[pos_inds]

        # sync num_pos across all gpus
        if self.sync_num_pos:
            num_pos_avg_per_gpu = reduce_mean(
                pos_inds.new_tensor(num_pos).float()).item()
            num_pos_avg_per_gpu = max(num_pos_avg_per_gpu, 1.0)
        else:
            num_pos_avg_per_gpu = num_pos

        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_points = flatten_points[pos_inds]

        pos_decoded_bbox_preds = self.bbox_coder.decode(
            pos_points, pos_bbox_preds)
        pos_decoded_target_preds = self.bbox_coder.decode(
            pos_points, pos_bbox_targets)
        iou_targets_ini = bbox_overlaps(
            pos_decoded_bbox_preds,
            pos_decoded_target_preds.detach(),
            is_aligned=True).clamp(min=1e-6)
        bbox_weights_ini = iou_targets_ini.clone().detach()
        bbox_avg_factor_ini = reduce_mean(
            bbox_weights_ini.sum()).clamp_(min=1).item()

        pos_decoded_bbox_preds_refine = \
            self.bbox_coder.decode(pos_points, pos_bbox_preds_refine)
        iou_targets_rf = bbox_overlaps(
            pos_decoded_bbox_preds_refine,
            pos_decoded_target_preds.detach(),
            is_aligned=True).clamp(min=1e-6)
        bbox_weights_rf = iou_targets_rf.clone().detach()
        bbox_avg_factor_rf = reduce_mean(
            bbox_weights_rf.sum()).clamp_(min=1).item()

        if num_pos > 0:
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds.detach(),
                weight=bbox_weights_ini,
                avg_factor=bbox_avg_factor_ini)

            loss_bbox_refine = self.loss_bbox_refine(
                pos_decoded_bbox_preds_refine,
                pos_decoded_target_preds.detach(),
                weight=bbox_weights_rf,
                avg_factor=bbox_avg_factor_rf)

            # build IoU-aware cls_score targets
            if self.use_vfl:
                pos_ious = iou_targets_rf.clone().detach()
                cls_iou_targets = torch.zeros_like(flatten_cls_scores)
                cls_iou_targets[pos_inds, pos_labels] = pos_ious
        else:
            loss_bbox = pos_bbox_preds.sum() * 0
            loss_bbox_refine = pos_bbox_preds_refine.sum() * 0
            if self.use_vfl:
                cls_iou_targets = torch.zeros_like(flatten_cls_scores)

        if self.use_vfl:
            loss_cls = self.loss_cls(
                flatten_cls_scores,
                cls_iou_targets,
                avg_factor=num_pos_avg_per_gpu)
        else:
            loss_cls = self.loss_cls(
                flatten_cls_scores,
                flatten_labels,
                weight=label_weights,
                avg_factor=num_pos_avg_per_gpu)
            
###################################        
        seg_tagets = self.get_seg_targets(cls_scores, batch_gt_sem_seg)
        flatten_seg_scores = [
            seg_score.permute(0, 2, 3, 1).reshape(-1, self.seg_out_channels)
            for seg_score in seg_scores
        ]        
        flatten_seg_scores = torch.cat(flatten_seg_scores)
        flatten_seg_targets = torch.cat(seg_tagets)
        loss_seg = self.loss_seg(
            flatten_seg_scores, flatten_seg_targets)   
###################################### 

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_bbox_rf=loss_bbox_refine,
            loss_seg = loss_seg)

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
        # print("batch_gt_instances:", batch_gt_instances)
        # construct the segmentation targets of shape (B, C, H, W) for each layer
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
            # lvl_seg_target[lvl_seg_target == 255] = 1.0
            seg_targets.append(lvl_seg_target)
        
        
        # flatten seg_targets
        flatten_seg_targets = [
            seg_target.permute(0, 2, 3, 1).reshape(-1, self.seg_out_channels)
            for seg_target in seg_targets
        ]

        return flatten_seg_targets

    def get_targets(
            self,
            cls_scores: List[Tensor],
            mlvl_points: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> tuple:
        """A wrapper for computing ATSS and FCOS targets for points in multiple
        images.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level with shape (N, num_points * num_classes, H, W).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 2).
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
            tuple:

            - labels_list (list[Tensor]): Labels of each level.
            - label_weights (Tensor/None): Label weights of all levels.
            - bbox_targets_list (list[Tensor]): Regression targets of each
              level, (l, t, r, b).
            - bbox_weights (Tensor/None): Bbox weights of all levels.
        """
        if self.use_atss:
            return self.get_atss_targets(cls_scores, mlvl_points,
                                         batch_gt_instances, batch_img_metas,
                                         batch_gt_instances_ignore)
        else:
            self.norm_on_bbox = False
            return self.get_fcos_targets(mlvl_points, batch_gt_instances)

    def _get_targets_single(self, *args, **kwargs):
        """Avoid ambiguity in multiple inheritance."""
        if self.use_atss:
            return ATSSHead._get_targets_single(self, *args, **kwargs)
        else:
            return FCOSHead._get_targets_single(self, *args, **kwargs)

    def get_fcos_targets(self, points: List[Tensor],
                         batch_gt_instances: InstanceList) -> tuple:
        """Compute FCOS regression and classification targets for points in
        multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.

        Returns:
            tuple:

            - labels (list[Tensor]): Labels of each level.
            - label_weights: None, to be compatible with ATSS targets.
            - bbox_targets (list[Tensor]): BBox targets of each level.
            - bbox_weights: None, to be compatible with ATSS targets.
        """
        labels, bbox_targets = FCOSHead.get_targets(self, points,
                                                    batch_gt_instances)
        label_weights = None
        bbox_weights = None
        return labels, label_weights, bbox_targets, bbox_weights

    def get_anchors(self,
                    featmap_sizes: List[Tuple],
                    batch_img_metas: List[dict],
                    device: str = 'cuda') -> tuple:
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            batch_img_metas (list[dict]): Image meta info.
            device (str): Device for returned tensors

        Returns:
            tuple:

            - anchor_list (list[Tensor]): Anchors of each image.
            - valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(batch_img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.atss_prior_generator.grid_priors(
            featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            multi_level_flags = self.atss_prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device=device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def get_atss_targets(
            self,
            cls_scores: List[Tensor],
            mlvl_points: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> tuple:
        """A wrapper for computing ATSS targets for points in multiple images.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level with shape (N, num_points * num_classes, H, W).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 2).
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
            tuple:

            - labels_list (list[Tensor]): Labels of each level.
            - label_weights (Tensor): Label weights of all levels.
            - bbox_targets_list (list[Tensor]): Regression targets of each
              level, (l, t, r, b).
            - bbox_weights (Tensor): Bbox weights of all levels.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(
            featmap_sizes
        ) == self.atss_prior_generator.num_levels == \
            self.fcos_prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = ATSSHead.get_targets(
            self,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            unmap_outputs=True)

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor) = cls_reg_targets

        bbox_targets_list = [
            bbox_targets.reshape(-1, 4) for bbox_targets in bbox_targets_list
        ]

        num_imgs = len(batch_img_metas)
        # transform bbox_targets (x1, y1, x2, y2) into (l, t, r, b) format
        bbox_targets_list = self.transform_bbox_targets(
            bbox_targets_list, mlvl_points, num_imgs)

        labels_list = [labels.reshape(-1) for labels in labels_list]
        label_weights_list = [
            label_weights.reshape(-1) for label_weights in label_weights_list
        ]
        bbox_weights_list = [
            bbox_weights.reshape(-1) for bbox_weights in bbox_weights_list
        ]
        label_weights = torch.cat(label_weights_list)
        bbox_weights = torch.cat(bbox_weights_list)
        return labels_list, label_weights, bbox_targets_list, bbox_weights

    def transform_bbox_targets(self, decoded_bboxes: List[Tensor],
                               mlvl_points: List[Tensor],
                               num_imgs: int) -> List[Tensor]:
        """Transform bbox_targets (x1, y1, x2, y2) into (l, t, r, b) format.

        Args:
            decoded_bboxes (list[Tensor]): Regression targets of each level,
                in the form of (x1, y1, x2, y2).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 2).
            num_imgs (int): the number of images in a batch.

        Returns:
            bbox_targets (list[Tensor]): Regression targets of each level in
                the form of (l, t, r, b).
        """
        # TODO: Re-implemented in Class PointCoder
        assert len(decoded_bboxes) == len(mlvl_points)
        num_levels = len(decoded_bboxes)
        mlvl_points = [points.repeat(num_imgs, 1) for points in mlvl_points]
        bbox_targets = []
        for i in range(num_levels):
            bbox_target = self.bbox_coder.encode(mlvl_points[i],
                                                 decoded_bboxes[i])
            bbox_targets.append(bbox_target)

        return bbox_targets

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Override the method in the parent class to avoid changing para's
        name."""
        pass


class DCKModule(nn.Module):  
    def __init__(self, channels,   
                 num_layers=1,   
                 kernel_size=7,   
                 group_channels=16,  
                 reduction_ratio=4):  
        super(DCKModule, self).__init__()  
        self.kernel_size = kernel_size  
        self.channels = channels  
        self.num_layers = num_layers  
        self.group_channels = group_channels  
        self.reduction_ratio = reduction_ratio  
        self.groups = self.channels // self.group_channels  

        self.convs1 = nn.ModuleList()  
        self.convs2 = nn.ModuleList()  

        for _ in range(num_layers):  
            self.convs1.append(ConvModule(  
                in_channels=channels,  
                out_channels=channels // reduction_ratio,  
                kernel_size=1,  
                conv_cfg=None,  
                norm_cfg=dict(type='BN'),  
                act_cfg=dict(type='ReLU')))  
            # self.convs2.append(nn.Conv2d(  
            #     in_channels=channels // reduction_ratio,  
            #     out_channels=self.groups * self.kernel_size * self.kernel_size,  
            #     kernel_size=1,  
            #     stride=1,  
            #     bias=False))  
            self.convs2.append(  
                ConvModule(  # 改用ConvModule包装  
                    in_channels=channels // reduction_ratio,  
                    out_channels=self.groups * self.kernel_size * self.kernel_size,  
                    kernel_size=1,  
                    stride=1,  
                    conv_cfg=None,  
                    norm_cfg=None,    
                    act_cfg=None,   
                    bias=True))     

        self.padding = (kernel_size - 1) // 2  

    def forward(self, feature_map, guide_map):  
        b, c, h, w = feature_map.size()  
        gc = self.group_channels  
        g = self.groups  
        n = self.kernel_size * self.kernel_size  
        for i in range(self.num_layers):  
            # 生成动态卷积核  
            dynamic_filters = self.convs2[i](self.convs1[i](guide_map))  
            # dynamic_filters 形状: (b, g * n, h, w)  
            dynamic_filters = dynamic_filters.view(b, g, n, h, w)  
            dynamic_filters = dynamic_filters.permute(0, 3, 4, 1, 2)  # 形状: (b, h, w, g, n)  

            # 提取输入特征图的拼块  
            input_patches = F.unfold(feature_map, kernel_size=self.kernel_size, padding=self.padding)  # (b, c * n, h * w)  
            input_patches = input_patches.view(b, c, n, h, w)  # (b, c, n, h, w)  
            input_patches = input_patches.view(b, g, gc, n, h, w)  # (b, g, gc, n, h, w)  
            input_patches = input_patches.permute(0, 4, 5, 1, 3, 2)  # 形状: (b, h, w, g, n, gc)  

            # 计算输出  
            out = torch.einsum('bhwgnc,bhwgn->bhwgc', input_patches, dynamic_filters)  
            out = out.permute(0, 3, 4, 1, 2).contiguous()  # 形状: (b, g, gc, h, w)  
            out = out.view(b, c, h, w)  

            # 残差连接  
            feature_map = out + feature_map  
        return feature_map  


class cross_involution(nn.Module):
    def __init__(self, channels, kernel_size, num_layers=1, stride=1):
        super(cross_involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        self.num_layers = num_layers
        self.reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for _ in range(num_layers):
            self.convs1.append(ConvModule(
                in_channels=channels,
                out_channels=channels // self.reduction_ratio,
                kernel_size=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')))
            self.convs2.append(ConvModule(
                in_channels=channels // self.reduction_ratio,
                out_channels=self.kernel_size**2 * self.groups,
                kernel_size=1,
                stride=1,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=None))

        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)

    def forward(self, feature_map, guide_map):
        for i in range(self.num_layers):
            weight = self.convs2[i](self.convs1[i](guide_map))
            b, c, h, w = weight.shape
            weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)
            out = self.unfold(feature_map).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
            out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
            feature_map = out + feature_map  # residual connection
        return feature_map
    
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
    

