# Copyright (c) GrokCV. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcv.cnn import ConvModule, Scale
from mmdet.registry import MODELS
from .fcos_MDDH import FCOSDepthHead
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule


@MODELS.register_module()
class FCOSDeCoDetHead(FCOSDepthHead):
    def __init__(self,
                 mapping=False,
                 condition_layers=1,
                 con_kernel_size=7,
                 group_channels=16,
                 con_reduction_ratio=4,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.condition_layers = condition_layers  
        self.con_kernel_size = con_kernel_size    
        self.group_channels = group_channels   
        self.reduction_ratio = con_reduction_ratio
        
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
        
        self.mapping = mapping
        self.map = nn.ReLU()
        self.absolute = nn.Sigmoid()

    def _init_layers(self) -> None:
        super()._init_layers()

    def forward_single(self, x: Tensor, scale: Scale, stride: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        cls_feat = x
        reg_feat = x
        seg_feat = x

        seg_feats = [seg_feat]  # Initialize list to store seg_feat features

        # Process segmentation features
        for seg_layer in self.seg_convs:
            seg_feat = seg_layer(seg_feat)
            seg_feats.append(seg_feat)  # Save seg_feat after each seg_layer

        # Process classification features
        for i, cls_layer in enumerate(self.cls_convs):
            cls_feat_condi = self.cross_involution_cls(cls_feat, seg_feats[i+1])  # Modulate with seg_feats
            cls_feat = cls_feat + cls_feat_condi  # Identity mapping, can add other operations here
            cls_feat = cls_layer(cls_feat)

        # Process regression features
        for i, reg_layer in enumerate(self.reg_convs):
            reg_feat_condi = self.cross_involution_reg(reg_feat, seg_feats[i+1])  # Modulate with seg_feats
            reg_feat = reg_feat_condi + reg_feat  # Identity mapping, can add other operations here
            reg_feat = reg_layer(reg_feat)

        # Use the last seg_feat for segmentation mapping
        seg_score = self.conv_seg(seg_feats[-1])

        # Compute final outputs
        cls_score = self.conv_cls(cls_feat)
        bbox_pred = self.conv_reg(reg_feat)

        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()

        if self.mapping:
            seg_score = self.map(seg_score)
        else: 
            seg_score = self.absolute(seg_score)

        return cls_score, bbox_pred, centerness, seg_score
    
    

  
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
            self.convs2.append(nn.Conv2d(  
                in_channels=channels // reduction_ratio,  
                out_channels=self.groups * self.kernel_size * self.kernel_size,  
                kernel_size=1,  
                stride=1,  
                bias=False))  

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
    

    
class ChannelExchange(BaseModule):
    """
    channel exchange
    Args:
        p (float, optional): p of the features will be exchanged.
            Defaults to 1/2.
    """
    def __init__(self, p=1/2):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = int(1/p)
        self.mlp_exchange = MLP(in_channels=1, hidden_channels= 64, out_channels=1)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        
        exchange_map = torch.arange(c) % self.p == 0
        # exchange_mask = exchange_map.unsqueeze(0).expand((N, -1))
        exchange_mask = exchange_map.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand((N, c, h, w))

        # Apply MLP to the selected features (exchange and non-exchange)
        x1_exchange = self.mlp_exchange(x1[:, exchange_map, :, :].reshape(-1, 1))
        x2_exchange = self.mlp_exchange(x2[:, exchange_map, :, :].reshape(-1, 1))
        
        # Reshape back to the original shape
        x1_processed = x1.clone()
        x2_processed = x2.clone()
        
        x1_processed[:, exchange_map, :, :] = x1_exchange.reshape(N, -1, h, w)
        x2_processed[:, exchange_map, :, :] = x2_exchange.reshape(N, -1, h, w)
        
        # Perform the channel exchange
        out_x1 = x1.clone()
        out_x2 = x2.clone()

        out_x1[exchange_mask] = x2_processed[exchange_mask]
        out_x2[exchange_mask] = x1_processed[exchange_mask]
        
        return out_x1, out_x2

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    
class Basic1d(nn.Module):
    def __init__(self, inplanes, planes, with_activation=True):
        super().__init__()
        self.fc = nn.Linear(inplanes, planes)
        self.with_activation = with_activation
        if with_activation:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        if self.with_activation:
            x = self.act(x)
        return x

class Dynamic_MLP_A(nn.Module):
    def __init__(self, inplanes, planes, loc_planes):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes

        self.get_weight = nn.Linear(loc_planes, inplanes * planes)
        self.norm = nn.LayerNorm(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img_fea, loc_fea):
        weight = self.get_weight(loc_fea)
        weight = weight.view(-1, self.inplanes, self.planes)

        img_fea = torch.bmm(img_fea.unsqueeze(1), weight).squeeze(1)
        img_fea = self.norm(img_fea)
        img_fea = self.relu(img_fea)

        return img_fea

class Dynamic_MLP_B(nn.Module):
    def __init__(self, inplanes, planes, loc_planes):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes

        self.conv11 = Basic1d(inplanes, inplanes, True)
        self.conv12 = nn.Linear(inplanes, inplanes)

        self.conv21 = Basic1d(loc_planes, inplanes, True)
        self.conv22 = nn.Linear(inplanes, inplanes * planes)

        self.br = nn.Sequential(
            nn.LayerNorm(planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = Basic1d(planes, planes, False)

    def forward(self, img_fea, loc_fea):
        weight11 = self.conv11(img_fea)
        weight12 = self.conv12(weight11)

        weight21 = self.conv21(loc_fea)
        weight22 = self.conv22(weight21).view(-1, self.inplanes, self.planes)

        img_fea = torch.bmm(weight12.unsqueeze(1), weight22).squeeze(1)
        img_fea = self.br(img_fea)
        img_fea = self.conv3(img_fea)

        return img_fea
   
class Dynamic_MLP_C(nn.Module):
    def __init__(self, inplanes, planes, loc_planes):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes

        self.conv11 = Basic1d(inplanes + loc_planes, inplanes, True)
        self.conv12 = nn.Linear(inplanes, inplanes)

        self.conv21 = Basic1d(inplanes + loc_planes, inplanes, True)
        self.conv22 = nn.Linear(inplanes, inplanes * planes)

        self.br = nn.Sequential(
            nn.LayerNorm(planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = Basic1d(planes, planes, False)

    def forward(self, img_fea, loc_fea):
        # Concatenate the image and location features
        B, N, C = img_fea.shape        
        
        cat_fea = torch.cat([img_fea, loc_fea], dim=2)

        # First set of convolutions
        weight11 = self.conv11(cat_fea)
        weight12 = self.conv12(weight11)

        # Second set of convolutions
        weight21 = self.conv21(cat_fea)
        weight22 = self.conv22(weight21).view(B, -1, self.inplanes, self.planes)

        # Apply dynamic weights
        img_fea = torch.matmul(weight12.unsqueeze(2), weight22).squeeze(2)  # Remove the added dimension
        img_fea = self.br(img_fea)
        img_fea = self.conv3(img_fea)

        return img_fea

class RecursiveBlock(nn.Module):
    def __init__(self, inplanes, planes, loc_planes, mlp_type='c'):
        super().__init__()
        if mlp_type.lower() == 'a':
            MLP = Dynamic_MLP_A
        elif mlp_type.lower() == 'b':
            MLP = Dynamic_MLP_B
        elif mlp_type.lower() == 'c':
            MLP = Dynamic_MLP_C

        self.dynamic_conv = MLP(inplanes, planes, loc_planes)

    def forward(self, img_fea, loc_fea):
        img_fea = self.dynamic_conv(img_fea, loc_fea)
        return img_fea, loc_fea

class FusionModule(nn.Module):
    def __init__(self, inplanes=256, planes=256, hidden=64, num_layers=1, mlp_type='c'):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.hidden = hidden

        self.conv1 = nn.Linear(inplanes, planes)

        conv2 = []
        if num_layers == 1:
            conv2.append(RecursiveBlock(planes, planes, loc_planes=planes, mlp_type=mlp_type))
        else:
            conv2.append(RecursiveBlock(planes, hidden, loc_planes=planes, mlp_type=mlp_type))
            for _ in range(1, num_layers - 1):
                conv2.append(RecursiveBlock(hidden, hidden, loc_planes=planes, mlp_type=mlp_type))
            conv2.append(RecursiveBlock(hidden, planes, loc_planes=planes, mlp_type=mlp_type))
        self.conv2 = nn.ModuleList(conv2)

        self.conv3 = nn.Linear(planes, inplanes)
        self.norm3 = nn.LayerNorm(inplanes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, img_fea, loc_fea):
        identity = img_fea

        img_fea = self.conv1(img_fea)

        for m in self.conv2:
            img_fea, loc_fea = m(img_fea, loc_fea)

        img_fea = self.conv3(img_fea)
        img_fea = self.norm3(img_fea)

        img_fea += identity

        return img_fea