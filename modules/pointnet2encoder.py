#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : pointnet2encoder.py
# Author            : Anonymous
# Date              : 25.06.2020
# Last Modified Date: 25.06.2020
# Last Modified By  : Hai-Yong Jiang
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2.pointnet2.pointnet2_modules import PointnetSAModuleMSG, PointnetSAModule, PointnetFPModule
from .pointnet2.pointnet2 import pytorch_utils as pt_utils
from .pointnet2.pointnet2 import pointnet2_utils
from utils.config import cfg
from utils.dataloader import label_dict, class_num

class Pointnet2MSGSEG(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Classification network

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier
        input_channels: int = 3
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, input_channels=3, use_xyz=True, bn=True, global_feat=True):
        super().__init__()

        self.global_feat = global_feat

        self.SA_modules = nn.ModuleList()

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[input_channels, 16, 16, 32], [input_channels, 32, 32, 64]],
                use_xyz=use_xyz,
                bn=bn
            )
        )

        input_channels_2 = 32 + 64
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[input_channels_2, 32, 32,
                       64], [input_channels_2, 64, 64, 128]],
                use_xyz=use_xyz,
                bn=bn
            )
        )

        input_channels_3 = 64 + 128
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=[0.2, 0.4],
                nsamples=[32, 64],
                mlps=[[input_channels_3, 64, 64,
                       128], [input_channels_3, 128, 128, 256]],
                use_xyz=use_xyz,
                bn=bn
            )
        )

        input_channels_4 = 128 + 256

        if global_feat:
            self.Middle_modules = PointnetSAModule(
                mlp=[input_channels_4, 256, 512], use_xyz=use_xyz,
                bn=bn
            )
        self.num_feat = 512
        
        self.pointwise_num_feat = 64 + 128 + 256 + 128 + 256

        self.feat_channels_list = [64, 128, 256, 128+256]

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + input_channels, 128, 64], bn=bn))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + input_channels_2, 256, 128], bn=bn))
        self.FP_modules.append(
            PointnetFPModule(mlp=[input_channels_4 + input_channels_3, 512, 256], bn=bn)
        )

        self.FC_layer = nn.Sequential(
            pt_utils.Conv1d(64, 32, bn=True), nn.Dropout(),
            pt_utils.Conv1d(32, class_num, activation=None)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """

        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        if self.global_feat:
            _, middle_features = self.Middle_modules(l_xyz[-1], l_features[-1])
        else:
            middle_features = None

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        sem_logits = self.FC_layer(l_features[0]).transpose(1, 2).contiguous()

        return middle_features, sem_logits, l_features, l_xyz