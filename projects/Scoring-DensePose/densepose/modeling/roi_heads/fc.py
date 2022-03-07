# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.layers import Conv2d

from ..utils import initialize_module_params
from .registry import ROI_DENSEPOSE_HEAD_REGISTRY

from typing import Any


@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class DensePoseFCHead(nn.Module):

    def __init__(self, cfg: CfgNode, input_channels: int):

        super(DensePoseFCHead, self).__init__()
        # fmt: off
        hidden_dim = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.HIDDEN_DIM
        self.n_stacked_fcs = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.NUM_STACKED_FCS
        self.n_stacked_convs = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.NUM_STACKED_CONVS
        kernel_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.CONV_HEAD_KERNEL
        heatmap_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE
        dim_out_patches = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1
        pool_resolution = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_RESOLUTION
        self.down_scale = heatmap_size // pool_resolution
        # fmt: on
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        n_channels = input_channels + dim_out_patches*3
        pad_size = kernel_size // 2
        for i in range(self.n_stacked_convs):
            layer = Conv2d(n_channels, hidden_dim, kernel_size, stride=1, padding=pad_size)
            layer_name = self._get_layer_name(i)
            self.add_module(layer_name, layer)
            n_channels = hidden_dim
        for i in range(self.n_stacked_fcs):
            layer = nn.Linear(n_channels, hidden_dim)
            layer_name = self._get_fc_layer_name(i)
            self.add_module(layer_name, layer)
            n_channels = hidden_dim
        self.n_out_channels = n_channels
        initialize_module_params(self)

    # def forward(self, features: torch.Tensor):
    #     """
    #     Apply DensePose fully convolutional head to the input features

    #     Args:
    #         features (tensor): input features
    #     Result:
    #         A tensor of DensePose head outputs
    #     """
    #     x = features
    #     x = self.avgpool(x)
    #     x = x.view(x.size(0), -1)
    #     output = x
    #     for i in range(self.n_stacked_fcs):
    #         layer_name = self._get_fc_layer_name(i)
    #         x = getattr(self, layer_name)(x)
    #         x = F.relu(x)
    #         output = x
    #     return output
    def forward(self, features: torch.Tensor, densepose_predictor_outputs: Any ):
        """
        Apply DensePose fully convolutional head to the input features

        Args:
            features (tensor): input features
        Result:
            A tensor of DensePose head outputs
        """
        if features.size(0) == 0:
            print(features.size())
            return features
        densepose_output = torch.cat((densepose_predictor_outputs.fine_segm, densepose_predictor_outputs.u, densepose_predictor_outputs.v), 1)
        densepose_output = F.max_pool2d(densepose_output, kernel_size=self.down_scale, stride=self.down_scale)
        x = torch.cat((features, densepose_output), 1)
        for i in range(self.n_stacked_convs):
            layer_name = self._get_layer_name(i)
            x = getattr(self, layer_name)(x)
            x = F.relu(x)
            output = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        output = x
        for i in range(self.n_stacked_fcs):
            layer_name = self._get_fc_layer_name(i)
            x = getattr(self, layer_name)(x)
            x = F.relu(x)
            output = x
        return output

    def _get_fc_layer_name(self, i: int):
        layer_name = "score_body_conv_fc{}".format(i + 1)
        return layer_name
    def _get_layer_name(self, i: int):
        layer_name = "score_body_conv_{}".format(i + 1)
        return layer_name
