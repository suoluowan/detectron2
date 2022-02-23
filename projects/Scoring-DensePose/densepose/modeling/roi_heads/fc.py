# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.layers import Conv2d

from ..utils import initialize_module_params
from .registry import ROI_DENSEPOSE_HEAD_REGISTRY


@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class DensePoseFCHead(nn.Module):

    def __init__(self, cfg: CfgNode, input_channels: int):

        super(DensePoseFCHead, self).__init__()
        # fmt: off
        hidden_dim = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.HIDDEN_DIM
        self.n_stacked_fcs = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.NUM_STACKED_FCS
        # fmt: on
        n_channels = input_channels
        for i in range(self.n_stacked_fcs):
            layer = nn.Linear(n_channels, hidden_dim)
            layer_name = self._get_fc_layer_name(i)
            self.add_module(layer_name, layer)
            n_channels = hidden_dim
        self.n_out_channels = n_channels
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        initialize_module_params(self)

    def forward(self, features: torch.Tensor):
        """
        Apply DensePose fully convolutional head to the input features

        Args:
            features (tensor): input features
        Result:
            A tensor of DensePose head outputs
        """
        x = features
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
        layer_name = "body_conv_fc{}".format(i + 1)
        return layer_name
