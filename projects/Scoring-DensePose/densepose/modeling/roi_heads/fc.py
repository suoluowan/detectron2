# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.layers import Conv2d, DeformConv

from ..utils import initialize_module_params
from .registry import ROI_DENSEPOSE_HEAD_REGISTRY

from typing import Any


@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class DensePoseFCHead(nn.Module):

    def __init__(self, cfg: CfgNode, input_channels: int):

        super(DensePoseFCHead, self).__init__()
        
        self.n_stacked_fcs = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.NUM_STACKED_FCS
        self.n_stacked_convs = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.NUM_STACKED_CONVS
        
        hidden_dim = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.HIDDEN_DIM
        kernel_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.CONV_HEAD_KERNEL
        pad_size = kernel_size // 2
        # input channel
        heatmap_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE
        dim_out_patches = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1
        n_segm_chan = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
        n_channels = input_channels + dim_out_patches

        score_pool_resolution = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.POOLER_RESOLUTION

        pool_resolution = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_RESOLUTION
        self.down_scale = heatmap_size // pool_resolution
        
        for i in range(self.n_stacked_convs):
            # layer = Conv2d(n_channels, hidden_dim, kernel_size, stride=1, padding=pad_size)
            layer = DeformConv(n_channels, hidden_dim, kernel_size, stride=1, padding=pad_size)
            layer_name = self._get_layer_name(i)
            self.add_module(layer_name, layer)
            n_channels = hidden_dim
        # self.avgpool = nn.AdaptiveAvgPool2d(score_pool_resolution)
        # self.offsetConv = Conv2d(n_channels, kernel_size*kernel_size*2, kernel_size, stride=1, padding=pad_size, bias=True)
        n_channels = n_channels * score_pool_resolution * score_pool_resolution
        hidden_dim = n_channels
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
    def forward(self, features: torch.Tensor, densepose_predictor_outputs: Any):
        """
        Apply DensePose fully convolutional head to the input features

        Args:
            features (tensor): input features
        Result:
            A tensor of DensePose head outputs
        """
        # if densepose_predictor_outputs.u.size(0) != 0:
        #     top_values, top_index = densepose_predictor_outputs.fine_segm.topk(8, dim=1, largest=True, sorted=True)
        #     densepose_output = torch.mean(top_values, dim=1, keepdim=True)
        #     # i_est = torch.argmax(densepose_predictor_outputs.fine_segm, dim=1, keepdim=True)
        #     # u = densepose_predictor_outputs.u.gather(1, i_est)
        #     # v = densepose_predictor_outputs.v.gather(1, i_est)
        #     # densepose_output = torch.cat((densepose_predictor_outputs.fine_segm, u, v), 1)
        #     densepose_output = F.max_pool2d(densepose_output, kernel_size=self.down_scale, stride=self.down_scale)
        densepose_output = F.max_pool2d(densepose_predictor_outputs.fine_segm, kernel_size=self.down_scale, stride=self.down_scale)
        # else:
        #     densepose_output = torch.zeros((features.size(0), 1, features.size(2), features.size(3)), device=features.device)
        x = torch.cat((features, densepose_output.detach()), 1)
        # x = features

        # ten conv
        if x.size(0) == 0:
            offset = x.detach()
        else:
            offset = torch.empty((1, 2 * 9 * 1 * 1, 1, 1), device=x.device)
            for c, (rel_offset_h, rel_offset_w) in enumerate([(0, -2), (0, -1), (0, 0), (0, 1), (0,2),(-2, 0), (-1, 0), (1, 0), (2, 0)]):
                offset[0, c * 2 + 0, 0, 0] = rel_offset_h
                offset[0, c * 2 + 1, 0, 0] = rel_offset_w
            offset = offset.repeat(x.size(0), 1, x.size(2), x.size(3))

        for i in range(self.n_stacked_convs):
            layer_name = self._get_layer_name(i)
            # x = getattr(self, layer_name)(x)
            x = getattr(self, layer_name)(x,offset)
            x = F.relu(x)
            output = x
        # x = self.avgpool(x)
        # if x.size(0) == 0:
        #     x = x
        # else:
        #     x = x.reshape(x.size(0), -1)
        # x = x.reshape(x.size(0), -1)
        # output = x
        # for i in range(self.n_stacked_fcs):
        #     layer_name = self._get_fc_layer_name(i)
        #     x = getattr(self, layer_name)(x)
        #     x = F.relu(x)
        #     output = x
        return output

    def _get_fc_layer_name(self, i: int):
        layer_name = "score_body_conv_fc{}".format(i + 1)
        return layer_name
    def _get_layer_name(self, i: int):
        layer_name = "score_body_conv_{}".format(i + 1)
        return layer_name
