# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.layers import Conv2d, DeformConv

from ..utils import initialize_module_params
from .registry import ROI_DENSEPOSE_HEAD_REGISTRY

from typing import Any
from torch.nn import Softmax

from .dynamic_conv import Dynamic_conv2d


@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class DensePoseFCHead(nn.Module):

    def __init__(self, cfg: CfgNode, input_channels: int):

        super(DensePoseFCHead, self).__init__()
        
        self.n_stacked_fcs = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.NUM_STACKED_FCS
        self.n_stacked_convs = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.NUM_STACKED_CONVS
        
        hidden_dim = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.HIDDEN_DIM
        kernel_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.CONV_HEAD_KERNEL
        pad_size = kernel_size // 2
        self.offset_pad_size = (kernel_size**2 - 1) // 4
        # input channel
        heatmap_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE
        dim_out_patches = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1
        n_segm_chan = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
        n_channels = input_channels

        score_pool_resolution = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.POOLER_RESOLUTION

        pool_resolution = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_RESOLUTION
        self.down_scale = heatmap_size // pool_resolution
        
        self.use_dynamic_conv = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.DYNAMIC
        if self.use_dynamic_conv:
            in_planes = 5
            self.dynamic_channels = n_channels
            # self.dynamic_weights = Dynamic_conv2d(in_planes, n_channels, self.dynamic_channels, kernel_size=1, stride=1, bias=False, K=3, ratio=0.2)
            # self.instance_bn = nn.BatchNorm2d(self.dynamic_channels)
            # self.relu = nn.ReLU(inplace=True)
            self.dynamic_conv1 = Dynamic_conv2d(in_planes, n_channels, self.dynamic_channels, kernel_size=3, stride=1, padding=1, bias=False, K=3, ratio=12.8)
            self.instance_bn1 = nn.BatchNorm2d(self.dynamic_channels)
            self.relu = nn.ReLU(inplace=True)
            in_planes = 1
            self.dynamic_conv2 = Dynamic_conv2d(in_planes, self.dynamic_channels, self.dynamic_channels, kernel_size=3, stride=1, padding=1, bias=False, K=3, ratio=64)
            self.instance_bn2 = nn.BatchNorm2d(self.dynamic_channels)
            n_channels = self.dynamic_channels

        for i in range(self.n_stacked_convs):
            layer = Conv2d(n_channels, hidden_dim, kernel_size, stride=1, padding=pad_size)
            # layer = DeformConv(n_channels, hidden_dim, kernel_size, stride=1, padding=pad_size)
            layer_name = self._get_layer_name(i)
            self.add_module(layer_name, layer)
            n_channels = hidden_dim
        # self.avgpool = nn.AdaptiveAvgPool2d(score_pool_resolution)
        n_channels = n_channels * score_pool_resolution * score_pool_resolution
        hidden_dim = n_channels
        for i in range(self.n_stacked_fcs):
            layer = nn.Linear(n_channels, hidden_dim)
            layer_name = self._get_fc_layer_name(i)
            self.add_module(layer_name, layer)
            n_channels = hidden_dim
        self.n_out_channels = n_channels

        self.offset_size = 2*kernel_size*kernel_size

        # self.query_conv = Conv2d(n_channels, n_channels//8, kernel_size=1)
        # self.key_conv = Conv2d(n_channels, n_channels//8, kernel_size=1)
        # self.value_conv = Conv2d(n_channels, n_channels, kernel_size=1)
        # self.softmax = Softmax(dim=3)
        # self.INF = INF
        # self.gamma = nn.Parameter(torch.zeros(1))

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
            # densepose_output = torch.cat((densepose_predictor_outputs.fine_segm, densepose_predictor_outputs.u, densepose_predictor_outputs.v), 1)
            # densepose_output = F.max_pool2d(densepose_output, kernel_size=self.down_scale, stride=self.down_scale)
        # densepose_output = F.max_pool2d(densepose_predictor_outputs.fine_segm, kernel_size=self.down_scale, stride=self.down_scale)
        # else:
        #     densepose_output = torch.zeros((features.size(0), 75, features.size(2), features.size(3)), device=features.device)
        # x = torch.cat((features, densepose_output.detach()), 1)
        x = features
        output = x
        for i in range(self.n_stacked_convs):
            layer_name = self._get_layer_name(i)
            x = getattr(self, layer_name)(x)
            # x = getattr(self, layer_name)(x,offset)
            x = F.relu(x)
            output = x
        if self.use_dynamic_conv and densepose_predictor_outputs.u.size(0) != 0:
            # i_max = F.softmax(densepose_predictor_outputs.fine_segm, dim=1).max(dim=1, keepdim=True)[0]
            # i_min = F.softmax(densepose_predictor_outputs.fine_segm, dim=1).min(dim=1, keepdim=True)[0]
            # stat_2 = i_max - i_min
            # stat = torch.cat([densepose_predictor_outputs.fine_segm, densepose_predictor_outputs.u, densepose_predictor_outputs.v], dim=1)
            top_values, top_index = F.softmax(densepose_predictor_outputs.fine_segm, dim=1).topk(4, dim=1, largest=True, sorted=True)
            stat_1 = torch.cat([top_values, top_values.mean(dim=1, keepdim=True)], dim=1)

            stat_2 = self._local_linear(densepose_predictor_outputs.fine_segm, densepose_predictor_outputs.u, densepose_predictor_outputs.v)
            # stat = torch.cat([stat_i, stat_iuv], dim=1)
            # batch_size, _, height, width = x.size()
            # x = x.view(1, -1, height, width)
            # i_est = torch.argmax(densepose_predictor_outputs.fine_segm, dim=1, keepdim=True)
            # u = densepose_predictor_outputs.u.gather(1, i_est)
            # v = densepose_predictor_outputs.v.gather(1, i_est)
            # densepose_output = torch.cat((densepose_predictor_outputs.fine_segm, densepose_predictor_outputs.u, densepose_predictor_outputs.v), 1)
            # dynamix_weights = self.dynamic_weights(densepose_output)
            # x = F.conv2d(x, weight=dynamix_weights, bias=None, stride=1, padding=0, dilation=1, groups=batch_size)
            # x = x.view(batch_size, self.dynamic_channels, x.size(-2), x.size(-1))
            # x = self.instance_bn(x)
            # x = self.relu(x)
            identity = x
            out = self.dynamic_conv1(stat_1, x)
            out = self.instance_bn1(out)
            out = self.relu(out)
            out = self.dynamic_conv2(stat_2, out)
            out = self.instance_bn2(out)
            out += identity
            out = self.relu(out)
            output = out
            # output +=  self.gamma*out

        # output = x

        # ten conv
        # if x.size(0) == 0:
        #     offset = x.detach()
        # else:
        #     offset = torch.empty((1, self.offset_size, 1, 1), device=x.device)
        #     k=0
        #     for c in torch.arange(self.offset_pad_size*2+1):
        #         offset[0, c * 2 + 0, 0, 0] = 0
        #         offset[0, c * 2 + 1, 0, 0] = c-self.offset_pad_size
        #         if c == self.offset_pad_size:
        #             continue
        #         offset[0, (k+self.offset_pad_size*2+1) * 2 + 0, 0, 0] = c-self.offset_pad_size
        #         offset[0, (k+self.offset_pad_size*2+1) * 2 + 1, 0, 0] = 0
        #         k += 1
        #     # for c, (rel_offset_h, rel_offset_w) in enumerate([(0, -2), (0, -1), (0, 0), (0, 1), (0,2),(-2, 0), (-1, 0), (1, 0), (2, 0)]):
        #     #     offset[0, c * 2 + 0, 0, 0] = rel_offset_h
        #     #     offset[0, c * 2 + 1, 0, 0] = rel_offset_w
        #     offset = offset.repeat(x.size(0), 1, x.size(2), x.size(3))

        
        
        # if x.size(0) != 0:
        #     x = self.forward_crissAtten(x)
        # output = x
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
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
    def _local_linear(self, i, u, v):
        b,c,h,w = i.size()
        i_est = torch.argmax(i, dim=1) # Nx1xHxW
        i_est_mask = F.one_hot(i_est.long(), c).permute(0,3,1,2).float() # Nx25xHxW
        u_est = (u*i_est_mask)[:,1:] # Nx24xHxW
        v_est = (v*i_est_mask)[:,1:] # Nx24xHxW
        i_est_mask = i_est_mask[:,1:]
        u_est_mean = u_est.sum(dim=3, keepdim=True) / i_est_mask.sum(dim=3, keepdim=True)
        v_est_mean = v_est.sum(dim=2, keepdim=True) / i_est_mask.sum(dim=2, keepdim=True)
        u_var = torch.sqrt(torch.sum(((u_est - u_est_mean)**2)*i_est_mask, dim=3) / i_est_mask.sum(dim=3)) # Nx24xH
        v_var = torch.sqrt(torch.sum(((v_est - v_est_mean)**2)*i_est_mask, dim=2) / i_est_mask.sum(dim=2)) # Nx24xH
        var = torch.sqrt(torch.einsum('bch,bcw->bchw', u_var, v_var)) # Nx24xHxW
        state = var.sum(dim=1, keepdim=True) # # Nx1xHxW
        return (torch.exp(-state)).detach()
