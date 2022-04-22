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

def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


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

        self.query_conv = Conv2d(n_channels, n_channels//8, kernel_size=1)
        self.key_conv = Conv2d(n_channels, n_channels//8, kernel_size=1)
        self.value_conv = Conv2d(n_channels, n_channels, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

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
    def forward_crissAtten(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        return self.gamma*(out_H + out_W) + x


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

        # ten conv
        # if x.size(0) == 0:
        #     offset = x.detach()
        # else:
        #     offset = torch.empty((1, self.offset_size, 1, 1), device=x.device)
            # for c in torch.arange(self.offset_pad_size*2+1):
            #     offset[0, c * 2 + 0, 0, 0] = 0
            #     offset[0, c * 2 + 1, 0, 0] = c-self.offset_pad_size
            # for c in torch.arange(self.offset_pad_size*2+1):
            #     offset[0, (c+self.offset_pad_size*2+1) * 2 + 0, 0, 0] = c-self.offset_pad_size
            #     offset[0, (c+self.offset_pad_size*2+1) * 2 + 1, 0, 0] = 0
            # for c, (rel_offset_h, rel_offset_w) in enumerate([(0, -2), (0, -1), (0, 0), (0, 1), (0,2),(-2, 0), (-1, 0), (1, 0), (2, 0)]):
            #     offset[0, c * 2 + 0, 0, 0] = rel_offset_h
            #     offset[0, c * 2 + 1, 0, 0] = rel_offset_w
            # offset = offset.repeat(x.size(0), 1, x.size(2), x.size(3))

        for i in range(self.n_stacked_convs):
            layer_name = self._get_layer_name(i)
            x = getattr(self, layer_name)(x)
            # x = getattr(self, layer_name)(x,offset)
            x = F.relu(x)
            output = x
        if x.size(0) != 0:
            x = self.forward_crissAtten(x)
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
