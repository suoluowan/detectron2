# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from torch import nn

from detectron2.config import CfgNode
from detectron2.layers import ConvTranspose2d, interpolate

from ...structures import DensePoseScoringPredictorOutput
from ..utils import initialize_module_params
from .registry import DENSEPOSE_PREDICTOR_REGISTRY

from detectron2.layers import Conv2d, DeformConv
from torch.nn import Softmax, Sigmoid
from typing import Any
from torch.nn import functional as F


@DENSEPOSE_PREDICTOR_REGISTRY.register()
class DensePoseScoringPredictor(nn.Module):


    def __init__(self, cfg: CfgNode, input_channels: int):

        super().__init__()
        dim_in = input_channels
        num_classes = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.SCORING_CLS_NUM
        kernel_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECONV_KERNEL
        self.scale_factor = cfg.MODEL.ROI_DENSEPOSE_HEAD.UP_SCALE

        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.scoring_lowres = nn.Linear(dim_in, num_classes)
        # # initialize_module_params(self)
        # nn.init.normal_(self.scoring_lowres.weight, mean=0, std=0.01)
        # nn.init.constant_(self.scoring_lowres.bias, 0)

        conf_vector = [Conv2d(5, 64, 1)]
        conf_vector += [nn.ReLU(inplace=True)]
        conf_vector += [Conv2d(64, num_classes, 1)]
        self.i_conf = nn.Sequential(*conf_vector)
        # self.sigmoid = Sigmoid()
        self.gamma = nn.Parameter(torch.zeros(1))
 
        self.scoring_lowres = ConvTranspose2d(
            dim_in, num_classes, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maskiou_fc1 = nn.Linear(dim_in, 256)
        self.maskiou_fc2 = nn.Linear(256, 256)
        self.mask_iou = nn.Linear(256, 1)
        self.i_acc = nn.Linear(256, 1)
        self.uv_acc = nn.Linear(256, 1)

        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(self.scoring_lowres.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # initialize_module_params(self)
    
    def interp2d(self, tensor_nchw: torch.Tensor):
        return interpolate(
            tensor_nchw, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )


    def forward(self, head_outputs: torch.Tensor, densepose_predictor_outputs: Any):
        # scoring_cls_score = self.sigmoid(self.interp2d(self.scoring_lowres(head_outputs)))
        if densepose_predictor_outputs.fine_segm.size(0) != 0:
            # stat_iuv = self._static_iuv_prediction(densepose_predictor_outputs.fine_segm, densepose_predictor_outputs.u, densepose_predictor_outputs.v)
            # stat = self._local_linear(densepose_predictor_outputs.fine_segm, densepose_predictor_outputs.u, densepose_predictor_outputs.v)
            top_values, top_index = F.softmax(densepose_predictor_outputs.fine_segm, dim=1).topk(4, dim=1, largest=True, sorted=True)
            stat = torch.cat([top_values, top_values.mean(dim=1, keepdim=True)], dim=1)
            # stat = torch.cat([stat_iuv, stat_i], dim=1)
            # i_max = F.softmax(densepose_predictor_outputs.fine_segm, dim=1).max(dim=1, keepdim=True)[0]
            # i_min = F.softmax(densepose_predictor_outputs.fine_segm, dim=1).min(dim=1, keepdim=True)[0]
            # stat = torch.cat([densepose_predictor_outputs.fine_segm, densepose_predictor_outputs.u, densepose_predictor_outputs.v], dim=1)

            quality_score = self.i_conf(stat)

        else:
            quality_score = 0.

        mask_outputs = self.avgpool(head_outputs)
        mask_outputs = torch.flatten(mask_outputs, 1)
        mask_outputs = F.relu(self.maskiou_fc1(mask_outputs))
        mask_outputs = F.relu(self.maskiou_fc2(mask_outputs))

        densepose_outputs = self.interp2d(self.scoring_lowres(head_outputs))

        return DensePoseScoringPredictorOutput(
            # densepose_score = densepose_outputs,
            densepose_score= densepose_outputs + self.gamma*quality_score,
            # densepose_score=torch.sqrt(scoring_cls_score*quality_score),
            mask_score = self.mask_iou(mask_outputs),
            i_score = self.i_acc(mask_outputs),
            uv_score = self.uv_acc(mask_outputs),
        )
    
    def _static_iuv_prediction(self, i, u, v):
        i_est = torch.argmax(i, dim=1) # NxHxW
        i_est_one_hot = F.one_hot(i_est.long(), 25)[:,:,:,1:].permute(0,3,1,2).float() # Nx24xHxW
        i_est_h_static = i_est_one_hot.sum(dim=3) # Nx24xH
        i_est_w_static = i_est_one_hot.sum(dim=2) # Nx24xW
        position = torch.arange(1, i_est_h_static.size(-1)+1, device=i_est_h_static.device, dtype=torch.float)[None,None,:]
        i_h_mean = (i_est_h_static * position) / i_est_h_static.sum(dim=-1, keepdim=True)
        i_h_mean[torch.isnan(i_h_mean)] = 0.
        i_w_mean = (i_est_w_static * position) / i_est_w_static.sum(dim=-1, keepdim=True)
        i_w_mean[torch.isnan(i_w_mean)] = 0.
        i_h_var = ((position - i_h_mean)**2)*((i_est_h_static>0).float()) # Nx24xH
        i_w_var = ((position - i_w_mean)**2)*((i_est_w_static>0).float()) # Nx24xW
        # i_stat_h = torch.cat([i_h_var, i_h_var.mean(dim=-1, keepdim=True)], dim=-1) # Nx24x(H+1)
        # i_stat_w = torch.cat([i_w_var, i_w_var.mean(dim=-1, keepdim=True)], dim=-1) # Nx24x(W+1)
        
        top_values_h, top_index_h = i_est_h_static.topk(4, dim=-1, largest=True, sorted=True) # Nx24x4
        # i_stat_h = torch.cat([top_values_h, top_values_h.mean(dim=-1, keepdim=True)], dim=-1) # Nx24x5
        top_values_w, top_index_w = i_est_w_static.topk(4, dim=-1, largest=True, sorted=True) # Nx24x4
        # i_stat_w = torch.cat([top_values_w, top_values_w.mean(dim=-1, keepdim=True)], dim=-1) # Nx24x5

        u_est = u.gather(1, i_est.unsqueeze(1)).repeat(1,24,1,1) # Nx24xHxW
        v_est = v.gather(1, i_est.unsqueeze(1)).repeat(1,24,1,1) # Nx24xHxW
        u_local = u_est.gather(2, top_index_h.unsqueeze(-1).repeat(1,1,1,u.size(3))).gather(3, top_index_w.unsqueeze(-2).repeat(1,1,4,1)) # Nx24x4x4
        v_local = v_est.gather(2, top_index_h.unsqueeze(-1).repeat(1,1,1,u.size(3))).gather(3, top_index_w.unsqueeze(-2).repeat(1,1,4,1)) # Nx24x4x4
        u_stat_w = (u_local.quantile(dim=2, q=0.5) - u_local.mean(dim=2))**2 # Nx24x4
        # u_stat_w = torch.cat([u_stat_w, u_stat_w.mean(dim=-1, keepdim=True)], dim=-1) # Nx24x5
        u_stat_h = (u_local.quantile(dim=3, q=0.5) - u_local.mean(dim=3))**2 # Nx24x4
        # u_stat_h = torch.cat([u_stat_h, u_stat_h.mean(dim=-1, keepdim=True)], dim=-1) # Nx24x5
        v_stat_w = (v_local.quantile(dim=2, q=0.5) - v_local.mean(dim=2))**2 # Nx24x4
        # v_stat_w = torch.cat([v_stat_w, v_stat_w.mean(dim=-1, keepdim=True)], dim=-1) # Nx24x5
        v_stat_h = (v_local.quantile(dim=3, q=0.5) - v_local.mean(dim=3))**2 # Nx24x4
        # v_stat_h = torch.cat([v_stat_h, v_stat_h.mean(dim=-1, keepdim=True)], dim=-1) # Nx24x5
        iuv_h_var = i_h_var.gather(-1, top_index_h)+(u_stat_h+v_stat_h)
        iuv_w_var = i_w_var.gather(-1, top_index_w)+(u_stat_w+v_stat_w)
        h_var = i_h_var.scatter(-1,top_index_h,iuv_h_var)  # Nx24xH
        w_var = i_w_var.scatter(-1,top_index_w,iuv_w_var)  # Nx24xW

        state = torch.einsum('bih,biw->bihw', i_h_var, i_w_var) # Nx24xHxW
        return (torch.exp(-state)).detach()
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



