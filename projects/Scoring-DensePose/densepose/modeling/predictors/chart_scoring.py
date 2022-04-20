# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from torch import nn

from detectron2.config import CfgNode
from detectron2.layers import ConvTranspose2d, interpolate

from ...structures import DensePoseScoringPredictorOutput
from ..utils import initialize_module_params
from .registry import DENSEPOSE_PREDICTOR_REGISTRY


@DENSEPOSE_PREDICTOR_REGISTRY.register()
class DensePoseScoringPredictor(nn.Module):


    def __init__(self, cfg: CfgNode, input_channels: int):

        super().__init__()
        dim_in = input_channels
        num_classes = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.SCORING_CLS_NUM
        kernel_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECONV_KERNEL
        self.scale_factor = cfg.MODEL.ROI_DENSEPOSE_HEAD.UP_SCALE

        # self.scoring_lowres = nn.Linear(dim_in, num_classes)
        # # initialize_module_params(self)
        # nn.init.normal_(self.scoring_lowres.weight, mean=0, std=0.01)
        # nn.init.constant_(self.scoring_lowres.bias, 0)

        self.scoring_lowres = ConvTranspose2d(
            dim_in, num_classes, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
        )
        initialize_module_params(self)
    
    def interp2d(self, tensor_nchw: torch.Tensor):
        return interpolate(
            tensor_nchw, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )


    def forward(self, head_outputs: torch.Tensor):
        return DensePoseScoringPredictorOutput(
            # densepose_score=self.scoring_lowres(head_outputs),
            densepose_score=self.interp2d(self.scoring_lowres(head_outputs)),
        )
