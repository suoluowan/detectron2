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
        num_classes = 1

        self.scoring_lowres = nn.Linear(dim_in, num_classes)
        # initialize_module_params(self)
        nn.init.normal_(self.scoring_lowres.weight, mean=0, std=0.01)
        nn.init.constant_(self.scoring_lowres.bias, 0)


    def forward(self, head_outputs: torch.Tensor):
        if head_outputs.size(0) == 0:
            print(head_outputs.size())
            return DensePoseScoringPredictorOutput(
            densepose_score=head_outputs,
            )
        return DensePoseScoringPredictorOutput(
            densepose_score=self.scoring_lowres(head_outputs),
        )
