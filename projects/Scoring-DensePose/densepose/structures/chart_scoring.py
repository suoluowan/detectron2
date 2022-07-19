# Copyright (c) Facebook, Inc. and its affiliates.

from dataclasses import dataclass
from typing import Union
import torch


@dataclass
class DensePoseScoringPredictorOutput:

    densepose_score: torch.Tensor
    mask_score: torch.Tensor
    i_score: torch.Tensor
    uv_score: torch.Tensor

    def __len__(self):
        """
        Number of instances (N) in the output
        """
        return self.densepose_score.size(0)

    def __getitem__(
        self, item: Union[int, slice, torch.BoolTensor]
    ) -> "DensePoseScoringPredictorOutput":

        if isinstance(item, int):
            return DensePoseScoringPredictorOutput(
                densepose_score=self.densepose_score[item].unsqueeze(0),
                mask_score=self.mask_score[item].unsqueeze(0),
                i_score=self.i_score[item].unsqueeze(0),
                uv_score=self.uv_score[item].unsqueeze(0),

            )
        else:
            return DensePoseScoringPredictorOutput(
                densepose_score=self.densepose_score[item],
                mask_score=self.mask_score[item],
                i_score=self.i_score[item],
                uv_score=self.uv_score[item],
            )

    def to(self, device: torch.device):
        densepose_score = self.densepose_score.to(device)
        mask_score = self.mask_score.to(device)
        return DensePoseScoringPredictorOutput(
            densepose_score=densepose_score, 
            mask_score=mask_score, 
            i_score=i_score, 
            uv_score=uv_score,
        )
