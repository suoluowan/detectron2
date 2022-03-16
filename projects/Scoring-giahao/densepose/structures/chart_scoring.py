# Copyright (c) Facebook, Inc. and its affiliates.

from dataclasses import dataclass
from typing import Union
import torch


@dataclass
class DensePoseScoringPredictorOutput:

    densepose_score: torch.Tensor

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
            )
        else:
            return DensePoseScoringPredictorOutput(
                densepose_score=self.densepose_score[item],
            )

    def to(self, device: torch.device):
        densepose_score = self.densepose_score.to(device)
        return DensePoseScoringPredictorOutput(densepose_score=densepose_score)
