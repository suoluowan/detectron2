# Copyright (c) Facebook, Inc. and its affiliates.

from .chart import DensePoseChartLoss
from .chart_with_confidences import DensePoseChartWithConfidenceLoss
from .cse import DensePoseCseLoss
from .registry import DENSEPOSE_LOSS_REGISTRY
from .chart_scoring import DensePoseScoringLoss


__all__ = [
    "DensePoseChartLoss",
    "DensePoseChartWithConfidenceLoss",
    "DensePoseCseLoss",
    "DensePoseScoringLoss",
    "DENSEPOSE_LOSS_REGISTRY",
]
