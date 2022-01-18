# Copyright (c) Facebook, Inc. and its affiliates.

from .chart import DensePoseChartLoss
from .chart_with_confidences import DensePoseChartWithConfidenceLoss
from .cse import DensePoseCseLoss
from .registry import DENSEPOSE_LOSS_REGISTRY
from .poisenet import DensePosePoiseNetLoss


__all__ = [
    "DensePoseChartLoss",
    "DensePoseChartWithConfidenceLoss",
    "DensePoseCseLoss",
    "DENSEPOSE_LOSS_REGISTRY",
    "DensePosePoiseNetLoss",
]
