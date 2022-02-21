# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any, List
import json
import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.structures import Instances

from .mask_or_segm import MaskOrSegmentationLoss
from .registry import DENSEPOSE_LOSS_REGISTRY
from .utils import (
    BilinearInterpolationHelper,
    ChartBasedAnnotationsAccumulator,
    PoiseNetBasedAnnotationsAccumulator,
    LossDict,
    extract_packed_annotations_from_matches,
)


@DENSEPOSE_LOSS_REGISTRY.register()
class DensePosePoiseNetLoss:
    """
    DensePose loss for chart-based training. A mesh is split into charts,
    each chart is given a label (I) and parametrized by 2 coordinates referred to
    as U and V. Ground truth consists of a number of points annotated with
    I, U and V values and coarse segmentation S defined for all pixels of the
    object bounding box. In some cases (see `COARSE_SEGM_TRAINED_BY_MASKS`),
    semantic segmentation annotations can be used as ground truth inputs as well.

    Estimated values are tensors:
     * U coordinates, tensor of shape [N, C, S, S]
     * V coordinates, tensor of shape [N, C, S, S]
     * fine segmentation estimates, tensor of shape [N, C, S, S] with raw unnormalized
       scores for each fine segmentation label at each location
     * coarse segmentation estimates, tensor of shape [N, D, S, S] with raw unnormalized
       scores for each coarse segmentation label at each location
    where N is the number of detections, C is the number of fine segmentation
    labels, S is the estimate size ( = width = height) and D is the number of
    coarse segmentation channels.

    The losses are:
    * regression (smooth L1) loss for U and V coordinates
    * cross entropy loss for fine (I) and coarse (S) segmentations
    Each loss has an associated weight
    """

    def __init__(self, cfg: CfgNode):
        """
        Initialize chart-based loss from configuration options

        Args:
            cfg (CfgNode): configuration options
        """
        # fmt: off
        self.heatmap_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE
        self.w_points     = cfg.MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS
        self.w_part       = cfg.MODEL.ROI_DENSEPOSE_HEAD.PART_WEIGHTS
        self.w_segm       = cfg.MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS
        self.n_segm_chan  = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
        # fmt: on
        self.segm_trained_by_masks = cfg.MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS
        self.segm_loss = MaskOrSegmentationLoss(cfg)

        self.part_loss_type  = cfg.MODEL.POISENET.PART_LOSS_TYPE
        self.freq_classes    = cfg.MODEL.POISENET.FREQ_CLASSES
        self.n_i_chan        = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES
        self.gamma           = 0.99

        block_fpath = cfg.MODEL.POISENET.BLOCK_FPATH
        with open(block_fpath, "rb") as bFile:
            self.block_file = json.loads(json.load(bFile))
        self.block_num       = cfg.MODEL.POISENET.BLOCK_NUM
        self.w_poise_cls     = cfg.MODEL.POISENET.POISE_CLS_WEIGHTS
        self.w_poise_reg     = cfg.MODEL.POISENET.POISE_REGRESSION_WEIGHTS

        self.smoothing       = 0.1
        self.confidense      = 1. - self.smoothing

    def __call__(
        self, proposals_with_gt: List[Instances], densepose_predictor_outputs: Any, **kwargs
    ) -> LossDict:
        """
        Produce chart-based DensePose losses

        Args:
            proposals_with_gt (list of Instances): detections with associated ground truth data
            densepose_predictor_outputs: an object of a dataclass that contains predictor outputs
                with estimated values; assumed to have the following attributes:
                * coarse_segm - coarse segmentation estimates, tensor of shape [N, D, S, S]
                * fine_segm - fine segmentation estimates, tensor of shape [N, C, S, S]
                * u - U coordinate estimates per fine labels, tensor of shape [N, C, S, S]
                * v - V coordinate estimates per fine labels, tensor of shape [N, C, S, S]
            where N is the number of detections, C is the number of fine segmentation
            labels, S is the estimate size ( = width = height) and D is the number of
            coarse segmentation channels.

        Return:
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_U`: smooth L1 loss for U coordinate estimates
             * `loss_densepose_V`: smooth L1 loss for V coordinate estimates
             * `loss_densepose_I`: cross entropy for raw unnormalized scores for fine
                 segmentation estimates given ground truth labels;
             * `loss_densepose_S`: cross entropy for raw unnormalized scores for coarse
                 segmentation estimates given ground truth labels;
        """
        # densepose outputs are computed for all images and all bounding boxes;
        # i.e. if a batch has 4 images with (3, 1, 2, 1) proposals respectively,
        # the outputs will have size(0) == 3+1+2+1 == 7

        if not len(proposals_with_gt):
            return self.produce_fake_densepose_losses(densepose_predictor_outputs)

        accumulator = PoiseNetBasedAnnotationsAccumulator()
        packed_annotations = extract_packed_annotations_from_matches(proposals_with_gt, accumulator, block_file=self.block_file)

        # NOTE: we need to keep the same computation graph on all the GPUs to
        # perform reduction properly. Hence even if we have no data on one
        # of the GPUs, we still need to generate the computation graph.
        # Add fake (zero) loss in the form Tensor.sum() * 0
        if packed_annotations is None:
            return self.produce_fake_densepose_losses(densepose_predictor_outputs)

        h, w = densepose_predictor_outputs.u_cls.shape[2:]
        interpolator = BilinearInterpolationHelper.from_matches(
            packed_annotations,
            (h, w),
        )

        j_valid_fg = interpolator.j_valid * (  # pyre-ignore[16]
            packed_annotations.fine_segm_labels_gt > 0
        )
        if not torch.any(j_valid_fg):
            return self.produce_fake_densepose_losses(densepose_predictor_outputs)

        losses_uv = self.produce_densepose_losses_uv(
            proposals_with_gt,
            densepose_predictor_outputs,
            packed_annotations,
            interpolator,
            j_valid_fg,  # pyre-ignore[6]
        )

        losses_segm = self.produce_densepose_losses_segm(
            proposals_with_gt,
            densepose_predictor_outputs,
            packed_annotations,
            interpolator,
            j_valid_fg,  # pyre-ignore[6]
        )

        return {**losses_uv, **losses_segm}

    def produce_fake_densepose_losses(self, densepose_predictor_outputs: Any) -> LossDict:
        """
        Fake losses for fine segmentation and U/V coordinates. These are used when
        no suitable ground truth data was found in a batch. The loss has a value 0
        and is primarily used to construct the computation graph, so that
        `DistributedDataParallel` has similar graphs on all GPUs and can perform
        reduction properly.

        Args:
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have the following attributes:
             * fine_segm - fine segmentation estimates, tensor of shape [N, C, S, S]
             * u - U coordinate estimates per fine labels, tensor of shape [N, C, S, S]
             * v - V coordinate estimates per fine labels, tensor of shape [N, C, S, S]
        Return:
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_U`: has value 0
             * `loss_densepose_V`: has value 0
             * `loss_densepose_I`: has value 0
             * `loss_densepose_S`: has value 0
        """
        losses_uv = self.produce_fake_densepose_losses_uv(densepose_predictor_outputs)
        losses_segm = self.produce_fake_densepose_losses_segm(densepose_predictor_outputs)
        return {**losses_uv, **losses_segm}

    def produce_fake_densepose_losses_uv(self, densepose_predictor_outputs: Any) -> LossDict:
        """
        Fake losses for U/V coordinates. These are used when no suitable ground
        truth data was found in a batch. The loss has a value 0
        and is primarily used to construct the computation graph, so that
        `DistributedDataParallel` has similar graphs on all GPUs and can perform
        reduction properly.

        Args:
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have the following attributes:
             * u - U coordinate estimates per fine labels, tensor of shape [N, C, S, S]
             * v - V coordinate estimates per fine labels, tensor of shape [N, C, S, S]
        Return:
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_U`: has value 0
             * `loss_densepose_V`: has value 0
        """
        return {
            "loss_densepose_U_cls": densepose_predictor_outputs.u_cls.sum() * 0,
            "loss_densepose_U_offset": densepose_predictor_outputs.u_offset.sum() * 0,
            "loss_densepose_V_cls": densepose_predictor_outputs.v_cls.sum() * 0,
            "loss_densepose_V_offset": densepose_predictor_outputs.v_offset.sum() * 0,
        }

    def produce_fake_densepose_losses_segm(self, densepose_predictor_outputs: Any) -> LossDict:
        """
        Fake losses for fine / coarse segmentation. These are used when
        no suitable ground truth data was found in a batch. The loss has a value 0
        and is primarily used to construct the computation graph, so that
        `DistributedDataParallel` has similar graphs on all GPUs and can perform
        reduction properly.

        Args:
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have the following attributes:
             * fine_segm - fine segmentation estimates, tensor of shape [N, C, S, S]
             * coarse_segm - coarse segmentation estimates, tensor of shape [N, D, S, S]
        Return:
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_I`: has value 0
             * `loss_densepose_S`: has value 0, added only if `segm_trained_by_masks` is False
        """
        losses = {
            "loss_densepose_I": densepose_predictor_outputs.fine_segm.sum() * 0,
            "loss_densepose_S": self.segm_loss.fake_value(densepose_predictor_outputs),
        }
        return losses

    def produce_densepose_losses_uv(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        packed_annotations: Any,
        interpolator: BilinearInterpolationHelper,
        j_valid_fg: torch.Tensor,
    ) -> LossDict:
        """
        Compute losses for U/V coordinates: smooth L1 loss between
        estimated coordinates and the ground truth.

        Args:
            proposals_with_gt (list of Instances): detections with associated ground truth data
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have the following attributes:
             * u - U coordinate estimates per fine labels, tensor of shape [N, C, S, S]
             * v - V coordinate estimates per fine labels, tensor of shape [N, C, S, S]
        Return:
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_U`: smooth L1 loss for U coordinate estimates
             * `loss_densepose_V`: smooth L1 loss for V coordinate estimates
        """
        fine_segm_gt = packed_annotations.fine_segm_labels_gt -1
        
        u_gt = packed_annotations.u_gt
        v_gt = packed_annotations.v_gt
        u_gt_cls = packed_annotations.u_gt_cls
        u_gt_offsets = packed_annotations.u_gt_offsets
        v_gt_cls = packed_annotations.v_gt_cls
        v_gt_offsets = packed_annotations.v_gt_offsets

        est_shape = densepose_predictor_outputs.u_cls.shape
        u_est_cls = interpolator.extract_at_points(
            densepose_predictor_outputs.u_cls.reshape(est_shape[0], -1, self.block_num, est_shape[2], est_shape[3]),
            block = True,
            block_slice = slice(None),
            slice_fine_segm = fine_segm_gt,
            w_ylo_xlo=interpolator.w_ylo_xlo[:, None],
            w_ylo_xhi=interpolator.w_ylo_xhi[:, None],
            w_yhi_xlo=interpolator.w_yhi_xlo[:, None],
            w_yhi_xhi=interpolator.w_yhi_xhi[:, None],
        )[j_valid_fg,:]
        u_est_offsets = interpolator.extract_at_points(
            densepose_predictor_outputs.u_offset.reshape(est_shape[0], -1, self.block_num, est_shape[2], est_shape[3]),
            block = True,
            block_slice = u_gt_cls,
            slice_fine_segm = fine_segm_gt,
        )[j_valid_fg] # J
        v_est_cls = interpolator.extract_at_points(
            densepose_predictor_outputs.v_cls.reshape(est_shape[0], -1, self.block_num, est_shape[2], est_shape[3]),
            block = True,
            block_slice = slice(None),
            w_ylo_xlo=interpolator.w_ylo_xlo[:, None],
            w_ylo_xhi=interpolator.w_ylo_xhi[:, None],
            w_yhi_xlo=interpolator.w_yhi_xlo[:, None],
            w_yhi_xhi=interpolator.w_yhi_xhi[:, None],
            slice_fine_segm = fine_segm_gt,
        )[j_valid_fg,:]
        v_est_offsets = interpolator.extract_at_points(
            densepose_predictor_outputs.v_offset.reshape(est_shape[0], -1, self.block_num, est_shape[2], est_shape[3]),
            block = True,
            block_slice = v_gt_cls,
            slice_fine_segm = fine_segm_gt,
        )[j_valid_fg]

        u_gt_cls = u_gt_cls[j_valid_fg]
        v_gt_cls = v_gt_cls[j_valid_fg]
        u_gt_offsets = u_gt_offsets[j_valid_fg] # J*2
        v_gt_offsets = v_gt_offsets[j_valid_fg] 

        return {
            "loss_densepose_U_cls": torch.sum(self.labelSmoothing(u_est_cls, u_gt_cls.long()))*self.w_poise_cls,
            "loss_densepose_U_offset": F.smooth_l1_loss(u_est_offsets, u_gt_offsets, reduction="sum")*self.w_poise_reg,
            "loss_densepose_V_cls": torch.sum(self.labelSmoothing(v_est_cls, v_gt_cls.long()))*self.w_poise_cls,
            "loss_densepose_V_offset": F.smooth_l1_loss(v_est_offsets, v_gt_offsets, reduction="sum")*self.w_poise_reg,
        }
    def labelSmoothing(self, x, target, bce=False):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = F.nll_loss(logprobs, target)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = nll_loss*self.confidense + smooth_loss*self.smoothing
        return loss

    def produce_densepose_losses_segm(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        packed_annotations: Any,
        interpolator: BilinearInterpolationHelper,
        j_valid_fg: torch.Tensor,
    ) -> LossDict:
        """
        Losses for fine / coarse segmentation: cross-entropy
        for segmentation unnormalized scores given ground truth labels at
        annotated points for fine segmentation and dense mask annotations
        for coarse segmentation.

        Args:
            proposals_with_gt (list of Instances): detections with associated ground truth data
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have the following attributes:
             * fine_segm - fine segmentation estimates, tensor of shape [N, C, S, S]
             * coarse_segm - coarse segmentation estimates, tensor of shape [N, D, S, S]
        Return:
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_I`: cross entropy for raw unnormalized scores for fine
                 segmentation estimates given ground truth labels
             * `loss_densepose_S`: cross entropy for raw unnormalized scores for coarse
                 segmentation estimates given ground truth labels;
                 may be included if coarse segmentation is only trained
                 using DensePose ground truth; if additional supervision through
                 instance segmentation data is performed (`segm_trained_by_masks` is True),
                 this loss is handled by `produce_mask_losses` instead
        """
        fine_segm_gt = packed_annotations.fine_segm_labels_gt[
            interpolator.j_valid  # pyre-ignore[16]
        ]
        fine_segm_est = interpolator.extract_at_points(
            densepose_predictor_outputs.fine_segm,
            slice_fine_segm=slice(None),
            w_ylo_xlo=interpolator.w_ylo_xlo[:, None],  # pyre-ignore[16]
            w_ylo_xhi=interpolator.w_ylo_xhi[:, None],  # pyre-ignore[16]
            w_yhi_xlo=interpolator.w_yhi_xlo[:, None],  # pyre-ignore[16]
            w_yhi_xhi=interpolator.w_yhi_xhi[:, None],  # pyre-ignore[16]
        )[interpolator.j_valid, :]
        if self.part_loss_type == "AEQL":
            J = fine_segm_gt.shape[0]
            E = self.exclude_func(J)
            T = self.threshold_func(fine_segm_gt, J)
            y_t = F.one_hot(fine_segm_gt, fine_segm_est.shape[1])
            M = torch.max(fine_segm_est, dim=1, keepdim=True)[0]

            prob = torch.softmax(fine_segm_est, axis=1).detach()
            top_values, top_index = prob.topk(18, dim=1, largest=False, sorted=True)
            mi = fine_segm_est.gather(1, top_index[torch.arange(J),-1].unsqueeze(-1))

            correlation = torch.exp(-(fine_segm_est-mi)/(M-mi)).detach()
            correlation = correlation.scatter(1, top_index, 1.)
            eql_w = 1. - E * T * (1. - y_t)*correlation

            x = (fine_segm_est-M) - torch.log(torch.sum(eql_w*torch.exp(fine_segm_est-M), dim=1)).unsqueeze(1).repeat(1, fine_segm_est.shape[1])
            smooth_loss = -x.mean(dim=-1)
            return {
                "loss_densepose_I": torch.sum(F.nll_loss(x, fine_segm_gt.long())*self.confidense + smooth_loss*self.smoothing) * self.w_part,
                "loss_densepose_S": self.segm_loss(
                    proposals_with_gt, densepose_predictor_outputs, packed_annotations
                ) * self.w_segm,
            }
        return {
            "loss_densepose_I": F.cross_entropy(fine_segm_est, fine_segm_gt.long()) * self.w_part,
            "loss_densepose_S": self.segm_loss(
                proposals_with_gt, densepose_predictor_outputs, packed_annotations
            )
            * self.w_segm,
        }
    def exclude_func(self, J):
        weight = torch.zeros((J), dtype=torch.float).cuda()
        beta = torch.Tensor(weight.shape).cuda().uniform_(0,1)
        weight[beta < self.gamma] = 1
        weight = weight.view(J, 1).expand(J, self.n_i_chan+1)
        return weight

    def threshold_func(self, gt_classes, J): 
        weight = torch.zeros(self.n_i_chan+1).cuda()
        if isinstance(self.freq_classes, list):
            for c in self.freq_classes:
                weight[c] = 1.
        elif self.freq_classes == "gompertz_decay":
            freq = torch.tensor([0,0.04157,0.1245,0.0554,0.0515,0.0310,0.0304,0.0164,0.0167,0.0531,0.0527,0.0143,0.0158,0.0385,0.0397,0.0340,0.0358,0.0419,0.0441,0.0207,0.0262,0.0486,0.0461,0.0623,0.0586], dtype=float).cuda()
            weight = 1-torch.exp(-8*torch.exp(-100*freq))
        weight = weight.unsqueeze(0)
        weight = weight.repeat(J, 1)
        return weight
