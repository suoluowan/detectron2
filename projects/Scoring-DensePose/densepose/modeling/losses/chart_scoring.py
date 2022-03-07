# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any, List
import torch
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.structures import Instances

from .mask_or_segm import MaskOrSegmentationLoss
from .registry import DENSEPOSE_LOSS_REGISTRY
from .utils import (
    BilinearInterpolationHelper,
    ChartBasedAnnotationsAccumulator,
    LossDict,
    extract_packed_annotations_from_matches,
    l2_loss,
    gfocal_loss,
    vfocal_loss,
)

import scipy.spatial.distance as ssd
from scipy.io import loadmat
import numpy as np
import pickle
import copy

@DENSEPOSE_LOSS_REGISTRY.register()
class DensePoseScoringLoss:

    def __init__(self, cfg: CfgNode):
        self.loss_weight     = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.LOSS_WEIGHT

        smpl_subdiv_fpath = '/home/sunjunyao/tmp/smpl/SMPL_subdiv.mat'
        pdist_transform_fpath = '/home/sunjunyao/tmp/smpl/SMPL_SUBDIV_TRANSFORM.mat'
        pdist_matrix_fpath = '/home/sunjunyao/tmp/smpl/Pdist_matrix.pkl'
        SMPL_subdiv = loadmat(smpl_subdiv_fpath)
        PDIST_transform = loadmat(pdist_transform_fpath)["index"].astype(np.int32)
        self.PDIST_transform = torch.from_numpy(PDIST_transform.squeeze()).cuda()
        UV = np.array([SMPL_subdiv["U_subdiv"], SMPL_subdiv["V_subdiv"]]).squeeze()
        ClosestVertInds = np.arange(UV.shape[1]) + 1
        self.Part_UVs = []
        self.Part_ClosestVertInds = []
        for i in range(24):
            self.Part_UVs.append(UV[:, SMPL_subdiv["Part_ID_subdiv"].squeeze() == (i + 1)])
            self.Part_ClosestVertInds.append(
                ClosestVertInds[SMPL_subdiv["Part_ID_subdiv"].squeeze() == (i + 1)]
            )
        with open(pdist_matrix_fpath, "rb") as hFile:
            arrays = pickle.load(hFile, encoding="latin1")
        self.Pdist_matrix = torch.from_numpy(arrays["Pdist_matrix"]).cuda()
        # self.Pdist_matrix = arrays["Pdist_matrix"]
        self.Part_ids = torch.tensor(np.array(SMPL_subdiv["Part_ID_subdiv"].squeeze()), dtype=torch.int64).cuda()
        self.mean_dist = torch.tensor([0, 0.351, 0.351, 0.107, 0.107, 0.126, 0.126, 0.237, 0.237, 0.237, 0.237, 0.173, 0.173, 0.173, 0.173, 0.142, 0.142, 0.142, 0.142, 0.128, 0.128, 0.128, 0.128, 0.150, 0.150]).cuda()

    def __call__(
        self, proposals_with_gt: List[Instances], densepose_predictor_outputs: Any, densepose_scoring_predictor_outputs: Any, **kwargs
    ) -> LossDict:
        if len(densepose_scoring_predictor_outputs) == 0:
            return self.produce_fake_densepose_scoring_losses(densepose_scoring_predictor_outputs)
        if not len(proposals_with_gt):
            return self.produce_fake_densepose_scoring_losses(densepose_scoring_predictor_outputs)

        accumulator = ChartBasedAnnotationsAccumulator()
        packed_annotations = extract_packed_annotations_from_matches(proposals_with_gt, accumulator)

        if packed_annotations is None:
            return self.produce_fake_densepose_scoring_losses(densepose_scoring_predictor_outputs)

        h, w = densepose_predictor_outputs.u.shape[2:]
        interpolator = BilinearInterpolationHelper.from_matches(
            packed_annotations,
            (h, w),
        )

        j_valid_fg = interpolator.j_valid * (  # pyre-ignore[16]
            packed_annotations.fine_segm_labels_gt > 0
        )
        if not torch.any(j_valid_fg):
            return self.produce_fake_densepose_scoring_losses(densepose_scoring_predictor_outputs)

        losses_score = self.produce_densepose_scoring_losses(
            proposals_with_gt,
            densepose_predictor_outputs,
            densepose_scoring_predictor_outputs,
            packed_annotations,
            interpolator,
            j_valid_fg,  # pyre-ignore[6]
        )

        return {**losses_score}

    def produce_fake_densepose_scoring_losses(self, densepose_scoring_predictor_outputs: Any) -> LossDict:
        return {
            "loss_densepose_score": densepose_scoring_predictor_outputs.densepose_score.sum() * 0,
        }

    def produce_densepose_scoring_losses(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        densepose_scoring_predictor_outputs: Any,
        packed_annotations: Any,
        interpolator: BilinearInterpolationHelper,
        j_valid_fg: torch.Tensor,
    ) -> LossDict:
        score_est = densepose_scoring_predictor_outputs.densepose_score
        bbox_indices = packed_annotations.bbox_indices
        # print(score_est.shape)
        score_gt = self.getDensePoseScore(densepose_predictor_outputs, packed_annotations, interpolator, j_valid_fg)
        score_est = score_est[bbox_indices].squeeze(1)
        # score_est = torch.exp(-(dist_est**2)/(2 * (0.255 ** 2)))
        # print(score_est)
        # print(score_gt)

        return {
            "loss_densepose_score": vfocal_loss(score_est, score_gt) * self.loss_weight,
        }
    
    def getDensePoseScore(
        self,
        densepose_predictor_outputs: Any,
        packed_annotations: Any,
        interpolator: BilinearInterpolationHelper,
        j_valid_fg: torch.Tensor,
    ):
        u_gt = packed_annotations.u_gt[j_valid_fg]
        u_est = interpolator.extract_at_points(densepose_predictor_outputs.u)[j_valid_fg].detach()
        v_gt = packed_annotations.v_gt[j_valid_fg]
        v_est = interpolator.extract_at_points(densepose_predictor_outputs.v)[j_valid_fg].detach()
        i_gt = packed_annotations.fine_segm_labels_gt[j_valid_fg]
        fine_segm_est = interpolator.extract_at_points(
            densepose_predictor_outputs.fine_segm,
            slice_fine_segm=slice(None),
            w_ylo_xlo=interpolator.w_ylo_xlo[:, None],  # pyre-ignore[16]
            w_ylo_xhi=interpolator.w_ylo_xhi[:, None],  # pyre-ignore[16]
            w_yhi_xlo=interpolator.w_yhi_xlo[:, None],  # pyre-ignore[16]
            w_yhi_xhi=interpolator.w_yhi_xhi[:, None],  # pyre-ignore[16]
        )[j_valid_fg, :].detach()
        i_est = torch.argmax(fine_segm_est, axis=1)
        # print(i_est.shape)
        point_bbox_indices = packed_annotations.point_bbox_indices[j_valid_fg]
        bbox_indices = packed_annotations.bbox_indices
        # bbox_indices_valid = torch.ones_like(bbox_indices)
        score_gt = torch.zeros_like(bbox_indices, dtype=torch.float32)
        for i_bbox, bbox_index in enumerate(bbox_indices):
            i_point_indices = ((point_bbox_indices == bbox_index).nonzero(as_tuple=True)[0])
            if len(i_point_indices) == 0:
                continue
            i_u_gt = u_gt[i_point_indices]
            i_v_gt = v_gt[i_point_indices]
            i_i_gt = i_gt[i_point_indices]
            i_u_est = u_est[i_point_indices]
            i_v_est = v_est[i_point_indices]
            i_i_est = i_est[i_point_indices]
            cVerts, cVertsGT = self.findAllClosestVerts(i_u_gt,i_v_gt,i_i_gt,i_u_est,i_v_est,i_i_est)
            dist = self.getDistances(cVertsGT, cVerts)
            Current_Mean_Distances = torch.gather(self.mean_dist, 0, self.Part_ids[cVertsGT[cVertsGT > 0].long() - 1])
            i_scores_gt = torch.exp(
                            -(dist ** 2) / (2 * (Current_Mean_Distances ** 2))
                        )
            i_scores_gt[dist == 3.0] = 0
            score_gt[i_bbox] = torch.mean(i_scores_gt)
        return score_gt
    
    def findAllClosestVerts(self, U_gt, V_gt, I_gt, U_points, V_points, Index_points):

        ClosestVerts = torch.ones(Index_points.shape).cuda() * -1
        for i in range(24):
            #
            if (i + 1) in Index_points:
                UVs = torch.stack(( U_points[Index_points == (i + 1)], V_points[Index_points == (i + 1)]), dim=1)
                if len(UVs.shape) == 1:
                    UVs = UVs.unsqueeze(axis=1)
                Current_Part_UVs = torch.tensor(self.Part_UVs[i], dtype=torch.float64).cuda()
                Current_Part_ClosestVertInds = torch.tensor(self.Part_ClosestVertInds[i], dtype=torch.float32).cuda()
                D = torch.cdist(Current_Part_UVs.transpose(1,0), UVs.double())
                ClosestVerts[Index_points == (i + 1)] = Current_Part_ClosestVertInds[
                        torch.argmin(D.squeeze().float(), axis=0)]
        #
        ClosestVertsGT = torch.ones(I_gt.shape).cuda() * -1
        for i in range(24):
            if (i + 1) in I_gt:
                UVs = torch.stack((U_gt[I_gt == (i + 1)], V_gt[I_gt == (i + 1)]), dim=1)
                if len(UVs.shape) == 1:
                    UVs = UVs.unsqueeze(axis=1)
                Current_Part_UVs = torch.tensor(self.Part_UVs[i], dtype=torch.float64).cuda()
                Current_Part_ClosestVertInds = torch.tensor(self.Part_ClosestVertInds[i], dtype=torch.float32).cuda()
                D = torch.cdist(Current_Part_UVs.transpose(1,0), UVs.double())
                ClosestVertsGT[I_gt == (i + 1)] = Current_Part_ClosestVertInds[torch.argmin(D.squeeze().float(), axis=0)]

        return ClosestVerts, ClosestVertsGT

    def getDistances(self, cVertsGT, cVerts):

        ClosestVertsTransformed = self.PDIST_transform[cVerts.long() - 1]
        ClosestVertsGTTransformed = self.PDIST_transform[cVertsGT.long() - 1]
        #
        ClosestVertsTransformed[cVerts.long() < 0] = 0
        ClosestVertsGTTransformed[cVertsGT.long() < 0] = 0
        
        #
        cVertsGT = ClosestVertsGTTransformed.long()
        cVerts = ClosestVertsTransformed.long()

        index_cVertsGT = (cVertsGT > 0).nonzero().flatten().detach()
        cVerts_filter = cVerts[index_cVertsGT]
        cVertsGT = cVertsGT[index_cVertsGT]

        oulter = (cVerts_filter > 0).long().cuda()

        cVerts_filter = (cVerts_filter - 1)*oulter
        cVertsGT_filter = (cVertsGT - 1)*oulter

        dists = torch.zeros(len(cVerts_filter), dtype=torch.float32).cuda()

        cVerts_max = torch.max(cVertsGT_filter, cVerts_filter)
        cVerts_min = torch.min(cVertsGT_filter, cVerts_filter)
        dist_matrix = torch.true_divide(cVerts_max*(cVerts_max-1), 2) + cVerts_min
        
        dists[cVerts_filter != cVertsGT_filter] = self.Pdist_matrix[dist_matrix[cVerts_filter != cVertsGT_filter].long()].squeeze()
        dists = dists*oulter - (oulter-1.)*3.
        return dists
        # return torch.from_numpy(np.atleast_1d(np.array(dists.cpu()).squeeze())).float().cuda()
