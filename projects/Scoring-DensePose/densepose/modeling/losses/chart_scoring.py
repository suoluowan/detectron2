# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any, List
import torch
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.structures import Instances

from .segm_iou import SegmentationIoULoss
from .registry import DENSEPOSE_LOSS_REGISTRY
from .utils import (
    BilinearInterpolationHelper,
    ChartBasedAnnotationsAccumulator,
    LossDict,
    extract_packed_annotations_from_matches,
    l2_loss,
    gfocal_loss,
    vfocal_loss,
    resample_data,
)

import scipy.spatial.distance as ssd
from scipy.io import loadmat
import numpy as np
import pickle
import copy
from pycm import *

@DENSEPOSE_LOSS_REGISTRY.register()
class DensePoseScoringLoss:

    def __init__(self, cfg: CfgNode):
        self.loss_weight     = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.LOSS_WEIGHT
        self.heatmap_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE
        self.loss_weight_instance     = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.LOSS_WEIGHT
        self.loss_weight_point     = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.LOSS_WEIGHT_POINT
        self.score_cls_num = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.SCORING_CLS_NUM

        self.segm_iou_loss = SegmentationIoULoss(cfg)
        self.loss_weight_segm = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.LOSS_WEIGHT_SEGM

        self.loss_weight_acc = cfg.MODEL.ROI_DENSEPOSE_HEAD.SCORING.LOSS_WEIGHT_ACC

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
            return self.produce_fake_scoring_losses(densepose_scoring_predictor_outputs)
        if not len(proposals_with_gt):
            return self.produce_fake_scoring_losses(densepose_scoring_predictor_outputs)

        accumulator = ChartBasedAnnotationsAccumulator()
        packed_annotations = extract_packed_annotations_from_matches(proposals_with_gt, accumulator)

        if packed_annotations is None:
            return self.produce_fake_scoring_losses(densepose_scoring_predictor_outputs)

        h, w = densepose_predictor_outputs.u.shape[2:]
        interpolator = BilinearInterpolationHelper.from_matches(
            packed_annotations,
            (h, w),
        )

        j_valid_fg = interpolator.j_valid * (  # pyre-ignore[16]
            packed_annotations.fine_segm_labels_gt > 0
        )
        if not torch.any(j_valid_fg):
            return self.produce_fake_scoring_losses(densepose_scoring_predictor_outputs)

        losses_densepose_score = self.produce_densepose_scoring_losses(
            proposals_with_gt,
            densepose_predictor_outputs,
            densepose_scoring_predictor_outputs,
            packed_annotations,
            interpolator,
            j_valid_fg,  # pyre-ignore[6]
        )
        losses_mask_score = self.produce_mask_scoring_losses(
            proposals_with_gt,
            densepose_predictor_outputs,
            densepose_scoring_predictor_outputs,
            packed_annotations,
            interpolator,
            j_valid_fg,  # pyre-ignore[6]
        )


        return {
            **losses_densepose_score, 
            **losses_mask_score,
        }

    def produce_fake_scoring_losses(self, densepose_scoring_predictor_outputs: Any) -> LossDict:
        # return {
        #     "loss_densepose_score": densepose_scoring_predictor_outputs.densepose_score.sum() * 0,
        # }
        losses_densepose_score = self.produce_fake_densepose_scoring_losses(densepose_scoring_predictor_outputs)
        losses_mask_score = self.produce_fake_mask_scoring_losses(densepose_scoring_predictor_outputs)

        return {
            **losses_densepose_score, 
            **losses_mask_score,
        }

    def produce_fake_densepose_scoring_losses(self, densepose_scoring_predictor_outputs: Any) -> LossDict:
        # return {
        #     "loss_densepose_score": densepose_scoring_predictor_outputs.densepose_score.sum() * 0,
        # }

        return {
        #     "loss_densepose_score_point": densepose_scoring_predictor_outputs.densepose_score.sum() * 0,
        #     "loss_densepose_score_instance": densepose_scoring_predictor_outputs.densepose_score.sum() * 0,
            "loss_densepose_score": densepose_scoring_predictor_outputs.densepose_score.sum() * 0, 
            "loss_i_score": densepose_scoring_predictor_outputs.i_score.sum() * 0, 
            "loss_uv_score": densepose_scoring_predictor_outputs.uv_score.sum() * 0, 

        }
    def produce_fake_mask_scoring_losses(self, densepose_scoring_predictor_outputs: Any) -> LossDict:
        # return {
        #     "loss_densepose_score": densepose_scoring_predictor_outputs.densepose_score.sum() * 0,
        # }

        return {
            "loss_mask_score": self.segm_iou_loss.fake_value(densepose_scoring_predictor_outputs),
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
        # score_est = densepose_scoring_predictor_outputs.densepose_score
        # bbox_indices = packed_annotations.bbox_indices
        # score_gt = self.getDensePoseScore(densepose_predictor_outputs, packed_annotations, interpolator, j_valid_fg)
        # score_est = score_est[bbox_indices].squeeze(1)

        # return {
        #     "loss_densepose_score": l2_loss(score_est, score_gt) * self.loss_weight,
        #     # "loss_densepose_score": F.smooth_l1_loss(score_est, score_gt, reduction="sum") * self.loss_weight,
        # }

        score_est = densepose_scoring_predictor_outputs.densepose_score
        bbox_indices = packed_annotations.bbox_indices
        score_est_point = interpolator.extract_at_points(
            score_est,
            slice_fine_segm=slice(None),
            w_ylo_xlo=interpolator.w_ylo_xlo[:, None],  # pyre-ignore[16]
            w_ylo_xhi=interpolator.w_ylo_xhi[:, None],  # pyre-ignore[16]
            w_yhi_xlo=interpolator.w_yhi_xlo[:, None],  # pyre-ignore[16]
            w_yhi_xhi=interpolator.w_yhi_xhi[:, None],  # pyre-ignore[16]
        )[j_valid_fg, :].squeeze(1)
        score_gt_instance, score_gt_point, index_valid, i_acc_gt, uv_acc_gt = self.getDensePoseScore(densepose_predictor_outputs, packed_annotations, interpolator, j_valid_fg)
        # score_gt_point = score_gt_point - 0.5
        # score_gt_point[score_gt_point<0] = 0
        # score_gt_point = torch.ceil(score_gt_point*(self.score_cls_num-1)/0.5)
        score_gt_point = torch.floor(score_gt_point*self.score_cls_num)
        score_gt_point[score_gt_point>=self.score_cls_num] = self.score_cls_num - 1
        # score_gt_point = F.one_hot(score_gt_point.long(), self.score_cls_num)
        score_est_point = score_est_point[index_valid]
        # print(score_est_point.shape, torch.sum(score_gt_point<10))
        # with torch.no_grad():
        #     coarse_segm_gt = resample_data(
        #         packed_annotations.coarse_segm_gt.unsqueeze(1),
        #         packed_annotations.bbox_xywh_gt,
        #         packed_annotations.bbox_xywh_est,
        #         self.heatmap_size,
        #         self.heatmap_size,
        #         mode="nearest",
        #         padding_mode="zeros",
        #     )
        # coarse_segm_gt = coarse_segm_gt > 0
        # score_est_instance = torch.sum((score_est[bbox_indices]*coarse_segm_gt), (2,3))/torch.sum(coarse_segm_gt, (2,3))
        # print(torch.sum((score_est[bbox_indices].squeeze(1))))
        # print(torch.sum(coarse_segm_gt, (1,2)))
        # print(score_est_instance, score_gt_instance)
        # score_est_instance = score_est[bbox_indices]

        i_acc_est = densepose_scoring_predictor_outputs.i_score
        uv_acc_est = densepose_scoring_predictor_outputs.uv_score
        
        return {
            # "loss_densepose_score_point": l2_loss(score_est_point, score_gt_point) * self.loss_weight_point,
            # "loss_densepose_score_instance": l2_loss(score_est_instance, score_gt_instance) * self.loss_weight_instance,
            "loss_densepose_score": F.cross_entropy(score_est_point, score_gt_point.long()) * self.loss_weight_point,  
            # "loss_densepose_score": F.binary_cross_entropy(score_est_point, score_gt_point.float()) * self.loss_weight_point, 
            "loss_i_score": l2_loss(i_acc_est, i_acc_gt) * self.loss_weight_acc,  
            "loss_uv_score": l2_loss(uv_acc_est, uv_acc_gt) * self.loss_weight_acc,  
        }

        # focal loss
        # pred_softmax = torch.softmax(score_est_point, axis=1)
        # # pred_softmax = pred_softmax.gather(1,fine_segm_gt.view(-1,1)) 
        # pred_softmax = F.nll_loss(pred_softmax, score_gt_point.long())
        # preds_logsoft = F.cross_entropy(score_est_point, score_gt_point.long(), reduction='none')
        # loss = torch.mul(torch.pow((1-pred_softmax), 2), preds_logsoft) 
        # return {
        #     "loss_densepose_score": loss.mean() * self.loss_weight_point,
        #     }
    
    def produce_mask_scoring_losses(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        densepose_scoring_predictor_outputs: Any,
        packed_annotations: Any,
        interpolator: BilinearInterpolationHelper,
        j_valid_fg: torch.Tensor,
    ) -> LossDict:
        return {
            "loss_mask_score": self.segm_iou_loss(
                proposals_with_gt, densepose_predictor_outputs, densepose_scoring_predictor_outputs, packed_annotations
            )
            * self.loss_weight_segm,
        }
        

    def getDensePoseScore(
        self,
        densepose_predictor_outputs: Any,
        packed_annotations: Any,
        interpolator: BilinearInterpolationHelper,
        j_valid_fg: torch.Tensor,
    ):
        i_gt = packed_annotations.fine_segm_labels_gt[j_valid_fg]
        fine_segm_est = interpolator.extract_at_points(
            densepose_predictor_outputs.fine_segm,
            slice_fine_segm=slice(None),
            w_ylo_xlo=interpolator.w_ylo_xlo[:, None],  # pyre-ignore[16]
            w_ylo_xhi=interpolator.w_ylo_xhi[:, None],  # pyre-ignore[16]
            w_yhi_xlo=interpolator.w_yhi_xlo[:, None],  # pyre-ignore[16]
            w_yhi_xhi=interpolator.w_yhi_xhi[:, None],  # pyre-ignore[16]
        ).detach()
        i_est = torch.argmax(fine_segm_est, axis=1)
        u_gt = packed_annotations.u_gt[j_valid_fg]
        u_est = interpolator.extract_at_points(
            densepose_predictor_outputs.u,
            # slice_fine_segm=i_est,
            )[j_valid_fg].detach()
        v_gt = packed_annotations.v_gt[j_valid_fg]
        v_est = interpolator.extract_at_points(
            densepose_predictor_outputs.v,
            # slice_fine_segm=i_est,
            )[j_valid_fg].detach()
        i_est = i_est[j_valid_fg]
        point_bbox_indices = packed_annotations.point_bbox_indices[j_valid_fg]
        bbox_indices = packed_annotations.bbox_indices

        cVerts, cVertsGT = self.findAllClosestVerts(u_gt,v_gt,i_gt,u_est,v_est,i_est)
        dist,index_valid = self.getDistances(cVertsGT, cVerts)
        Current_Mean_Distances = torch.gather(self.mean_dist, 0, self.Part_ids[cVertsGT[index_valid].long() - 1])
        scores_gt_point = torch.exp(-(dist ** 2) / (2 * (Current_Mean_Distances ** 2)))
        scores_gt_point[dist == 3.0] = 0
        point_bbox_indices = point_bbox_indices[index_valid]
        score_gt_instance = torch.zeros_like(bbox_indices, dtype=torch.float32)
        # print("densepose_predictor_outputs.fine_segm", torch.sum(torch.isnan(densepose_predictor_outputs.fine_segm)), densepose_predictor_outputs.fine_segm.size())
        # print("fine_segm_est", torch.sum(torch.isnan(fine_segm_est)), fine_segm_est.size())
        # print("i_est", i_est)
        # print("cVerts", cVerts)
        # print("index_valid", index_valid)
        # print("bbox_indices", bbox_indices)
        
        # for i_bbox, bbox_index in enumerate(bbox_indices):
        #     i_point_indices = ((point_bbox_indices == bbox_index).nonzero(as_tuple=True)[0])
        #     # print(len(i_point_indices))
        #     if len(i_point_indices) == 0:
        #         continue
        #     # score_gt_instance[i_bbox] = torch.mean(scores_gt_point[i_point_indices])
        #     score_gt_instance[i_bbox] = torch.mean(scores_gt_point[i_point_indices])

        i_acc_gt = torch.zeros_like(bbox_indices, dtype=torch.float32)
        uv_acc_gt = torch.zeros_like(bbox_indices, dtype=torch.float32)
        for i_bbox, bbox_index in enumerate(bbox_indices):
            i_point_indices = ((point_bbox_indices == bbox_index).nonzero(as_tuple=True)[0])
            if len(i_point_indices) == 0:
                continue
            i_gt_b = i_gt[i_point_indices]
            i_est_b = i_est[i_point_indices]
            u_gt_b = u_gt[i_point_indices]
            u_est_b = u_est[i_point_indices]
            v_gt_b = v_gt[i_point_indices]
            v_est_b = v_est[i_point_indices]

            if len(torch.unique(i_gt_b)) == 1:
                i_acc_gt[i_bbox] = (i_est_b == i_gt_b[0]).sum()/len(i_est_b)
            else:
                cm = ConfusionMatrix(actual_vector=np.array(i_gt_b.cpu()).astype(np.int32), predict_vector=np.array(i_est_b.cpu()).astype(np.int32))
                i_acc_gt[i_bbox] = torch.tensor(cm.overall_stat['F1 Micro'], device=i_gt_b.device)
            dist_uv = ((u_est_b-u_gt_b)**2 + (v_est_b-v_gt_b)**2)**(1/2)
            uv_acc_gt[i_bbox] = torch.exp(-(dist_uv.mean())*3)

        return score_gt_instance, scores_gt_point, index_valid, i_acc_gt, uv_acc_gt

        # score_gt = torch.zeros_like(bbox_indices, dtype=torch.float32)
        # for i_bbox, bbox_index in enumerate(bbox_indices):
        #     i_point_indices = ((point_bbox_indices == bbox_index).nonzero(as_tuple=True)[0])
        #     if len(i_point_indices) == 0:
        #         continue
        #     i_u_gt = u_gt[i_point_indices]
        #     i_v_gt = v_gt[i_point_indices]
        #     i_i_gt = i_gt[i_point_indices]
        #     i_u_est = u_est[i_point_indices]
        #     i_v_est = v_est[i_point_indices]
        #     i_i_est = i_est[i_point_indices]
        #     cVerts, cVertsGT = self.findAllClosestVerts(i_u_gt,i_v_gt,i_i_gt,i_u_est,i_v_est,i_i_est)
        #     dist,index_valid = self.getDistances(cVertsGT, cVerts)
        #     Current_Mean_Distances = torch.gather(self.mean_dist, 0, self.Part_ids[cVertsGT[index_valid].long() - 1])
        #     i_scores_gt = torch.exp(
        #                     -(dist ** 2) / (2 * (Current_Mean_Distances ** 2))
        #                 )
        #     # i_scores_gt = dist / Current_Mean_Distances
        #     # i_scores_gt[dist == 3.0] = 273
        #     # print(np.array(i_scores_gt.cpu()).tolist())
        #     score_gt[i_bbox] = torch.mean(i_scores_gt)
        # return score_gt
    
    def findAllClosestVerts(self, U_gt, V_gt, I_gt, U_points, V_points, Index_points):

        ClosestVerts = torch.ones(Index_points.shape).cuda() * -1
        ClosestVertsGT = torch.ones(I_gt.shape).cuda() * -1
        for i in range(24):
            #
            Current_Part_UVs = torch.tensor(self.Part_UVs[i], dtype=torch.float64).cuda()
            Current_Part_ClosestVertInds = torch.tensor(self.Part_ClosestVertInds[i], dtype=torch.float32).cuda()
            if (i + 1) in Index_points:
                UVs = torch.stack(( U_points[Index_points == (i + 1)], V_points[Index_points == (i + 1)]), dim=1)
                if len(UVs.shape) == 1:
                    UVs = UVs.unsqueeze(axis=1)
                # Current_Part_UVs = torch.tensor(self.Part_UVs[i], dtype=torch.float64).cuda()
                # Current_Part_ClosestVertInds = torch.tensor(self.Part_ClosestVertInds[i], dtype=torch.float32).cuda()
                D = torch.cdist(Current_Part_UVs.transpose(1,0), UVs.double())
                ClosestVerts[Index_points == (i + 1)] = Current_Part_ClosestVertInds[
                        torch.argmin(D.squeeze(1).float(), axis=0)]
            if (i + 1) in I_gt:
                UVs = torch.stack((U_gt[I_gt == (i + 1)], V_gt[I_gt == (i + 1)]), dim=1)
                if len(UVs.shape) == 1:
                    UVs = UVs.unsqueeze(axis=1)
                # Current_Part_UVs = torch.tensor(self.Part_UVs[i], dtype=torch.float64).cuda()
                # Current_Part_ClosestVertInds = torch.tensor(self.Part_ClosestVertInds[i], dtype=torch.float32).cuda()
                D = torch.cdist(Current_Part_UVs.transpose(1,0), UVs.double())
                ClosestVertsGT[I_gt == (i + 1)] = Current_Part_ClosestVertInds[torch.argmin(D.squeeze(1).float(), axis=0)]
        #
        # ClosestVertsGT = torch.ones(I_gt.shape).cuda() * -1
        # for i in range(24):
        #     if (i + 1) in I_gt:
        #         UVs = torch.stack((U_gt[I_gt == (i + 1)], V_gt[I_gt == (i + 1)]), dim=1)
        #         if len(UVs.shape) == 1:
        #             UVs = UVs.unsqueeze(axis=1)
        #         Current_Part_UVs = torch.tensor(self.Part_UVs[i], dtype=torch.float64).cuda()
        #         Current_Part_ClosestVertInds = torch.tensor(self.Part_ClosestVertInds[i], dtype=torch.float32).cuda()
        #         D = torch.cdist(Current_Part_UVs.transpose(1,0), UVs.double())
        #         ClosestVertsGT[I_gt == (i + 1)] = Current_Part_ClosestVertInds[torch.argmin(D.squeeze(1).float(), axis=0)]

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
        # print("ClosestVertsTransformed", np.array(cVerts.detach().cpu()).tolist())
        # print("ClosestVertsGTTransformed", np.array(cVertsGT.detach().cpu()).tolist())

        index_cVertsGT = (cVertsGT > 0).nonzero().flatten().detach()

        # index_cVerts = set(np.array((cVerts > 0).nonzero().flatten().detach().cpu()))
        # index_valid = list(set(np.array(index_cVertsGT.cpu())) & index_cVerts)
        # cVerts_filter = cVerts[index_valid]-1
        # cVertsGT_filter = cVertsGT[index_valid]-1

        cVerts_filter = cVerts[index_cVertsGT]
        cVertsGT = cVertsGT[index_cVertsGT]

        oulter = (cVerts_filter > 0).long().cuda()

        cVerts_filter = (cVerts_filter - 1)*oulter
        cVertsGT_filter = (cVertsGT - 1)*oulter
        
        # index_cVerts_filter = (cVerts_filter > 0).nonzero().flatten().detach()
        # cVerts_filter = cVerts[index_cVerts_filter]-1
        # cVertsGT_filter = cVertsGT[index_cVerts_filter]-1

        dists = torch.zeros(len(cVerts_filter), dtype=torch.float32).cuda()

        cVerts_max = torch.max(cVertsGT_filter, cVerts_filter)
        cVerts_min = torch.min(cVertsGT_filter, cVerts_filter)
        dist_matrix = torch.true_divide(cVerts_max*(cVerts_max-1), 2) + cVerts_min
        dists[cVerts_filter != cVertsGT_filter] = self.Pdist_matrix[dist_matrix[cVerts_filter != cVertsGT_filter].long()].squeeze(1)
        dists = dists*oulter - (oulter-1.)*3
        return dists,index_cVertsGT
        # return torch.from_numpy(np.atleast_1d(np.array(dists.cpu()).squeeze())).float().cuda()

    def produce_densepose_losses_for_none_proposal(self, feature: Any) -> LossDict:
        losses = {
            "loss_densepose_score": feature.sum() * 0,
            }
        return {**losses}