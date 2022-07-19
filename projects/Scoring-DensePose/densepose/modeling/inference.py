# Copyright (c) Facebook, Inc. and its affiliates.
from dataclasses import fields
from typing import Any, List
import torch

from detectron2.structures import Instances


def densepose_inference(densepose_predictor_output: Any, detections: List[Instances], densepose_scoring_predictor_outputs=None):
    """
    Splits DensePose predictor outputs into chunks, each chunk corresponds to
    detections on one image. Predictor output chunks are stored in `pred_densepose`
    attribute of the corresponding `Instances` object.

    Args:
        densepose_predictor_output: a dataclass instance (can be of different types,
            depending on predictor used for inference). Each field can be `None`
            (if the corresponding output was not inferred) or a tensor of size
            [N, ...], where N = N_1 + N_2 + .. + N_k is a total number of
            detections on all images, N_1 is the number of detections on image 1,
            N_2 is the number of detections on image 2, etc.
        detections: a list of objects of type `Instance`, k-th object corresponds
            to detections on k-th image.
    """
    k = 0
    for detection_i in detections:
        if densepose_predictor_output is None:
            # don't add `pred_densepose` attribute
            continue
        n_i = len(detection_i)
        PredictorOutput = type(densepose_predictor_output)
        output_i_dict = {}
        
        # we assume here that `densepose_predictor_output` is a dataclass object
        for field in fields(densepose_predictor_output):
            field_value = getattr(densepose_predictor_output, field.name)
            # slice tensors
            if isinstance(field_value, torch.Tensor):
                output_i_dict[field.name] = field_value[k : k + n_i]
            # leave others as is
            else:
                output_i_dict[field.name] = field_value
        detection_i.pred_densepose = PredictorOutput(**output_i_dict)

        # add pred_densepose_score attribute
        if densepose_scoring_predictor_outputs is not None:
            ScoreOutput = type(densepose_scoring_predictor_outputs)
            output_i_score_dict = {}
            for field in fields(densepose_scoring_predictor_outputs):
                field_value = getattr(densepose_scoring_predictor_outputs, field.name)
                if isinstance(field_value, torch.Tensor):
                    output_i_score_dict[field.name] = field_value[k : k + n_i]
                else:
                    output_i_score_dict[field.name] = field_value
            # print(output_i_score_dict["densepose_score"].squeeze(1))
            # detection_i.densepose_scores = torch.sigmoid(output_i_score_dict["densepose_score"].squeeze(1))
            # detection_i.densepose_scores = torch.sqrt(detection_i.scores * torch.sigmoid(output_i_score_dict["densepose_score"]).squeeze())
            # detection_i.densepose_scores = detection_i.scores*0.5 + torch.sigmoid(output_i_score_dict["densepose_score"]).squeeze()*0.5
            detection_i.densepose_scores = output_i_score_dict["densepose_score"]
            detection_i.mask_scores = output_i_score_dict["mask_score"]
            detection_i.i_scores = output_i_score_dict["i_score"]
            detection_i.uv_scores = output_i_score_dict["uv_score"]
            # detection_i.densepose_scores = torch.argmax(output_i_score_dict["densepose_score"], dim=1)*0.1*0.1 + detection_i.scores*0.9
        k += n_i
