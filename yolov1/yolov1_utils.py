"""Utilities for YOLOv1"""
from typing import List

import torch


def intersection_over_union(
    boxes_preds: torch.Tensor,
    boxes_labels: torch.Tensor,
) -> torch.Tensor:
    """Intersection over union algorithm implementation

    Args:
        boxes_preds (torch.Tensor): Prediction boxes
        boxes_labels (torch.Tensor): Target boxes

    Returns:
        torch.Tensor: IOU for provided boxes
    """
    box1_x1 = boxes_preds[..., 0:1]
    box1_y1 = boxes_preds[..., 1:2]
    box1_x2 = boxes_preds[..., 2:3]
    box1_y2 = boxes_preds[..., 3:4]

    box2_x1 = boxes_labels[..., 0:1]
    box2_y1 = boxes_labels[..., 1:2]
    box2_x2 = boxes_labels[..., 2:3]
    box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) + (y2 - y1).clamp(0)

    box1_area = torch.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = torch.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_maximum_suppression(
    bboxes: List[List[float]],
    iou_threshold: float,
    threshold: float,
) -> list:
    """Suppressing low-prediction-score bound boxes

    Args:
        bboxes (List[List[float]]): List of bounding boxes
        iou_threshold (float): Threshould for intersection over union score
        threshold (float): Threshold for prediction score

    Returns:
        list: List of filtered bounded boxes
    """
    # bbox = [class, pred, x, y, w, h]
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]), torch.tensor(box[2:])
            )
            < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms
