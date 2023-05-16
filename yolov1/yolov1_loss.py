"""Loss function implementation for YOLOv1"""
import torch
from torch import nn

from yolov1_utils import intersection_over_union


class YOLOv1Loss(nn.Module):
    """YOLOv1 loss implementation"""

    lambda_noobj = 0.5
    lambda_coord = 5

    def __init__(self, split_size: int, num_boxes: int, num_classes: int) -> None:
        super().__init__()
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass through the loss function

        Args:
            predictions (torch.Tensor): Prediction labels
            targets (torch.Tensor): Target labels

        Returns:
            torch.Tensor: MSE loss between target and prediction labels
        """
        predictions = predictions.reshape(
            -1,
            self.split_size,
            self.split_size,
            self.num_classes + self.num_boxes * 5,
        )

        iou_b1 = intersection_over_union(predictions[..., 21:25], targets[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], targets[..., 21:25])

        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        _, best_box = torch.max(ious, dim=0)
        exists_box = targets[..., 20].unsqueeze(3)  # in paper this is Iobj_i

        box_predictions = exists_box * (
            best_box * predictions[..., 26:30]
            + (1 - best_box) * predictions[..., 21:25]
        )
        box_targets = exists_box * targets[..., 21:25]

        # We added 1e-6 to avoid division by zero
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4]) + 1e-6
        )

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        pred_box = (
            best_box * predictions[..., 25:26]
            + (1 - best_box) * predictions[..., 20:21]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * targets[..., 20:21]),
        )

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * targets[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * targets[..., 20:21], start_dim=1),
        )

        class_loss = self.mse(
            self.flatten((exists_box * predictions[..., :20]), end_dim=-2),
            self.flatten((exists_box * targets[..., :20]), end_dim=-2),
        )

        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss
