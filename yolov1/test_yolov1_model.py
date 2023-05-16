"""Testing YOLOv1 model"""
import torch

from yolov1_model import YOLOv1


def test_output_dimension():
    split_size = 7
    num_boxes = 2
    num_classes = 20

    model = YOLOv1(split_size=split_size, num_boxes=num_boxes, num_classes=num_classes)
    imgs = torch.randn((2, 3, 448, 448))

    out = model(imgs)

    assert out.shape == (2, 1470)
