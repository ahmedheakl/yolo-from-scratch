"""Implementation of YOLOv1"""
from typing import List, Any

import torch
from torch import nn


architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    """Convolutional block"""

    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()

        # bias=False because batchnorm is used
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass of the block"""
        img = self.conv(img)
        img = self.batchnorm(img)
        return self.leakyrelu(img)


class YOLOv1(nn.Module):
    """YOLOv1 implementation"""

    def __init__(self, in_channels: int = 3, **kwargs) -> None:
        super().__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcn = self._create_fcn_layers(**kwargs)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network

        Args:
            img (torch.Tensor): Input image

        Returns:
            torch.Tensor: Output of the network
        """
        img = self.darknet(img)
        img = self.fcn(torch.flatten(img, start_dim=1))
        return img

    def _create_conv_layers(self, architecture: List[Any]) -> nn.Module:
        layers = []
        in_channels = self.in_channels

        for architect in architecture:
            if isinstance(architect, tuple):
                layers += [
                    CNNBlock(
                        in_channels,
                        architect[1],
                        kernel_size=architect[0],
                        stride=architect[2],
                        padding=architect[3],
                    ),
                ]
                in_channels = architect[1]
            elif isinstance(architect, str):
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
            elif isinstance(architect, list):
                conv1 = architect[0]
                conv2 = architect[1]
                num_repeats = architect[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        ),
                    ]

                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        ),
                    ]

                    in_channels = conv2[1]
        return nn.Sequential(*layers)

    def _create_fcn_layers(
        self, split_size: int, num_boxes: int, num_classes: int
    ) -> nn.Module:
        out_dim = split_size * split_size * (num_boxes * 5 + num_classes)
        return nn.Sequential(
            nn.Flatten(),
            # In the original paper, they used 4096 neurons
            nn.Linear(1024 * split_size * split_size, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, out_dim),
        )


def main() -> None:
    """Main function"""
    split_size = 7
    num_boxes = 2
    num_classes = 20

    model = YOLOv1(split_size=split_size, num_boxes=num_boxes, num_classes=num_classes)
    imgs = torch.randn((2, 3, 448, 448))
    print(model(imgs).shape)


if __name__ == "__main__":
    main()
