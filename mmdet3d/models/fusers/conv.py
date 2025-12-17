from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["ConvFuser"]


@FUSERS.register_module()
class ConvFuser(nn.Sequential):
    # CHANGE 1: Added 'dropout' argument with a default of 0.5 (50% drop)
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.5) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            # CHANGE 2: Add Dropout2d as the first layer.
            # This randomly zeros out 50% of the fused feature map during training.
            nn.Dropout2d(dropout),
            
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))

