import torch
import torch.nn.functional as F
from torch import nn


class InterpolateSparse2d(nn.Module):
    """
    Interpolate 3D tensor given N sparse 2D positions
    Input
      x: list([C, H, W]) feature tensors at different scales (e.g. from a U-Net), ONLY the last one is used
      pos: [N, 2] tensor of positions

    Returns
      [N, C] sampled features at 2d positions
    """

    def __init__(self, mode="bicubic"):
        super().__init__()
        self.mode = mode
        self.name = "InterpolateSparse2d"

    def get_full_descriptor_map(self, x):
        raise NotImplementedError(
            "get_full_descriptor_map is not implemented for InterpolateSparse2d"
        )

    def forward(self, x, pos):
        x = x[-1]  # only use the last layer

        # check if grid is float32
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        x = F.grid_sample(x, pos[:, None], mode=self.mode, align_corners=True)
        return x.permute(0, 2, 3, 1).squeeze(-2)
