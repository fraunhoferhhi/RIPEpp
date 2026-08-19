import torch
import torch.nn.functional as F
from torch import nn


class DirectSample(nn.Module):
    """
    Directly samples from the given feature map at the given positions using nearest neighbor.
    Input
      x: [B, C, H, W] feature tensor
      pos: [B, N, 2] tensor of positions (normalized to -1 to 1)

    Returns
      [B, N, C] sampled features at 2d positions
    """

    def __init__(self):
        super().__init__()
        self.name = "direct_sample"

    def extract_values(self, x, pos):
        """Extract values from tensor x at the positions given by pos.

        Args:
        - x (Tensor): Tensor of size (B, C, H, W).
        - pos (Tensor): Tensor of size (B, N, 2) containing the x, y positions (normalized to -1 to 1).

        Returns:
        - values (Tensor): Tensor of size (B, N, C) with the values from f at the positions given by p.
        """
        B = x.shape[0]

        # Allow callers to pass unbatched positions for B=1.
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
        if pos.size(0) != B:
            raise ValueError(
                f"pos batch dim ({pos.size(0)}) must match x batch dim ({B})"
            )

        # grid_sample expects grid of shape (B, H_out, W_out, 2), use (B, 1, N, 2) for point sampling
        x = F.grid_sample(x, pos[:, None], mode="nearest", align_corners=False)
        return x.permute(0, 2, 3, 1).squeeze(1)

    @torch.no_grad()
    def get_full_descriptor_map(self, x):
        """Returns a copy of the full descriptor map from the feature tensor x.
        For debugging/visualization purposes.

        Args:
        - x (Tensor): Tensor of size (B, C, H, W).

        Returns:
        - desc_map (Tensor): Tensor of size (B, C, H, W)
        """

        return x.clone()

    def forward(self, x, pos):
        desc = self.extract_values(x, pos)
        return desc
