import torch
import torch.nn.functional as F
from torch import nn


class HyperColumnFeatures(nn.Module):
    """
    Interpolate 3D tensor given N sparse 2D positions
    Input
      x: list([C, H, W]) list of feature tensors at different scales (e.g. from a U-Net) -> extract hypercolumn features
      pos: [N, 2] tensor of positions

    Returns
      [N, C] sampled features at 2d positions
    """

    def __init__(self, mode="bilinear"):
        super().__init__()
        self.mode = mode
        self.name = "HyperColumnFeatures"

    def extract_values_at_poses(self, x, pos):
        """Extract values from tensor x at the positions given by pos.

        Args:
        - x (Tensor): Tensor of size (B, C, H, W).
        - pos (Tensor): Tensor of size (B, N, 2) containing the x, y positions (normalized to -1 to 1).

        Returns:
        - values (Tensor): Tensor of size (B, N, C) with the values from f at the positions given by p.
        """

        # check if grid is float32
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        x = F.grid_sample(x, pos[:, None], mode=self.mode, align_corners=False)
        return x.permute(0, 2, 3, 1).squeeze(1)

    @torch.no_grad()
    def get_full_descriptor_map(self, x):
        """Returns a copy of the full descriptor map from the feature tensor x.
        For debugging/visualization purposes.

        Upsamples all feature maps to the size of the largest one and concatenates them.

        Args:
        - x (List[Tensor]): List of tensors different scales. First one is the largest.

        Returns:
        - desc_map (Tensor): Tensor of size (B, C, H, W)
        """

        B, C, H, W = x[0].size()
        desc_map = []

        for layer in x:
            layer_upsampled = F.interpolate(
                layer, size=(H, W), mode="bilinear", align_corners=True
            )
            desc_map.append(layer_upsampled)

        desc_map = torch.cat(desc_map, dim=1)  # Concatenate along channel dimension

        return desc_map

    def forward(self, x, pos):
        descs = []

        for layer in x:
            desc = self.extract_values_at_poses(layer, pos)
            descs.append(desc)

        return descs
