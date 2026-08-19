import torch
import torch.nn as nn
import torch.nn.functional as F

from ripepp.utils.utils import get_batch, get_required_keys


class InfoNCE_Loss(nn.Module):
    required_keys_image = ["desc"]
    required_keys_matching = ["idx_matches", "ransac_inliers"]

    def __init__(self, tau=0.05, symmetric=False):
        super().__init__()
        self.tau = tau
        self.symmetric = symmetric  # whether to compute loss in both directions

    def infonce_loss(self, desc_1, desc_2, matches_1_2, inliers_1_2):
        """
        desc_1: [N1, d] descriptors from anchor image, must be L2-normalized
        desc_2: [N2, d] descriptors from positive image, must be L2-normalized
        matches_1_2: [M, 2] putative matches between desc_1 and desc_2
        inliers_1_2: [M] binary mask of inlier matches
        tau: float, temperature parameter
        """

        # check if descriptors are normalized
        if not torch.allclose(
            desc_1.norm(dim=1),
            torch.ones(desc_1.size(0), device=desc_1.device),
            atol=1e-3,
        ):
            raise ValueError("Descriptors desc_1 are not L2-normalized")
        if not torch.allclose(
            desc_2.norm(dim=1),
            torch.ones(desc_2.size(0), device=desc_2.device),
            atol=1e-3,
        ):
            raise ValueError("Descriptors desc_2 are not L2-normalized")

        sim = desc_1 @ desc_2.T / self.tau  # [N1, N2]

        log_probs = F.log_softmax(sim, dim=1)  # [N1, N2]

        match_inliers = matches_1_2[inliers_1_2]
        L = -log_probs[match_inliers[:, 0], match_inliers[:, 1]].sum()

        num_inliers = inliers_1_2.sum()
        if num_inliers == 0:
            return torch.tensor(0.0, device=desc_1.device)
        else:
            return L / num_inliers

    def forward(self, inp_image1, inp_image2, inp_match_1_2, b, **kwargs):
        desc_1 = get_batch(b, get_required_keys(inp_image1, self.required_keys_image))
        desc_2 = get_batch(b, get_required_keys(inp_image2, self.required_keys_image))
        matches_1_2, inliers_1_2 = get_batch(b, *get_required_keys(inp_match_1_2, self.required_keys_matching))

        if not self.symmetric:
            return self.infonce_loss(desc_1, desc_2, matches_1_2, inliers_1_2)
        else:
            L1 = self.infonce_loss(desc_1, desc_2, matches_1_2, inliers_1_2)
            L2 = self.infonce_loss(desc_2, desc_1, matches_1_2.flip(1), inliers_1_2)
            return 0.5 * (L1 + L2)
