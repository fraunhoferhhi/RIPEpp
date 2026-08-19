import numpy as np
import torch
import torch.nn as nn

from ripepp.utils.utils import get_batch, get_required_keys


def ensure_3d(tensor):
    return tensor.unsqueeze(0) if tensor.dim() == 2 else tensor


def weighted_softmax_stable(logits: torch.Tensor, weights: torch.Tensor, dim: int = -1):
    """
    Numerically stable weighted softmax:
    σ_w(z)_i = (w_i * exp(z_i)) / Σ_j (w_j * exp(z_j))

    Args:
        logits: Tensor of shape (..., K)
        weights: Tensor of shape (..., K) or (K,) broadcastable to logits
        dim: Dimension over which to apply softmax

    Returns:
        probs: Weighted softmax probabilities, same shape as logits
    """
    # Ensure weights are positive
    weights = torch.clamp(weights, min=1e-12)

    # Subtract max for numerical stability
    max_logits, _ = torch.max(logits, dim=dim, keepdim=True)
    shifted_logits = logits - max_logits

    exp_logits = torch.exp(shifted_logits)
    weighted_exp = weights * exp_logits
    probs = weighted_exp / torch.sum(weighted_exp, dim=dim, keepdim=True)
    return probs


# Fm can be np.array or torch.Tensor
def sampson_epipolar_distance_matrix(
    pts1: torch.Tensor, pts2: torch.Tensor, Fm, squared: bool = True, eps: float = 1e-8
) -> torch.Tensor:
    """Return Sampson distance matrix between all pairs of points in pts1 and pts2.

    Args:
        pts1: Points from the left image with shape (N1, 2|3).
        pts2: Points from the right image with shape (N2, 2|3).
        Fm: Fundamental matrix with shape (3, 3).
        squared: If True (default), the squared distance is returned.
        eps: Small constant for safe sqrt.

    Returns:
        A distance matrix of shape (N1, N2).
    """
    if isinstance(Fm, np.ndarray):
        Fm = torch.tensor(Fm, device=pts1.device, dtype=torch.float32)

    if pts1.shape[-1] == 2:
        ones = torch.ones_like(pts1[..., :1])
        pts1 = torch.cat([pts1, ones], dim=-1)

    if pts2.shape[-1] == 2:
        ones = torch.ones_like(pts2[..., :1])
        pts2 = torch.cat([pts2, ones], dim=-1)

    # Expand dimensions: (N1, 1, 3) and (1, N2, 3)
    pts1_exp = pts1[:, None, :]  # (N1, 1, 3)
    pts2_exp = pts2[None, :, :]  # (1, N2, 3)

    # Compute lines
    F_t = Fm.T
    line1_in_2 = pts1_exp @ F_t  # (N1, 1, 3)
    line2_in_1 = pts2_exp @ Fm  # (1, N2, 3)

    # Numerator: (x'^T F x)^2
    numerator = (pts2_exp @ Fm @ pts1_exp.transpose(-1, -2)).squeeze(-1).squeeze(-1).pow(2)

    # Denominator
    denom1 = line1_in_2[..., :2].norm(dim=-1).pow(2)  # (N1, 1)
    denom2 = line2_in_1[..., :2].norm(dim=-1).pow(2)  # (1, N2)
    denominator = denom1 + denom2

    out = numerator / denominator
    if squared:
        return out
    return (out + eps).sqrt()


class WeightedInfoNCE_Loss(nn.Module):
    required_keys_image = ["desc", "kpts"]
    required_keys_matching = ["idx_matches", "ransac_inliers", "Fm"]

    def __init__(self, tau=0.05, alpha=0.0001, symmetric=False):
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        self.symmetric = symmetric  # whether to compute loss in both directions

    def weighted_infonce_loss(
        self,
        desc_1,
        desc_2,
        matches_1_2,
        inliers_1_2,
        kpts_1=None,
        kpts_2=None,
        F_1_2=None,
    ):
        """
        anc_desc: [N1, d] descriptors from anchor image, L2-normalized
        pos_desc: [N2, d] descriptors from positive image, L2-normalized
        anc_pos_matches: [M, 2] putative matches between anc_desc and pos_desc
        anc_pos_inliers: [M] binary mask of inlier matches
        tau: float, temperature parameter
        anc_kpts: [N1, 2] keypoints from anchor image
        pos_kpts: [N2, 2] keypoints from positive image
        neg_kpts: [N3, 2] keypoints from negative image
        anc_pos_F: [3, 3] fundamental matrix between anchor and positive image
        anc_neg_F: [3, 3] fundamental matrix between anchor and negative image
        """

        if not isinstance(F_1_2, torch.Tensor):
            F_1_2 = torch.tensor(F_1_2, device=desc_1.device, dtype=torch.float32)

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

        # kornia expects (B, N, 2) keypoints
        # kpts_1 = ensure_3d(kpts_1)
        # kpts_2 = ensure_3d(kpts_2)
        # F_1_2 = ensure_3d(F_1_2)

        sim = desc_1 @ desc_2.T / self.tau  # [N1, N2]

        # weights = sampson_epipolar_distance(kpts_1, kpts_2, F_1_2)  # [N1, N2]
        weights = sampson_epipolar_distance_matrix(kpts_1, kpts_2, F_1_2)  # [N1, N2]
        weights = torch.exp(-self.alpha * weights)  # [N1, N2]

        log_probs = torch.log(weighted_softmax_stable(sim, weights, dim=1) + 1e-12)  # [N1, N2+N3]

        match_inliers = matches_1_2[inliers_1_2]
        L = -log_probs[match_inliers[:, 0], match_inliers[:, 1]].sum()

        inliers_1_2_sum = inliers_1_2.sum()
        if inliers_1_2_sum == 0:
            return torch.tensor(0.0, device=desc_1.device)
        else:
            return L / inliers_1_2_sum

    def forward(
        self,
        inp_image1,
        inp_image2,
        inp_match_1_2,
        b,
        **kwargs,
    ):
        desc_1, kpts_1 = get_batch(b, *get_required_keys(inp_image1, self.required_keys_image))
        desc_2, kpts_2 = get_batch(b, *get_required_keys(inp_image2, self.required_keys_image))
        matches_1_2, inliers_1_2, F_1_2 = get_batch(b, *get_required_keys(inp_match_1_2, self.required_keys_matching))

        if not self.symmetric:
            return self.weighted_infonce_loss(
                desc_1,
                desc_2,
                matches_1_2,
                inliers_1_2,
                kpts_1,
                kpts_2,
                F_1_2,
            )
        else:
            L1 = self.weighted_infonce_loss(
                desc_1,
                desc_2,
                matches_1_2,
                inliers_1_2,
                kpts_1,
                kpts_2,
                F_1_2,
            )

            L2 = self.weighted_infonce_loss(
                desc_2,
                desc_1,
                matches_1_2.flip(1),
                inliers_1_2,
                kpts_2,
                kpts_1,
                F_1_2.T,
            )
            return 0.5 * (L1 + L2)
