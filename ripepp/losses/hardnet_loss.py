import torch
import torch.nn as nn
import torch.nn.functional as F

from ripepp.utils.utils import get_batch, get_required_keys


def second_nearest_neighbor(desc1, desc2):
    if desc2.shape[0] < 2:  # We cannot perform snn check, so output empty matches
        raise ValueError("desc2 should have at least 2 descriptors")

    dist = torch.cdist(desc1, desc2, p=2)

    vals, idxs = torch.topk(dist, 2, dim=1, largest=False)
    idxs_in_2 = idxs[:, 1]
    idxs_in_1 = torch.arange(0, idxs_in_2.size(0), device=dist.device)

    matches_idxs = torch.cat([idxs_in_1.view(-1, 1), idxs_in_2.view(-1, 1)], 1)

    return vals[:, 1].view(-1, 1), matches_idxs


def hardnet_loss(desc1, desc2, matches, inliers, label, pos_margin=1.0, neg_margin=1.0):
    """
    desc1: [N1, d] descriptors from image 1
    desc2: [N2, d] descriptors from image 2
    matches: [M, 2] putative matches between desc1 and desc2
    inliers: [M] binary mask of inlier matches
    label: bool, if True compute positive loss, else negative loss
    pos_margin: float, margin for positive loss
    neg_margin: float, margin for negative loss
    """

    if inliers.sum() < 8:  # if there are too few inliers, calculate loss on all matches
        inliers = torch.ones_like(inliers)

    matched_inliers_descs1 = desc1[matches[:, 0][inliers]]
    matched_inliers_descs2 = desc2[matches[:, 1][inliers]]

    if label:
        snn_match_dists_1, idx1 = second_nearest_neighbor(matched_inliers_descs1, desc2)
        snn_match_dists_2, idx2 = second_nearest_neighbor(matched_inliers_descs2, desc1)

        dists = torch.hstack((snn_match_dists_1, snn_match_dists_2))
        min_dists_idx = torch.min(dists, dim=1).indices.unsqueeze(1)

        dists_hard = torch.gather(dists, 1, min_dists_idx).squeeze(-1)
        dists_pos = F.pairwise_distance(matched_inliers_descs1, matched_inliers_descs2)

        hard_loss = torch.clamp(pos_margin + dists_pos - dists_hard, min=0.0)

        hard_loss = hard_loss.mean()

    else:
        dists = F.pairwise_distance(matched_inliers_descs1, matched_inliers_descs2)
        hard_loss = torch.clamp(neg_margin - dists, min=0.0)

        hard_loss = hard_loss.mean()

    return hard_loss


class HardNetLoss(nn.Module):
    required_keys_image = ["desc"]
    required_keys_matching = ["idx_matches", "ransac_inliers"]

    def __init__(self, pos_margin=1.0, neg_margin=1.0):
        super().__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

    def forward(self, inp_image1, inp_image2, inp_match_1_2, label, b):
        desc1 = get_batch(b, get_required_keys(inp_image1, self.required_keys_image))
        desc2 = get_batch(b, get_required_keys(inp_image2, self.required_keys_image))
        matches, inliers = get_batch(b, *get_required_keys(inp_match_1_2, self.required_keys_matching))
        label_b = get_batch(b, label)

        return hardnet_loss(desc1, desc2, matches, inliers, label_b, self.pos_margin, self.neg_margin)
