import concurrent.futures

import torch

from ripepp.utils.utils import get_required_keys


class ConcurrentMatchingPipeline:
    required_keys = ["combined_desc", "kpts", "mask_matching"]

    def __init__(self, matcher, robust_estimator, min_num_matches=8, max_workers=12):
        self.matcher = matcher
        self.robust_estimator = robust_estimator
        self.min_num_matches = min_num_matches

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    @torch.no_grad()
    def __call__(self, inp_1, inp_2, inl_th, label=None):
        pdesc1, kpts1, selected_mask1 = get_required_keys(inp_1, self.required_keys)
        pdesc2, kpts2, selected_mask2 = get_required_keys(inp_2, self.required_keys)

        dev = pdesc1.device
        B = pdesc1.shape[0]

        batch_rel_idx_matches = [None] * B
        batch_idx_matches = [None] * B
        future_results = [None] * B

        for b in range(B):
            if selected_mask1[b].sum() < 16 or selected_mask2[b].sum() < 16:
                continue

            dists, idx_matches = self.matcher(pdesc1[b][selected_mask1[b]], pdesc2[b][selected_mask2[b]])

            batch_rel_idx_matches[b] = idx_matches.clone()

            # calculate ABSOLUTE indexes
            idx_matches[:, 0] = torch.nonzero(selected_mask1[b], as_tuple=False)[idx_matches[:, 0]].squeeze()
            idx_matches[:, 1] = torch.nonzero(selected_mask2[b], as_tuple=False)[idx_matches[:, 1]].squeeze()

            batch_idx_matches[b] = idx_matches

            # if not enough matches
            if idx_matches.shape[0] < self.min_num_matches:
                ransac_inliers = torch.zeros((idx_matches.shape[0]), device=dev).bool()
                future_results[b] = (None, ransac_inliers)
                continue

            # use label information to exclude negative pairs from geometric filtering process -> enforces more descriminative descriptors
            if label is not None and label[b] == 0:
                ransac_inliers = torch.ones((idx_matches.shape[0]), device=dev).bool()
                future_results[b] = (None, ransac_inliers)
                continue

            mkpts1 = kpts1[b][idx_matches[:, 0]]
            mkpts2 = kpts2[b][idx_matches[:, 1]]

            future_results[b] = self.executor.submit(self.robust_estimator, mkpts1, mkpts2, inl_th)

        batch_ransac_inliers = [None] * B
        batch_Fm = [None] * B

        for b in range(B):
            future_result = future_results[b]
            if future_result is None:
                ransac_inliers = None
                Fm = None
            elif isinstance(future_result, tuple):
                Fm, ransac_inliers = future_result
            else:
                Fm, ransac_inliers = future_result.result()

                # if no inliers
                if ransac_inliers.sum() == 0:
                    ransac_inliers = ransac_inliers.squeeze(
                        -1
                    )  # kornia.geometry.ransac.RANSAC returns (N, 1) tensor if no inliers and (N,) tensor if inliers
                    Fm = None

            batch_ransac_inliers[b] = ransac_inliers
            batch_Fm[b] = Fm

        return {
            "rel_idx_matches": batch_rel_idx_matches,
            "idx_matches": batch_idx_matches,
            "ransac_inliers": batch_ransac_inliers,
            "Fm": batch_Fm,
        }
