from ripepp.utils.utils import get_batch, get_required_keys, get_rewards


def rl_loss(
    inp_matching,
    inp_image1,
    inp_image2,
    batch_id,
    label,
    cfg,
    fp_penalty,
    inlier_reward_value,
    outlier_penalty_value,
    kernel_fn=None,
):
    required_keys_matching = ["rel_idx_matches", "idx_matches", "ransac_inliers", "Fm"]
    required_keys_images = ["mask_matching", "kpts", "logprobs", "mask_padding"]

    # xor for use_distance_based_rewards and use_normalized_rewards
    assert not (cfg.use_distance_based_rewards and cfg.use_normalized_rewards), (
        "Cannot use both distance-based rewards and normalized rewards at the same time."
    )

    rel_idx_matches, abs_idx_matches, ransac_inliers, Fm = get_batch(
        batch_id, *get_required_keys(inp_matching, required_keys_matching)
    )
    mask_selection_for_matching_1, kpts1, logprobs1, mask_padding_grid_1 = get_batch(
        batch_id, *get_required_keys(inp_image1, required_keys_images)
    )
    mask_selection_for_matching_2, kpts2, logprobs2, mask_padding_grid_2 = get_batch(
        batch_id, *get_required_keys(inp_image2, required_keys_images)
    )
    label_b = get_batch(batch_id, label)

    # ignore if less than 16 keypoints have been detected
    if rel_idx_matches is None:
        return None, 0

    # Fm = batch_Fm[b]

    # every keypoint with every other keypoint, but WITHOUT keypoint in the padding area
    dense_logprobs = logprobs1[mask_padding_grid_1].view(-1, 1) + logprobs2[mask_padding_grid_2].view(1, -1)

    H, W = inp_image1["heatmap"].shape[-2:]

    dense_rewards = get_rewards(
        kpts1,
        kpts2,
        mask_selection_for_matching_1,
        mask_selection_for_matching_2,
        mask_padding_grid_1,
        mask_padding_grid_2,
        rel_idx_matches,
        abs_idx_matches,
        ransac_inliers,
        label_b,
        H,
        W,
        fp_penalty,
        cfg.train_mode,
        inlier_reward_value,
        outlier_penalty_value,
        cfg.use_distance_based_rewards,
        cfg.use_normalized_rewards,
        Fm,
        kernel_fn,
    )

    current_loss_policy = (dense_rewards * dense_logprobs).view(-1)

    return current_loss_policy, dense_rewards.sum()
