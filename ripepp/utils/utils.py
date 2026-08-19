import io
import random
import subprocess
from pathlib import Path
from typing import List

import cv2
import kornia.feature as KF
import kornia.geometry as KG
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.epipolar import sampson_epipolar_distance
from omegaconf import OmegaConf
from torchvision.utils import make_grid
from torchvision.transforms.functional import resize

from ripepp import utils

log = utils.get_pylogger(__name__)


def unnormalize_coords(x_n, h, w):
    x = torch.stack((w * (x_n[..., 0] + 1) / 2, h * (x_n[..., 1] + 1) / 2), dim=-1)  # [-1+1/h, 1-1/h] -> [0.5, h-0.5]
    return x


def normalize_coords(x, h, w):
    x = torch.stack((2 * (x[..., 0] / w) - 1, 2 * (x[..., 1] / h) - 1), dim=-1)  # [-1+1/h, 1-1/h] -> [0.5, h-0.5]
    return x


def extract_patches_from_inds(x: torch.Tensor, inds: torch.Tensor, patch_size: int):
    B, H, W = x.shape
    B, N = inds.shape
    unfolder = nn.Unfold(kernel_size=patch_size, padding=patch_size // 2, stride=1)
    unfolded_x: torch.Tensor = unfolder(x[:, None])  # B x K_H * K_W x H * W
    patches = torch.gather(
        unfolded_x,
        dim=2,
        index=inds[:, None, :].expand(B, patch_size**2, N),
    )  # B x K_H * K_W x N
    return patches


def gridify(x, window_size):
    """Turn a tensor of BxCxHxW into a tensor of
    BxCx(H//window_size)x(W//window_size)x(window_size**2)

    Params:
        x: Input tensor of shape BxCxHxW
        window_size: Size of the window

    Returns:
        x: Output tensor of shape BxCx(H//window_size)x(W//window_size)x(window_size**2)
    """

    assert x.dim() == 4, "Input tensor x must have 4 dimensions"

    B, C, H, W = x.shape
    x = (
        x.unfold(2, window_size, window_size)
        .unfold(3, window_size, window_size)
        .reshape(B, C, H // window_size, W // window_size, window_size**2)
    )

    return x


def check_all_received_grads(model, params_to_ignore=None):
    params_to_ignore = params_to_ignore or []

    all_updated = True
    for name, param in model.named_parameters():
        if param.grad is None and not any(p in name for p in params_to_ignore):
            log.info(f"Parameter {name} has no gradient")
            all_updated = False

    return all_updated


def to_normed_coords(flow, h1, w1):
    normalized_flow = torch.stack(
        (
            2 * flow[..., 0] / (w1 - 1) - 1,  # Normalize x flow
            2 * flow[..., 1] / (h1 - 1) - 1,  # Normalize y flow
        ),
        axis=-1,
    )
    return normalized_flow


def get_grid(B, H, W, device):
    x1_n = torch.meshgrid(
        *[torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=device) for n in (B, H, W)],
        indexing="ij",
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
    return x1_n


def cv2_matches_from_kornia(match_dists: torch.Tensor, match_idxs: torch.Tensor) -> List[cv2.DMatch]:
    return [cv2.DMatch(idx[0].item(), idx[1].item(), d.item()) for idx, d in zip(match_idxs, match_dists, strict=False)]


def to_pixel_coords(flow, h1, w1):
    flow = torch.stack(
        (
            w1 * (flow[..., 0] + 1) / 2,
            h1 * (flow[..., 1] + 1) / 2,
        ),
        axis=-1,
    )
    return flow


def to_cv_kpts(kpts, scores):
    kp = kpts.cpu().numpy().astype(np.int16)
    s = scores.cpu().numpy()

    cv_kp = [cv2.KeyPoint(kp[i][0], kp[i][1], 6, 0, s[i]) for i in range(len(kp))]

    return cv_kp


@torch.no_grad()
def sample_keypoints(
    scoremap,
    num_samples=8192,
    use_nms=True,
    sample_topk=False,
    return_scoremap=False,
    sharpen=False,
    upsample=False,
    increase_coverage=False,
    remove_borders=False,
):
    device = scoremap.device

    # scoremap = scoremap**2
    log_scoremap = (scoremap + 1e-10).log()
    if upsample:
        log_scoremap = F.interpolate(log_scoremap[:, None], scale_factor=3, mode="bicubic", align_corners=False)[
            :, 0
        ]  # .clamp(min = 0)
        scoremap = log_scoremap.exp()
    B, H, W = scoremap.shape
    if increase_coverage:
        weights = (-(torch.linspace(-2, 2, steps=51, device=device) ** 2)).exp()[None, None]
        # 10000 is just some number for maybe numerical stability, who knows. :), result is invariant anyway
        local_density_x = F.conv2d(
            (scoremap[:, None] + 1e-6) * 10000,
            weights[..., None, :],
            padding=(0, 51 // 2),
        )
        local_density = F.conv2d(local_density_x, weights[..., None], padding=(51 // 2, 0))[:, 0]
        scoremap = scoremap * (local_density + 1e-8) ** (-1 / 2)
    grid = get_grid(B, H, W, device=device).reshape(B, H * W, 2)
    if sharpen:
        laplace_operator = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], device=device) / 4
        scoremap = scoremap[:, None] - 0.5 * F.conv2d(scoremap[:, None], weight=laplace_operator, padding=1)
        scoremap = scoremap[:, 0].clamp(min=0)
    if use_nms:
        scoremap = scoremap * (scoremap == F.max_pool2d(scoremap, (3, 3), stride=1, padding=1))
    if remove_borders:
        frame = torch.zeros_like(scoremap)
        # we hardcode 4px, could do it nicer, but whatever
        frame[..., 4:-4, 4:-4] = 1
        scoremap = scoremap * frame
    if sample_topk:
        inds = torch.topk(scoremap.reshape(B, H * W), k=num_samples).indices
    else:
        inds = torch.multinomial(scoremap.reshape(B, H * W), num_samples=num_samples, replacement=False)
    kps = torch.gather(grid, dim=1, index=inds[..., None].expand(B, num_samples, 2))
    if return_scoremap:
        return kps, torch.gather(scoremap.reshape(B, H * W), dim=1, index=inds)
    return kps


def save_source_code(wd, experiment_folder):
    log.info(f"Saving source code to: {experiment_folder / 'ripe_src.zip'}")

    command = ["zip", "-qr", experiment_folder / "ripe_src.zip", wd / "ripepp"]
    _ = subprocess.run(command, check=True, shell=False)  # nosec


def save_hydra_config_file(cfg, path):
    log.info(f"Saving hydra config to: {path}")

    # dumps to file:
    with open(path, "w") as f:
        OmegaConf.save(cfg, f)


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []

    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().numpy())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation=60, ha="right")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()


def plot_grid(warped, title="Grid Vis", mpl=True):
    # visualize
    g = None
    n = warped[0].shape[0]

    for i in range(0, n, 16):
        if i + 16 <= n:
            for w in warped:
                pad_val = 0.7 if i // 16 % 2 == 1 else 0
                gw = make_grid(
                    w[i : i + 16].detach().clone().cpu(),
                    padding=4,
                    pad_value=pad_val,
                    nrow=16,
                )
                g = gw if g is None else torch.cat((g, gw), 1)

    if mpl:
        fig = plt.figure(figsize=(12, 3), dpi=100)
        plt.imshow(np.clip(g.permute(1, 2, 0).numpy()[..., ::-1], 0, 1))
        return fig


def grab_mpl_fig(fig):
    """Transform current drawn fig into a np array."""
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format="raw", dpi=100)
    io_buf.seek(0)
    img_arr = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
    )
    io_buf.close()
    return img_arr
    # plt.imshow(img_arr) ; plt.show(); input()


def warp_pts(pts, H):
    pts = torch.vstack([pts.t(), torch.ones(1, pts.shape[0], device=pts.device)])
    warped = torch.matmul(H, pts)
    warped = warped / warped[2, ...]
    warped = warped.t()[:, :2]

    return warped


def get_mixed_rewards(kps1, kps2, mnn_matches, inlier_matches, label, step, max_step):
    with torch.no_grad():
        reward = 1.0 if label else -1.0

        weight_mnn = 1.0 - (step / max_step)
        weight_inl = 0.0 + (step / max_step)

        dense_returns = torch.zeros((len(kps1), len(kps2)), device=kps1.device)

        dense_returns[mnn_matches[:, 0], mnn_matches[:, 1]] = reward * weight_mnn
        dense_returns[inlier_matches[:, 0], inlier_matches[:, 1]] = reward * weight_inl

    return dense_returns, dense_returns.sum()


def get_rewards(
    kps1,
    kps2,
    selected_mask1,
    selected_mask2,
    padding_mask1,
    padding_mask2,
    rel_idx_matches,
    abs_idx_matches,
    ransac_inliers,
    label,
    H,
    W,
    penalty=0.0,
    training_mode="positive_and_negative",
    inlier_reward=1.0,
    outlier_penalty=-0.1,
    distance_based_reward=False,
    use_normalized_rewards=False,
    Fm=None,
    kernel_fn=None,
):
    with torch.no_grad():
        if training_mode == "positive_and_negative":
            reward = 1.0 if label else -1.0

            dense_returns = torch.zeros((len(kps1), len(kps2)), device=kps1.device)

            if use_normalized_rewards:
                num_inliers = ransac_inliers.sum().item()
                num_kps = len(kps1)

                reward = reward / (num_inliers / num_kps + 0.01)

            dense_returns[
                abs_idx_matches[:, 0][ransac_inliers],
                abs_idx_matches[:, 1][ransac_inliers],
            ] = reward

            dense_returns = dense_returns[padding_mask1, :][:, padding_mask2]

            if penalty != 0.0:
                # pos. pair: small penalty for not finding a match
                # neg. pair: small reward for not finding a match
                penalty_val = penalty if label else -penalty

                dense_returns[dense_returns == 0.0] = penalty_val
        elif training_mode == "positive_only":
            # initialize all rewards with penalty (for not finding a match)
            dense_returns = torch.full((len(kps1), len(kps2)), fill_value=penalty, device=kps1.device)

            # Use Sampson distance-based rewards if Fm and kernel_fn are provided
            if distance_based_reward and Fm is not None:
                # Convert Fm to torch tensor if it's numpy array
                if isinstance(Fm, np.ndarray):
                    Fm = torch.tensor(Fm, device=kps1.device, dtype=torch.float32)

                m_kpts1 = kps1[abs_idx_matches[:, 0]]
                m_kpts2 = kps2[abs_idx_matches[:, 1]]

                # convert to pixel coordinates for distance calculation
                m_kpts1 = unnormalize_coords(m_kpts1, H, W)
                m_kpts2 = unnormalize_coords(m_kpts2, H, W)

                sampson_distances = sampson_epipolar_distance(
                    m_kpts1.unsqueeze(0), m_kpts2.unsqueeze(0), Fm.unsqueeze(0)
                ).squeeze(0)

                distance_based_rewards = kernel_fn(sampson_distances, outlier_penalty=outlier_penalty)

                dense_returns[abs_idx_matches[:, 0], abs_idx_matches[:, 1]] = distance_based_rewards
            else:
                if use_normalized_rewards:
                    num_inliers = ransac_inliers.sum().item()
                    num_kps = len(kps1)
                    num_outliers = len(ransac_inliers) - num_inliers

                    inlier_reward = inlier_reward / (num_inliers / num_kps + 0.01)
                    outlier_penalty = outlier_penalty / (num_outliers / num_kps + 0.01)

                # Discrete rewards (original behavior)
                # reward matches that are inliers
                dense_returns[
                    abs_idx_matches[:, 0][ransac_inliers],
                    abs_idx_matches[:, 1][ransac_inliers],
                ] = inlier_reward

                # penalize matches that are outliers
                dense_returns[
                    abs_idx_matches[:, 0][~ransac_inliers],
                    abs_idx_matches[:, 1][~ransac_inliers],
                ] = outlier_penalty

            dense_returns = dense_returns[padding_mask1, :][:, padding_mask2]
        else:
            raise ValueError("training_mode must be either 'positive_only' or 'positive_and_negative'")

    return dense_returns


def get_values_baseline(
    num_cells,
    abs_idx_matches,
    ransac_inliers,
    label,
    device,
):
    with torch.no_grad():
        reward = 1.0 if label else -1.0

        dense_values_baseline = torch.zeros((num_cells, num_cells), device=device)
        dense_values_baseline[abs_idx_matches[:, 0][ransac_inliers], abs_idx_matches[:, 1][ransac_inliers]] = reward

    return dense_values_baseline


def get_positive_corrs(kps1, kps2, H, px_thr=1.5):
    with torch.no_grad():
        warped = warp_pts(kps2["xy"], torch.inverse(H))

        d_mat = torch.cdist(kps1["xy"], warped)
        x_vmins, x_mins = torch.min(d_mat, dim=1)
        y_mins = torch.arange(len(x_mins), device=d_mat.device).long()

        # grab indices of positive correspondences & filter too close kps in the same image
        y_mins = y_mins[(x_vmins < px_thr)]  # * (self_vmins > 2.)]
        x_mins = x_mins[(x_vmins < px_thr)]  # * (self_vmins > 2.)]

    return (
        torch.hstack((y_mins.unsqueeze(1), x_mins.unsqueeze(1))),
        kps1["patches"][y_mins],
        kps2["patches"][x_mins],
    )


def get_dense_rewards(kps1, kps2, H, penalty=0.0, px_thr=1.5):
    with torch.no_grad():
        warped = warp_pts(kps2, torch.inverse(H))

        d_mat = torch.cdist(kps1, warped)
        x_vmins, x_mins = torch.min(d_mat, dim=1)
        y_mins = torch.arange(len(x_mins)).long()

        d_mat[y_mins, x_mins] *= -1.0
        d_mat[d_mat >= 0.0] = 0.0
        d_mat[d_mat < -px_thr] = 0.0
        d_mat[d_mat != 0.0] = 1.0

        reward_mat = d_mat
        reward_sum = reward_mat.sum()
        reward_mat[reward_mat == 0.0] = penalty
    return reward_mat, reward_sum


def dry_run_print(
    p1,
    p2,
    net,
    conf_inference,
    i=0,
    mode="start",
    transformation_model="homography",
    output_path=Path("."),
    save_outputs: bool = True,
):
    estimate_matches_image_pair(
        src_image=p1,
        trg_image=p2,
        extractor=net,
        transformation_model=transformation_model,
        image_name=f"dry_run_{mode}_{i}",
        output_path=output_path,
        conf_inference=conf_inference,
        save_outputs=save_outputs,
    )


def estimate_matches_image_pair(
    src_image,
    trg_image,
    extractor,
    transformation_model,
    image_name,
    output_path,
    conf_inference,
    save_outputs: bool = True,
):
    if save_outputs:
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

    if src_image.dim() == 3:
        src_image = src_image.unsqueeze(0)
        trg_image = trg_image.unsqueeze(0)

    org_src_image = None
    org_trg_image = None

    if save_outputs:
        org_src_image = src_image.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0)
        org_trg_image = trg_image.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0)

        # revert normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        org_src_image = std * org_src_image + mean
        org_trg_image = std * org_trg_image + mean

        org_src_image = (org_src_image * 255).astype(np.uint8)
        org_trg_image = (org_trg_image * 255).astype(np.uint8)

        # convert from RGB to BGR
        org_src_image = org_src_image[..., ::-1]
        org_trg_image = org_trg_image[..., ::-1]

    with torch.no_grad():
        # Compute kps and features
        try:
            kps1, descs1, score_1 = extractor.detectAndCompute(src_image, **conf_inference)
            kps2, descs2, score_2 = extractor.detectAndCompute(trg_image, **conf_inference)
        # check if it is RuntimeError with "No keypoints detected."
        except RuntimeError as e:
            if "No keypoints detected" in str(e):
                print(f"No keypoints detected. Skipping image pair {image_name}")
                return
            else:
                raise e

        # In distributed setups (e.g. SyncBatchNorm), running detectAndCompute may require
        # collectives across all ranks. Allow non-zero ranks to participate without doing any
        # visualization or file I/O.
        if not save_outputs:
            return

        assert org_src_image is not None and org_trg_image is not None

        if kps1.dim() == 3:
            # check batch sizes are 1
            assert kps1.size(0) == 1 and kps2.size(0) == 1, "Batch size should be 1"

            kps1, descs1, score_1 = (
                kps1.squeeze(0),
                descs1.squeeze(0),
                score_1.squeeze(0),
            )
            kps2, descs2, score_2 = (
                kps2.squeeze(0),
                descs2.squeeze(0),
                score_2.squeeze(0),
            )

        matcher = KF.DescriptorMatcher("mnn")
        dists, indexes = matcher(descs1, descs2)

        cv2_matches = cv2_matches_from_kornia(dists, indexes)

        # do RANSAC
        matched_src_pts = kps1[indexes[:, 0]]
        matched_trg_pts = kps2[indexes[:, 1]]

        kps1 = to_cv_kpts(kps1, score_1)
        kps2 = to_cv_kpts(kps2, score_2)

        if len(matched_src_pts) < 8:
            print(f"Not enough matches. Found only {len(matched_src_pts)} matches")

            result = cv2.drawMatches(
                org_src_image,
                kps1,
                org_trg_image,
                kps2,
                cv2_matches,
                None,
                matchColor=(0, 255, 0),
                matchesMask=None,
                # matchesMask=None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )

        else:
            H, mask = KG.ransac.RANSAC(model_type=transformation_model, inl_th=0.5)(matched_src_pts, matched_trg_pts)
            matchesMask = mask.int().ravel().tolist()

            # Draw RAW matches
            result = cv2.drawMatches(
                org_src_image,
                kps1,
                org_trg_image,
                kps2,
                cv2_matches,
                None,
                matchColor=(0, 255, 0),
                matchesMask=matchesMask,
                singlePointColor=(0, 0, 255),
                flags=cv2.DrawMatchesFlags_DEFAULT,
            )

        plt.figure(figsize=(15, 10))
        plt.imshow(result[..., ::-1])

        plt.tight_layout()
        plt.savefig(output_path / f"{image_name}.png")
        plt.close()


def get_other_random_id(idx: int, len_dataset: int, min_dist: int = 20):
    for _ in range(10):
        tgt_id = random.randint(0, len_dataset - 1)
        if abs(idx - tgt_id) >= min_dist:
            return tgt_id

    raise ValueError(f"Could not find target image with distance >= {min_dist} from source image {idx}")


def cv_resize_and_pad_to_shape(image, new_shape, padding_color=(0, 0, 0)):
    """Resizes image to new_shape with maintaining the aspect ratio and pads with padding_color if
    needed.

    Params:
        image: Image to be resized.
        new_shape: Expected (height, width) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    h, w = image.shape[:2]

    scale_h = new_shape[0] / h
    scale_w = new_shape[1] / w

    scale = None
    if scale_w * h > new_shape[0]:
        scale = scale_h
    elif scale_h * w > new_shape[1]:
        scale = scale_w
    else:
        scale = max(scale_h, scale_w)

    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    image = cv2.resize(image, (new_w, new_h))

    missing_h = new_shape[0] - new_h
    missing_w = new_shape[1] - new_w

    top, bottom = missing_h // 2, missing_h - (missing_h // 2)
    left, right = missing_w // 2, missing_w - (missing_w // 2)

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image


def top_m_mask(t, M):
    """Create a maskof the same shape as t, where the M largest values in t along dim=1 are set to
    1, and the rest are set to 0.

    Args:
        t (torch.Tensor): Input tensor of shape (B, N).
        M (int): Number of top values to select in each row.

    Returns:
        torch.Tensor: Binary mask tensor of shape (B, N).
    """
    B, N = t.shape
    assert M < N, "M must be smaller than N"

    # Get the indices of the top M values per batch
    top_indices = torch.topk(t, M, dim=1).indices

    # Create a zero mask
    mask = torch.zeros_like(t, dtype=torch.bool)

    # Set the selected indices to 1
    mask.scatter_(1, top_indices, 1)

    return mask


def get_required_keys(dict, required_keys):
    # create a function the returns the values for the required keys or raises an error if not all keys are present

    for key in required_keys:
        if key not in dict:
            raise ValueError(f"Key {key} not found in input dictionary. Available keys: {list(dict.keys())}")

    vals = [dict[key] for key in required_keys]

    return vals if len(vals) > 1 else vals[0]


def get_batch(b, *args):
    vals = [arg[b] for arg in args]

    return vals if len(vals) > 1 else vals[0]


def resize_image(image, min_size=512, max_size=768):
    """Resize image to a new size while maintaining the aspect ratio.

    Params:
        image (torch.tensor): Image to be resized.
        min_size (int): Minimum size of the smaller dimension.
        max_size (int): Maximum size of the larger dimension.

    Returns:
        image: Resized image.
    """

    h, w = image.shape[-2:]

    aspect_ratio = w / h

    if w > h:
        new_w = max(min_size, min(max_size, w))
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = max(min_size, min(max_size, h))
        new_w = int(new_h * aspect_ratio)

    new_size = (new_h, new_w)

    image = resize(image, new_size)

    return image
