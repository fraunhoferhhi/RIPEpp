"""
Shared utilities for SCARED dataset processing and visualization.
"""

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile

logger = logging.getLogger(__name__)

# Image dimensions (fixed for SCARED)
WIDTH, HEIGHT = 1280, 1024


def load_scene_points(
    test_dir: Path, dataset: str, keyframe: str, frame_id: str
) -> np.ndarray:
    """Load scene points TIFF for a frame (left image only).

    The TIFF contains left image stacked on top of right image (2048 x 1280).
    We only use the left image (top 1024 rows).

    Args:
        test_dir: Path to test directory
        dataset: Dataset name (e.g., "dataset_8")
        keyframe: Keyframe name (e.g., "keyframe_1")
        frame_id: Frame ID (e.g., "000123")

    Returns:
        np.ndarray of shape (1024, 1280, 3) with 3D coordinates per pixel.
        Invalid pixels have value (0, 0, 0).
    """
    tiff_path = (
        test_dir
        / dataset
        / keyframe
        / "data"
        / "scene_points"
        / f"scene_points{frame_id}.tiff"
    )

    if not tiff_path.exists():
        raise FileNotFoundError(f"Scene points file not found: {tiff_path}")

    img = tifffile.imread(str(tiff_path))

    # Extract left image (top 1024 rows)
    left = img[:HEIGHT, :, :]
    return left


def load_rgb_image(
    test_dir: Path, dataset: str, keyframe: str, frame_id: str
) -> np.ndarray:
    """Load RGB image for a frame (left image).

    Args:
        test_dir: Path to test directory
        dataset: Dataset name (e.g., "dataset_8")
        keyframe: Keyframe name (e.g., "keyframe_1")
        frame_id: Frame ID (e.g., "000123")

    Returns:
        np.ndarray of shape (1024, 1280, 3) RGB image
    """
    img_path = (
        test_dir
        / dataset
        / keyframe
        / "data"
        / "rgb_frames_left"
        / f"frame_{frame_id}.png"
    )

    if not img_path.exists():
        raise FileNotFoundError(f"RGB image not found: {img_path}")

    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def compute_projection(pts1: np.ndarray, meta1: dict, meta2: dict, K: np.ndarray):
    """Project 3D points from frame1's camera to frame2's image plane.

    Args:
        pts1: Scene points array (1024, 1280, 3) in frame1 camera coordinates
        meta1: Frame1 metadata with R and T
        meta2: Frame2 metadata with R and T
        K: Camera intrinsics matrix (3x3)

    Returns:
        tuple: (pts2_2d, in_bounds, valid1_mask)
            - pts2_2d: 2D projected points (N, 2) for points with positive depth
            - in_bounds: boolean mask for pts2_2d indicating which are in image bounds
            - valid1_mask: boolean mask (H, W) of valid pixels in frame1
    """
    # Get valid mask for frame1
    valid1 = np.any(pts1 != 0, axis=-1)  # (1024, 1280) bool

    # Get 3D points from frame1 (in camera1 coordinates)
    pts1_cam1 = pts1[valid1]  # (N, 3)

    if len(pts1_cam1) == 0:
        return np.array([]).reshape(0, 2), np.array([], dtype=bool), valid1

    # Scene points are in CAMERA coordinates
    # Transform: cam1 -> world -> cam2
    R1, T1 = meta1["R"], meta1["T"]
    R2, T2 = meta2["R"], meta2["T"]

    # Compute relative transformation
    R_rel = R2 @ R1.T
    T_rel = -R2 @ R1.T @ T1 + T2

    # Transform from camera1 to camera2 coordinates
    pts2_cam = (R_rel @ pts1_cam1.T).T + T_rel  # (N, 3)

    # Project to frame2 image (only points in front of camera)
    valid_depth = pts2_cam[:, 2] > 0
    if not np.any(valid_depth):
        return np.array([]).reshape(0, 2), np.array([], dtype=bool), valid1

    pts2_proj = pts2_cam[valid_depth]
    pts2_2d = (K @ pts2_proj.T).T
    pts2_2d = pts2_2d[:, :2] / pts2_2d[:, 2:3]

    # Check bounds
    in_bounds = (
        (pts2_2d[:, 0] >= 0)
        & (pts2_2d[:, 0] < WIDTH)
        & (pts2_2d[:, 1] >= 0)
        & (pts2_2d[:, 1] < HEIGHT)
    )

    return pts2_2d, in_bounds, valid1


def compute_overlap(pts1: np.ndarray, meta1: dict, meta2: dict, K: np.ndarray) -> float:
    """Compute overlap ratio from frame1 to frame2.

    Args:
        pts1: Scene points array (1024, 1280, 3) in frame1 camera coordinates
        meta1: Frame1 metadata with R and T
        meta2: Frame2 metadata with R and T
        K: Camera intrinsics matrix (3x3)

    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    pts2_2d, in_bounds, valid1 = compute_projection(pts1, meta1, meta2, K)

    valid_count = valid1.sum()
    if valid_count == 0:
        return 0.0

    return in_bounds.sum() / valid_count


def visualize_projection(
    test_dir: Path,
    dataset: str,
    keyframe: str,
    f1: str,
    f2: str,
    meta1: dict,
    meta2: dict,
    output_path: Path,
    title: str = "",
    subsample: int = 10,
):
    """Create visualization of projection from frame1 to frame2.

    Args:
        test_dir: Path to test directory
        dataset: Dataset name
        keyframe: Keyframe name
        f1: Frame1 ID
        f2: Frame2 ID
        meta1: Frame1 metadata
        meta2: Frame2 metadata
        output_path: Path to save the image
        title: Optional title prefix
        subsample: Subsample factor for point plotting
    """
    # Load data
    pts1 = load_scene_points(test_dir, dataset, keyframe, f1)
    pts2 = load_scene_points(test_dir, dataset, keyframe, f2)
    rgb1 = load_rgb_image(test_dir, dataset, keyframe, f1)
    rgb2 = load_rgb_image(test_dir, dataset, keyframe, f2)

    # Build camera matrix
    K = np.array(
        [[meta1["fx"], 0, meta1["cx"]], [0, meta1["fy"], meta1["cy"]], [0, 0, 1]]
    )

    # Compute projection
    pts2_2d, in_bounds, valid1 = compute_projection(pts1, meta1, meta2, K)

    # Compute overlap
    overlap = in_bounds.sum() / valid1.sum() if valid1.sum() > 0 else 0.0

    # Valid masks
    valid2 = np.any(pts2 != 0, axis=-1)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Frame 1
    axes[0, 0].imshow(rgb1)
    axes[0, 0].set_title(f"Frame {f1} - RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(valid1, cmap="gray")
    axes[0, 1].set_title(f"Frame {f1} - Valid pixels\n({valid1.sum()} pixels)")
    axes[0, 1].axis("off")

    # Depth visualization
    depth1 = np.zeros((HEIGHT, WIDTH))
    depth1[valid1] = pts1[valid1][:, 2]
    axes[0, 2].imshow(depth1, cmap="viridis")
    axes[0, 2].set_title(f"Frame {f1} - Z-coordinate")
    axes[0, 2].axis("off")

    # Row 2: Frame 2
    axes[1, 0].imshow(rgb2)
    axes[1, 0].set_title(f"Frame {f2} - RGB")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(valid2, cmap="gray")
    axes[1, 1].set_title(f"Frame {f2} - Valid pixels\n({valid2.sum()} pixels)")
    axes[1, 1].axis("off")

    # RGB with projected points overlay
    axes[1, 2].imshow(rgb2)

    if len(pts2_2d) > 0:
        # In-bounds points (green)
        pts_inbounds = pts2_2d[in_bounds][::subsample]
        if len(pts_inbounds) > 0:
            axes[1, 2].scatter(
                pts_inbounds[:, 0],
                pts_inbounds[:, 1],
                c="green",
                s=1,
                alpha=0.5,
                label=f"In bounds ({in_bounds.sum()})",
            )

        # Out-of-bounds points (red)
        pts_outbounds = pts2_2d[~in_bounds][::subsample]
        if len(pts_outbounds) > 0:
            axes[1, 2].scatter(
                pts_outbounds[:, 0],
                pts_outbounds[:, 1],
                c="red",
                s=1,
                alpha=0.3,
                label=f"Out of bounds ({(~in_bounds).sum()})",
            )

        axes[1, 2].legend(loc="upper right")

    axes[1, 2].set_title(f"Frame {f2} - Projected points\nOverlap: {overlap:.2%}")
    axes[1, 2].axis("off")

    suptitle = f"{dataset}/{keyframe}: Frame {f1} → Frame {f2}"
    if title:
        suptitle = f"{title}\n{suptitle}"
    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
