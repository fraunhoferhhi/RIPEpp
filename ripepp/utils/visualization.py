from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def save_image_tensor(img_tensor, output_path, name="image"):
    """
    Save an image tensor as PNG.

    Args:
        img_tensor: Tensor of shape [C, H, W] or [B, C, H, W] (normalized or unnormalized)
        output_path: Path to save the image
        name: Name for the output file
    """
    # Handle batch dimension
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]

    # Move to CPU and convert to numpy [C, H, W] -> [H, W, C]
    img_np = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)

    # Handle normalization - check if values are in [-1, 1] or [0, 1] range
    img_min = img_np.min()

    # If normalized with ImageNet stats, denormalize
    if img_min < 0:  # Likely normalized
        # Standard ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean

    # Clip to [0, 1]
    img_np = np.clip(img_np, 0, 1)

    # Save
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 10))
    plt.imshow(img_np)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path / f"{name}.png", dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


def visualize_descriptor_map_pca(
    desc_map, output_path, name="descriptor_map", input_image=None
):
    """
    Visualize descriptor map by reducing to 3 channels using PCA.

    Args:
        desc_map: Tensor of shape [C, H, W] or [B, C, H, W]
        output_path: Path to save the visualization
        name: Name for the output file
        input_image: Optional input image tensor to save alongside for reference
    """
    # Save input image if provided
    if input_image is not None:
        save_image_tensor(input_image, output_path, name=f"{name}_input")

    # Handle batch dimension
    if desc_map.dim() == 4:
        desc_map = desc_map[0]  # Take first in batch
    elif desc_map.dim() != 3:
        raise ValueError(
            f"Expected descriptor map to have 3 or 4 dimensions, got {desc_map.dim()}"
        )

    # desc_map is [C, H, W]
    C, H, W = desc_map.shape

    if C < 3:
        print(
            f"Warning: Descriptor has only {C} channels, cannot apply PCA for 3 components. Skipping."
        )
        return None

    # Move to CPU and convert to numpy
    desc_np = desc_map.detach().cpu().numpy()

    # Reshape to [H*W, C] for PCA
    desc_flat = desc_np.reshape(C, -1).T  # [H*W, C]

    # Apply PCA to reduce to 3 components
    pca = PCA(n_components=3)
    desc_pca = pca.fit_transform(desc_flat)  # [H*W, 3]

    # Reshape back to [H, W, 3]
    desc_pca = desc_pca.reshape(H, W, 3)

    # Normalize to [0, 1] for each channel independently
    for i in range(3):
        channel = desc_pca[:, :, i]
        channel_min = channel.min()
        channel_max = channel.max()
        if channel_max - channel_min > 1e-6:
            desc_pca[:, :, i] = (channel - channel_min) / (channel_max - channel_min)
        else:
            desc_pca[:, :, i] = 0.5  # Set to mid-gray if channel is constant

    # Save as image
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 10))
    plt.imshow(desc_pca)
    plt.axis("off")
    plt.title(
        f"PCA Descriptor Visualization\nExplained variance: {pca.explained_variance_ratio_.sum():.2%}"
    )
    plt.tight_layout()
    plt.savefig(output_path / f"{name}.png", dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"Saved descriptor visualization to {output_path / f'{name}.png'}")
    print(f"  Shape: [{C}, {H}, {W}] -> [{H}, {W}, 3]")
    print(
        f"  PCA explained variance ratio: {pca.explained_variance_ratio_} (total: {pca.explained_variance_ratio_.sum():.2%})"
    )

    return desc_pca
