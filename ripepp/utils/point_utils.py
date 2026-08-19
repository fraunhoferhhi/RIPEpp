import numpy as np
import torch


def index_to_image_coordinates(index, img_dim):
    """Convert a 1D index to 2D image coordinates.

    Args:
        index: an integer representing the 1D index. (index 0 is the top-left corner of the image, index W*H-1 is the bottom-right corner, row-major order)
        img_dim: a tuple containing the dimensions of the image. [H, W]

    Returns:
        coordinates: a tuple containing the 2D coordinates of the image. [u, v]

        +------> u (W)
        |
        |
        |
        v

        v (H)
    """

    _, W = img_dim
    v = index // W
    u = index % W
    return u, v


def image_coordinates_to_index(coordinates, img_dim):
    """Convert 2D image coordinates to a 1D index.

    Args:
        coordinates: a tuple containing the 2D coordinates of the image. [u, v]
        img_dim: a tuple containing the dimensions of the image. [H, W]

    Returns:
        index: an integer representing the 1D index. (index 0 is the top-left corner of the image, index W*H-1 is the bottom-right corner, row-major order)

        +------> u (W)
        |
        |
        |
        v

        v (H)
    """

    if isinstance(coordinates, tuple):
        _, W = img_dim
        u, v = coordinates
        return v * W + u
    elif isinstance(coordinates, torch.Tensor) or isinstance(coordinates, np.ndarray):
        _, W = img_dim
        u = coordinates[:, 0]
        v = coordinates[:, 1]
        return v * W + u
    else:
        raise ValueError("coordinates must be a tuple, tensor or numpy array.")


def denormalize(img_dim, l_t):
    """Convert coordinates in the range [0, 1] to coordinates in the range [0, H] and [0, W],
    respectively.

    Args:
        img_dim: a matrix or tensor containing the dimensions of the image. [W, H] ATTENTION: W is the first element.
        l_t: a 2D tensor of shape (B, 2). Contains the glimpse
            coordinates [x, y] for the current timestep `t`.

           1^ y          ----> +------> W
            |                  |
            |                  |
           0+----> 1 x         v H

    Returns:
        l_t_absolute: a 2D tensor of shape (B, 2). Contains the
            denormalized glimpse coordinates [x, y] for the current
            timestep `t`.
    """

    # check if W is bigger than H
    # assert img_dim[0] > img_dim[1], "W must be bigger than H."

    if isinstance(img_dim, torch.Tensor):
        l_t_swapped = l_t.clone()
    elif isinstance(img_dim, np.ndarray):
        l_t_swapped = l_t.copy()
    else:
        raise ValueError("img_dim must be a tensor or numpy array.")

    # check if l_t is a single point
    if single_point := len(l_t_swapped.shape) == 1:
        l_t_swapped = l_t_swapped.reshape(1, -1)

    l_t_swapped[:, 1] = 1 - l_t_swapped[:, 1]

    if isinstance(l_t, torch.Tensor):
        l_t_absolute = (l_t_swapped * img_dim).floor()
        l_t_absolute = l_t_absolute.long()
    elif isinstance(l_t, np.ndarray):
        l_t_absolute = np.floor(l_t_swapped * img_dim)
        l_t_absolute = l_t_absolute.astype(int)
    else:
        raise ValueError("l_t must be a tensor or numpy array.")

    if single_point:
        l_t_absolute = l_t_absolute.squeeze()

    return l_t_absolute
