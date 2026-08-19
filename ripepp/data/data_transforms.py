import collections
import collections.abc
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as TF


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class Transform:
    def __init__(self):
        pass

    def apply_transform(self, img, mask, transform_function):
        img, mask = transform_function(img, mask)
        return img, mask


class Normalize(Transform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        return self.apply_transform(img, mask, self.transform_function)

    def transform_function(self, img, mask):
        img = TF.normalize(img, mean=self.mean, std=self.std)
        return img, mask


class Crop(Transform):
    def __init__(self, crop_height, crop_width):
        if crop_height % 2 != 0 or crop_width % 2 != 0:
            raise ValueError("Crop dimensions must be even")
        self.output_size = (crop_height, crop_width)

    def __call__(self, img, mask):
        return self.apply_transform(img, mask, self.transform_function)

    def transform_function(self, img, mask):
        # pad if needed to ensure even dimensions
        if img.shape[-1] % 2 != 0:
            img = TF.pad(img, [0, 1])
            mask = TF.pad(mask, [0, 1])
        if img.shape[-2] % 2 != 0:
            img = TF.pad(img, [0, 0, 0, 1])
            mask = TF.pad(mask, [0, 0, 0, 1])

        x1 = (img.shape[-1] - self.output_size[1]) // 2
        y1 = (img.shape[-2] - self.output_size[0]) // 2
        img_crop = img[:, y1 : y1 + self.output_size[0], x1 : x1 + self.output_size[1]]
        mask_crop = mask[
            :, y1 : y1 + self.output_size[0], x1 : x1 + self.output_size[1]
        ]
        return img_crop, mask_crop


class ResizeAndPad(Transform):
    def __init__(self, target_size_longer_side=768, fill_value=0):
        self.target_size = target_size_longer_side
        self.fill_value = fill_value
        if fill_value not in (0, 1):
            raise ValueError("Fill value must be either 0 or 1")

    def __call__(self, img, mask):
        return self.apply_transform(img, mask, self.transform_function)

    def transform_function(self, img, mask):
        w, h = img.shape[-1], img.shape[-2]
        _, new_w, new_h = self.compute_resize(w, h)
        img_resized = TF.resize(img, [new_h, new_w])
        mask_resized = TF.resize(mask, [new_h, new_w], TF.InterpolationMode.NEAREST)
        img_padded, _ = self.apply_padding(img_resized, new_w, new_h)
        mask_padded, _ = self.apply_padding(mask_resized, new_w, new_h)
        return img_padded, mask_padded

    def compute_resize(self, w, h):
        if w > h:
            scale = self.target_size / w
            new_w = self.target_size
            new_h = int(h * scale)
        else:
            scale = self.target_size / h
            new_h = self.target_size
            new_w = int(w * scale)
        return scale, new_w, new_h

    def apply_padding(self, img, new_w, new_h):
        pad_w = (self.target_size - new_w) // 2
        pad_h = (self.target_size - new_h) // 2
        padding = [
            pad_w,
            pad_h,
            self.target_size - new_w - pad_w,
            self.target_size - new_h - pad_h,
        ]
        img_padded = TF.pad(img, padding, fill=self.fill_value)
        return img_padded, padding


class RandomAffine(Transform):
    def __init__(self, degrees, translate, scale):
        if isinstance(degrees, (int, float)):
            degrees = (-degrees, degrees)
        self.degrees = degrees
        self.translate = translate
        self.scale = scale if scale is not None else 0.0

    def __call__(self, img, mask):
        return self.apply_transform(img, mask, self.transform_function)

    def transform_function(self, img, mask):
        img_t, mask_t, _ = self._affine(img, mask.to(torch.float32))
        return img_t, mask_t.to(torch.uint8)

    def _affine(self, img, mask):
        angle = random.uniform(self.degrees[0], self.degrees[1])
        translation = (
            random.randint(-self.translate[0], self.translate[0]),
            random.randint(-self.translate[1], self.translate[1]),
        )
        scale = random.uniform(1 - self.scale, 1 + self.scale)
        theta = self._get_affine_matrix(
            angle, translation, scale, img.shape[-1], img.shape[-2]
        )
        grid = F.affine_grid(
            theta.unsqueeze(0), img.unsqueeze(0).size(), align_corners=False
        )
        img_transformed = F.grid_sample(
            img.unsqueeze(0), grid, align_corners=False
        ).squeeze(0)
        mask_transformed = F.grid_sample(
            mask.unsqueeze(0), grid, align_corners=False, mode="nearest"
        ).squeeze(0)
        H = self._get_unnormalized_affine_matrix(theta, img.shape[-1], img.shape[-2])
        H = torch.inverse(H)  # retained if needed later
        return img_transformed, mask_transformed, H

    def _get_affine_matrix(
        self, angle, translate, scale, width, height, inverted=False
    ):
        rot = np.deg2rad(angle)
        tx, ty = translate
        cx, cy = 0.0, 0.0
        tx /= width // 2
        ty /= height // 2
        theta = self._build_transformation_matrix(tx, ty, rot, cx, cy, scale, inverted)
        return theta

    def _build_transformation_matrix(self, tx, ty, rot, cx, cy, scale, inverted=False):
        a = math.cos(rot)
        b = -math.sin(rot)
        c = math.sin(rot)
        d = math.cos(rot)
        if inverted:
            # Inverted rotation matrix with scale and shear
            # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
            matrix = [d, -b, 0.0, -c, a, 0.0]
            matrix = [x / scale for x in matrix]
            # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
            matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
            matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
            # Apply center translation: C * RSS^-1 * C^-1 * T^-1
            matrix[2] += cx
            matrix[5] += cy
        else:
            matrix = [a, b, 0.0, c, d, 0.0]
            matrix = [x * scale for x in matrix]
            # Apply inverse of center translation: RSS * C^-1
            matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
            matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
            # Apply translation and center : T * C * RSS * C^-1
            matrix[2] += cx + tx
            matrix[5] += cy + ty
        return torch.tensor(matrix).reshape(2, 3)

    def _get_unnormalized_affine_matrix(self, theta, width, height):
        # from: https://discuss.pytorch.org/t/affine-transformation-matrix-paramters-conversion/19522/18
        theta_h = torch.eye(3, device=theta.device)
        theta_h[:2] = theta
        norm = torch.tensor([[2 / width, 0, -1], [0, 2 / height, -1], [0, 0, 1]])
        unnorm = torch.tensor(
            [[width / 2, 0, width / 2], [0, height / 2, height / 2], [0, 0, 1]]
        )
        H = unnorm @ theta_h @ norm

        return H


class DivisibleBy(Transform):
    def __init__(self, edge_divisible_by=8, antialias=True):
        """Resize the input image so that its dimensions are divisible by edge_divisible_by. Possibly affects the aspect ratio. Uses scaling with the least possible change to the original size."""
        self.edge_divisible_by = edge_divisible_by
        self.antialias = antialias

    def __call__(self, img, mask):
        return self.apply_transform(img, mask, self.transform_function)

    def transform_function(self, img, mask):
        new_size = self.get_new_image_size(img)
        img = TF.resize(img, new_size, antialias=self.antialias)
        mask = TF.resize(mask, new_size, TF.InterpolationMode.NEAREST)
        return img, mask

    def get_new_image_size(self, img):
        h, w = img.shape[-2:]

        df = self.edge_divisible_by

        new_h = round(h / df) * df
        new_w = round(w / df) * df

        return (new_h, new_w)


class Resize(Transform):
    def __init__(
        self, output_size, edge_divisible_by=None, side="long", antialias=True
    ):
        self.output_size = output_size
        self.edge_divisible_by = edge_divisible_by
        self.side = side
        self.antialias = antialias

    def __call__(self, img, mask):
        return self.apply_transform(img, mask, self.transform_function)

    def transform_function(self, img, mask):
        new_size = self.get_new_image_size(img)
        img = TF.resize(img, new_size, antialias=self.antialias)
        mask = TF.resize(mask, new_size, TF.InterpolationMode.NEAREST)
        return img, mask

    def get_new_image_size(self, img):
        h, w = img.shape[-2:]

        if isinstance(self.output_size, collections.abc.Iterable):
            assert len(self.output_size) == 2
            size = tuple(self.output_size)
        elif self.output_size is None:
            size = (h, w)
        else:
            side_size = self.output_size
            aspect_ratio = w / h
            if self.side not in ("short", "long", "vert", "horz"):
                raise ValueError(
                    f"side can be one of 'short', 'long', 'vert', and 'horz'. Got '{self.side}'"
                )
            if self.side == "vert":
                size = (side_size, int(side_size * aspect_ratio))
            elif self.side == "horz":
                size = (int(side_size / aspect_ratio), side_size)
            elif (self.side == "short") ^ (aspect_ratio < 1.0):
                size = (side_size, int(side_size * aspect_ratio))
            else:
                size = (int(side_size / aspect_ratio), side_size)
        if self.edge_divisible_by is not None:
            df = self.edge_divisible_by
            size = tuple(int(x // df * df) for x in size)
        return size


class CenterCropRespectingOrientation(Transform):
    def __init__(self, output_size):
        if isinstance(output_size, collections.abc.Iterable):
            assert len(output_size) == 2
            self.output_size = output_size
        else:
            self.output_size = (output_size, output_size)

    def __call__(self, img, mask):
        return self.apply_transform(img, mask, self.transform_function)

    def transform_function(self, img, mask):
        img = self.crop_respecting_orientation(img)
        mask = self.crop_respecting_orientation(mask)
        return img, mask

    def crop_respecting_orientation(self, img):
        H, W = img.shape[-2:]
        resolution_out = sorted(self.output_size)[:: +1 if W > H else -1]
        img = TF.center_crop(img, resolution_out)
        return img


class RotateToLandscape(Transform):
    def __call__(self, img, mask):
        return self.apply_transform(img, mask, self.transform_function)

    def transform_function(self, img, mask):
        img = self.rotate_to_landscape(img)
        mask = self.rotate_to_landscape(mask)
        return img, mask

    def rotate_to_landscape(self, img):
        H, W = img.shape[-2:]
        if W < H:
            img = TF.rotate(img, angle=90, expand=True)
            assert img.shape[-2:] == (W, H), (
                f"Expected shape {(W, H)}, got {img.shape[-2:]}"
            )
        return img


class RandomSolarize(Transform):
    def __init__(self, threshold=128, p=0.5):
        """
        Randomly solarize the image by inverting all pixel values above a threshold.

        Args:
            threshold: Pixel values above this will be inverted
            p: Probability of applying solarization
        """
        self.threshold = threshold
        self.p = p

    def __call__(self, img, mask):
        return self.apply_transform(img, mask, self.transform_function)

    def transform_function(self, img, mask):
        if random.random() < self.p:
            img = TF.solarize(img, threshold=self.threshold)
        return img, mask


class RandomColorJitter(Transform):
    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, p=0.5):
        """
        Randomly change the brightness, contrast, saturation and hue of an image.

        Args:
            brightness: How much to jitter brightness. brightness_factor is chosen
                uniformly from [max(0, 1 - brightness), 1 + brightness]
            contrast: How much to jitter contrast. contrast_factor is chosen
                uniformly from [max(0, 1 - contrast), 1 + contrast]
            saturation: How much to jitter saturation. saturation_factor is chosen
                uniformly from [max(0, 1 - saturation), 1 + saturation]
            hue: How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue].
                Should be in [0, 0.5]
            p: Probability of applying color jitter
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, img, mask):
        return self.apply_transform(img, mask, self.transform_function)

    def transform_function(self, img, mask):
        if random.random() < self.p:
            # Randomly sample the factors
            brightness_factor = None
            if self.brightness > 0:
                brightness_factor = random.uniform(
                    max(0, 1 - self.brightness), 1 + self.brightness
                )

            contrast_factor = None
            if self.contrast > 0:
                contrast_factor = random.uniform(
                    max(0, 1 - self.contrast), 1 + self.contrast
                )

            saturation_factor = None
            if self.saturation > 0:
                saturation_factor = random.uniform(
                    max(0, 1 - self.saturation), 1 + self.saturation
                )

            hue_factor = None
            if self.hue > 0:
                hue_factor = random.uniform(-self.hue, self.hue)

            # Apply transforms in random order (like torchvision.transforms.ColorJitter)
            transforms = []
            if brightness_factor is not None:
                transforms.append(
                    lambda img: TF.adjust_brightness(img, brightness_factor)
                )
            if contrast_factor is not None:
                transforms.append(lambda img: TF.adjust_contrast(img, contrast_factor))
            if saturation_factor is not None:
                transforms.append(
                    lambda img: TF.adjust_saturation(img, saturation_factor)
                )
            if hue_factor is not None:
                transforms.append(lambda img: TF.adjust_hue(img, hue_factor))

            random.shuffle(transforms)
            for t in transforms:
                img = t(img)

        return img, mask


class RandomGaussianBlur(Transform):
    def __init__(self, kernel_size=5, p=0.5):
        """
        Randomly apply Gaussian blur to the image.

        Args:
            kernel_size: Size of the Gaussian kernel. If int, uses square kernel.
            p: Probability of applying Gaussian blur
        """
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]
        else:
            self.kernel_size = kernel_size

        self.p = p

    def __call__(self, img, mask):
        return self.apply_transform(img, mask, self.transform_function)

    def transform_function(self, img, mask):
        if random.random() < self.p:
            # Apply Gaussian blur
            img = TF.gaussian_blur(img, kernel_size=self.kernel_size, sigma=None)

        return img, mask


class RandomGaussianNoise(Transform):
    def __init__(self, mean=0.0, std=0.1, p=0.5):
        """
        Randomly add Gaussian noise to the image.

        Args:
            mean: Mean of the Gaussian noise
            std: Standard deviation of the Gaussian noise. Can be a single float or
                a tuple (std_min, std_max) to sample uniformly
            p: Probability of adding Gaussian noise
        """
        self.mean = mean

        if isinstance(std, collections.abc.Iterable):
            self.std_min, self.std_max = std
        else:
            self.std_min = std
            self.std_max = std

        self.p = p

    def __call__(self, img, mask):
        return self.apply_transform(img, mask, self.transform_function)

    def transform_function(self, img, mask):
        if random.random() < self.p:
            # Sample std
            std = random.uniform(self.std_min, self.std_max)

            img = TF.gaussian_noise(img, mean=self.mean, sigma=std)

        return img, mask


class RandomRotate90(Transform):
    def __init__(self, p=0.2):
        """
        Randomly rotate image and mask by 90, 180, or 270 degrees.

        Args:
            p: Probability of applying rotation
        """
        self.p = p
        self.angles = [90, 180, 270]

    def __call__(self, img, mask):
        return self.apply_transform(img, mask, self.transform_function)

    def transform_function(self, img, mask):
        if random.random() < self.p:
            angle = random.choice(self.angles)
            img = TF.rotate(img, angle=angle, expand=True)
            mask = TF.rotate(mask, angle=angle, expand=True)
        return img, mask
