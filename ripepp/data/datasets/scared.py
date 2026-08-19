import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torchvision.io import read_image

from ripepp import utils
from ripepp.data.data_transforms import Compose

log = utils.get_pylogger(__name__)

from .base_dataset import BaseDataset


class SCARED(BaseDataset):
    """
    SCARED surgical endoscopy dataset for video frame pairs.

    The dataset contains multiple datasets with keyframes, each containing video frames.
    Pairs are created from frames at a fixed distance N apart within the same keyframe.

    Directory structure:
        {root}/{stage}/dataset_*/keyframe_*/data/frame_data/rgb_frames_left/frame_*.png

    Args:
        root: Path to dataset root directory
        frame_distance: Distance N between frames to form a pair (e.g., pair frame i with frame i+N)
        stage: Dataset stage ('train', 'val', or 'test')
        transforms: Optional image transformations to apply
        positive_only: If True, return only positive pairs. If False, interleave positive/negative pairs.
    """

    def __init__(
        self,
        root: str,
        frame_distance: int = 5,
        stage: str = "train",
        transforms: Optional[Callable] = None,
        positive_only: bool = True,
    ) -> None:
        super().__init__(positive_only=positive_only)

        self.root = Path(root)
        self.frame_distance = frame_distance
        self.stage = stage
        self.transforms = transforms
        self.epoch = 0

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset not found at {self.root}")

        if transforms is None:
            self.transforms = Compose([])
        else:
            self.transforms = transforms

        if self.stage not in ["train", "val"]:
            raise RuntimeError(
                f"Unknown option '{self.stage}' as training stage variable. Valid options: 'train', 'val', 'test'"
            )

        # Build pairs
        self._build_pairs()

        if not self.positive_only and len(self.dataset_names) < 2:
            raise ValueError("At least two datasets in SCARED are required to create negative pairs.")

        log.info(
            f"Initialized SCARED dataset with {len(self.positive_pairs)} positive pairs "
            f"(frame_distance={frame_distance}, positive_only={self.positive_only})"
        )

    def _build_pairs(self) -> None:
        """Build pairs from all datasets and keyframes."""
        self.positive_pairs: List[Tuple[str, str, str, str]] = []  # (src_path, trg_path, dataset_name, keyframe_name)
        self.pairs_by_dataset: Dict[str, List[int]] = {}  # dataset_name -> list of pair indices
        self.dataset_names: List[str] = []

        stage_path = self.root / self.stage

        if not stage_path.exists():
            raise FileNotFoundError(f"Stage directory not found at {stage_path}")

        # Iterate through dataset_* directories
        dataset_dirs = sorted(stage_path.glob("dataset_*"))

        if len(dataset_dirs) == 0:
            raise FileNotFoundError(f"No dataset directories found in {stage_path}")

        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name
            dataset_pair_indices: List[int] = []

            # Iterate through keyframe_* directories
            keyframe_dirs = sorted(dataset_dir.glob("keyframe_*"))

            for keyframe_dir in keyframe_dirs:
                keyframe_name = keyframe_dir.name
                frames_dir = keyframe_dir / "data" / "rgb_frames_left"

                if not frames_dir.exists():
                    log.warning(f"Frames directory not found at {frames_dir}, skipping")
                    continue

                # Get all frame files and sort by frame number
                frame_files = sorted(
                    frames_dir.glob("frame_*.png"),
                    key=lambda x: int(x.stem.replace("frame_", "")),
                )

                if len(frame_files) < self.frame_distance + 1:
                    log.warning(
                        f"Keyframe {keyframe_dir} has only {len(frame_files)} frames, "
                        f"need at least {self.frame_distance + 1} for frame_distance={self.frame_distance}, skipping"
                    )
                    continue

                # Create pairs with distance N
                num_pairs = 0
                for i in range(len(frame_files) - self.frame_distance):
                    src_frame = frame_files[i]
                    trg_frame = frame_files[i + self.frame_distance]

                    pair_idx = len(self.positive_pairs)
                    self.positive_pairs.append((str(src_frame), str(trg_frame), dataset_name, keyframe_name))
                    dataset_pair_indices.append(pair_idx)
                    num_pairs += 1

                if num_pairs > 0:
                    log.info(f"  {dataset_name}/{keyframe_name}: {len(frame_files)} frames -> {num_pairs} pairs")

            if len(dataset_pair_indices) > 0:
                self.pairs_by_dataset[dataset_name] = dataset_pair_indices
                self.dataset_names.append(dataset_name)

        log.info(f"Found {len(self.dataset_names)} datasets with pairs: {self.dataset_names}")

    def _get_negative_pair(self, pair_idx: int) -> Tuple[str, str]:
        """
        Get a negative pair for the given positive pair index.

        Source image comes from the positive pair, target image comes from a different dataset.

        Args:
            pair_idx: Index of the positive pair

        Returns:
            Tuple of (src_path, trg_path) for the negative pair
        """
        src_path, _, dataset_name, _ = self.positive_pairs[pair_idx]

        # Select a different dataset
        other_datasets = [d for d in self.dataset_names if d != dataset_name]
        other_dataset = random.choice(other_datasets)

        # Select a random pair from the other dataset
        other_pair_idx = random.choice(self.pairs_by_dataset[other_dataset])
        _, trg_path, _, _ = self.positive_pairs[other_pair_idx]

        return src_path, trg_path

    def resample(self) -> None:
        """
        Resample pairs (for compatibility with resampling mechanism).

        For video sequences, pairs are deterministic, so we just increment the epoch counter.
        """
        self.epoch += 1
        log.info(f"SCARED epoch {self.epoch} (pairs unchanged for video sequences)")

    def __len__(self) -> int:
        if self.positive_only:
            return len(self.positive_pairs)
        return 2 * len(self.positive_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Determine if this is a positive or negative sample
        positive_sample = idx % 2 == 0 or self.positive_only
        if not self.positive_only:
            pair_idx = idx // 2
        else:
            pair_idx = idx

        # Get paths
        if positive_sample:
            src_path, trg_path, _, _ = self.positive_pairs[pair_idx]
        else:
            src_path, trg_path = self._get_negative_pair(pair_idx)

        # Load images
        src_img = read_image(src_path) / 255.0
        trg_img = read_image(trg_path) / 255.0

        _, H_src, W_src = src_img.shape
        _, H_trg, W_trg = trg_img.shape

        # Create masks (all ones, no padding initially)
        src_mask = torch.ones((1, H_src, W_src), dtype=torch.uint8)
        trg_mask = torch.ones((1, H_trg, W_trg), dtype=torch.uint8)

        # Apply transforms
        if self.transforms:
            src_img, src_mask = self.transforms(src_img, src_mask)
            trg_img, trg_mask = self.transforms(trg_img, trg_mask)

        sample = {
            "src_image": src_img,
            "trg_image": trg_img,
            "src_mask": src_mask.to(torch.bool),
            "trg_mask": trg_mask.to(torch.bool),
            "label": positive_sample,
            "src_path": src_path,
            "trg_path": trg_path,
        }

        return sample
