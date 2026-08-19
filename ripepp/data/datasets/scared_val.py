"""
SCARED Validation Dataset with overlap-filtered random pairs.

Samples random frame pairs from the same keyframe, filtered by minimum overlap ratio.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from ripepp import utils
from ripepp.data.data_transforms import Compose
from ripepp.utils.image_utils import Camera, cameras2F
from ripepp.utils.scared_utils import compute_overlap, load_scene_points

log = utils.get_pylogger(__name__)


class SCARED_Val(Dataset):
    """
    SCARED validation dataset with overlap-filtered random pairs.

    Samples random frame pairs from the same keyframe, filtered by minimum overlap ratio.
    Provides camera intrinsics and poses for benchmark evaluation.

    Directory structure:
        {root}/val/dataset_*/keyframe_*/data/
            - frame_data/frame_data{frame_id}.json
            - scene_points/scene_points{frame_id}.tiff
            - rgb_frames_left/frame_{frame_id}.png

    Args:
        root: Path to dataset root directory
        num_pairs: Number of pairs to sample (default: 200)
        min_overlap: Minimum overlap ratio for pair filtering (default: 0.4)
        seed: Random seed for reproducibility (default: 42)
        transforms: Optional image transformations to apply
    """

    def __init__(
        self,
        root: str,
        num_pairs: int = 200,
        min_overlap: float = 0.4,
        seed: int = 42,
        transforms: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.num_pairs = num_pairs
        self.min_overlap = min_overlap
        self.seed = seed

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset not found at {self.root}")

        if transforms is None:
            self.transforms = Compose([])
        else:
            self.transforms = transforms

        self.val_path = self.root / "val"
        if not self.val_path.exists():
            raise FileNotFoundError(f"Validation directory not found at {self.val_path}")

        # Discover frames and sample pairs
        self.frames_by_keyframe = self._discover_frames()
        self.pairs = self._sample_pairs_with_overlap()

        log.info(
            f"Initialized SCARED_Val dataset with {len(self.pairs)} pairs "
            f"(num_pairs={num_pairs}, min_overlap={min_overlap}, seed={seed})"
        )

    def _discover_frames(self) -> Dict[Tuple[str, str], List[str]]:
        """
        Discover all frames in the validation directory.

        Returns:
            Dict mapping (dataset_name, keyframe_name) to list of frame IDs
        """
        frames_by_keyframe: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        dataset_dirs = sorted(self.val_path.glob("dataset_*"))
        if len(dataset_dirs) == 0:
            raise FileNotFoundError(f"No dataset directories found in {self.val_path}")

        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name
            keyframe_dirs = sorted(dataset_dir.glob("keyframe_*"))

            for keyframe_dir in keyframe_dirs:
                keyframe_name = keyframe_dir.name
                frames_dir = keyframe_dir / "data" / "rgb_frames_left"

                if not frames_dir.exists():
                    log.warning(f"Frames directory not found at {frames_dir}, skipping")
                    continue

                # Get all frame files and extract frame IDs
                frame_files = sorted(frames_dir.glob("frame_*.png"))
                for frame_file in frame_files:
                    # Extract frame ID from filename (e.g., "frame_000123.png" -> "000123")
                    frame_id = frame_file.stem.replace("frame_", "")
                    frames_by_keyframe[(dataset_name, keyframe_name)].append(frame_id)

                if len(frames_by_keyframe[(dataset_name, keyframe_name)]) > 0:
                    log.info(
                        f"  {dataset_name}/{keyframe_name}: "
                        f"{len(frames_by_keyframe[(dataset_name, keyframe_name)])} frames"
                    )

        total_frames = sum(len(frames) for frames in frames_by_keyframe.values())
        log.info(f"Found {total_frames} frames across {len(frames_by_keyframe)} keyframes")

        return frames_by_keyframe

    def _load_frame_metadata(self, dataset: str, keyframe: str, frame_id: str) -> Dict[str, Any]:
        """
        Load camera metadata from JSON file for a frame.

        Args:
            dataset: Dataset name (e.g., "dataset_8")
            keyframe: Keyframe name (e.g., "keyframe_1")
            frame_id: Frame ID (e.g., "000123")

        Returns:
            Dict with camera parameters: fx, fy, cx, cy, R, T
        """
        json_path = self.val_path / dataset / keyframe / "data" / "frame_data" / f"frame_data{frame_id}.json"

        if not json_path.exists():
            raise FileNotFoundError(f"Frame metadata not found: {json_path}")

        with open(json_path) as f:
            data = json.load(f)

        # Extract camera intrinsics from KL (left camera)
        KL = np.array(data["camera-calibration"]["KL"])
        fx, fy = KL[0, 0], KL[1, 1]
        cx, cy = KL[0, 2], KL[1, 2]

        # Extract camera pose (4x4 transformation matrix)
        pose = np.array(data["camera-pose"])
        R = pose[:3, :3]
        T = pose[:3, 3]

        return {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "R": R,
            "T": T,
            "K": KL,
        }

    def _compute_pair_overlap(
        self, dataset: str, keyframe: str, frame_id_1: str, frame_id_2: str
    ) -> Tuple[float, Dict, Dict]:
        """
        Compute overlap ratio between two frames.

        Args:
            dataset: Dataset name
            keyframe: Keyframe name
            frame_id_1: First frame ID
            frame_id_2: Second frame ID

        Returns:
            Tuple of (overlap_ratio, metadata1, metadata2)
        """
        meta1 = self._load_frame_metadata(dataset, keyframe, frame_id_1)
        meta2 = self._load_frame_metadata(dataset, keyframe, frame_id_2)

        # Load scene points for first frame
        pts1 = load_scene_points(self.val_path, dataset, keyframe, frame_id_1)

        # Compute overlap using scared_utils
        K = np.array([[meta1["fx"], 0, meta1["cx"]], [0, meta1["fy"], meta1["cy"]], [0, 0, 1]])

        overlap = compute_overlap(pts1, meta1, meta2, K)

        return overlap, meta1, meta2

    def _sample_pairs_with_overlap(self) -> List[Dict[str, Any]]:
        """
        Sample random pairs with overlap filtering.

        Returns:
            List of pair metadata dicts
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        pairs: List[Dict[str, Any]] = []
        all_keyframes = list(self.frames_by_keyframe.keys())

        # Track attempts to avoid infinite loop
        max_attempts = self.num_pairs * 100
        attempts = 0

        while len(pairs) < self.num_pairs and attempts < max_attempts:
            attempts += 1

            # Randomly select a keyframe
            dataset, keyframe = random.choice(all_keyframes)
            frames = self.frames_by_keyframe[(dataset, keyframe)]

            if len(frames) < 2:
                continue

            # Randomly select two different frames
            frame_id_1, frame_id_2 = random.sample(frames, 2)

            try:
                overlap, meta1, meta2 = self._compute_pair_overlap(dataset, keyframe, frame_id_1, frame_id_2)
            except Exception as e:
                log.warning(f"Error computing overlap for {dataset}/{keyframe}/{frame_id_1}-{frame_id_2}: {e}")
                continue

            if overlap >= self.min_overlap:
                pairs.append(
                    {
                        "dataset": dataset,
                        "keyframe": keyframe,
                        "frame_id_1": frame_id_1,
                        "frame_id_2": frame_id_2,
                        "overlap": overlap,
                        "meta1": meta1,
                        "meta2": meta2,
                    }
                )

                if len(pairs) % 50 == 0:
                    log.info(f"Sampled {len(pairs)}/{self.num_pairs} pairs (attempts: {attempts})")

        if len(pairs) < self.num_pairs:
            log.warning(
                f"Could only sample {len(pairs)} pairs with overlap >= {self.min_overlap} (requested {self.num_pairs})"
            )

        return pairs

    def _create_camera(self, meta: Dict[str, Any]) -> Camera:
        """Create a Camera object from frame metadata."""
        K = torch.tensor(
            [[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]],
            dtype=torch.float32,
        )
        R = torch.tensor(meta["R"], dtype=torch.float32)
        T = torch.tensor(meta["T"], dtype=torch.float32)
        return Camera(K, R, T)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.pairs[idx]

        dataset = pair["dataset"]
        keyframe = pair["keyframe"]
        frame_id_1 = pair["frame_id_1"]
        frame_id_2 = pair["frame_id_2"]
        meta1 = pair["meta1"]
        meta2 = pair["meta2"]

        # Load images
        src_path = self.val_path / dataset / keyframe / "data" / "rgb_frames_left" / f"frame_{frame_id_1}.png"
        trg_path = self.val_path / dataset / keyframe / "data" / "rgb_frames_left" / f"frame_{frame_id_2}.png"

        src_img = read_image(str(src_path)) / 255.0
        trg_img = read_image(str(trg_path)) / 255.0

        _, H_src, W_src = src_img.shape
        _, H_trg, W_trg = trg_img.shape

        # Create masks (all ones, no padding initially)
        src_mask = torch.ones((1, H_src, W_src), dtype=torch.uint8)
        trg_mask = torch.ones((1, H_trg, W_trg), dtype=torch.uint8)

        # Store original sizes before transforms
        orig_size_src = (H_src, W_src)
        orig_size_trg = (H_trg, W_trg)

        # Apply transforms
        if self.transforms:
            src_img, src_mask = self.transforms(src_img, src_mask)
            trg_img, trg_mask = self.transforms(trg_img, trg_mask)

        # Create Camera objects
        src_camera = self._create_camera(meta1)
        trg_camera = self._create_camera(meta2)

        # Compute relative pose (source to target)
        R1 = torch.tensor(meta1["R"], dtype=torch.float32)
        T1 = torch.tensor(meta1["T"], dtype=torch.float32)
        R2 = torch.tensor(meta2["R"], dtype=torch.float32)
        T2 = torch.tensor(meta2["T"], dtype=torch.float32)

        # Relative rotation and translation: R_rel = R2 @ R1.T, T_rel = T2 - R_rel @ T1
        s2t_R = R2 @ R1.T
        s2t_T = T2 - s2t_R @ T1

        # Compute fundamental matrix
        F = cameras2F(src_camera, trg_camera)

        sample = {
            "src_image": src_img,
            "trg_image": trg_img,
            "src_mask": src_mask.to(torch.bool),
            "trg_mask": trg_mask.to(torch.bool),
            "orig_size_src": orig_size_src,
            "orig_size_trg": orig_size_trg,
            "src_camera": src_camera,
            "trg_camera": trg_camera,
            "s2t_R": s2t_R,
            "s2t_T": s2t_T,
            "F": F,
            "src_path": str(src_path),
            "trg_path": str(trg_path),
            "overlap": pair["overlap"],
        }

        return sample
