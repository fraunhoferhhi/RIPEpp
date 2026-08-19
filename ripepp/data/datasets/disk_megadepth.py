import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torchvision.io import read_image

from ripepp import utils
from ripepp.data.data_transforms import Compose
from ripepp.utils.data_utils import TorchSerializedList

from .base_dataset import BaseDataset

log = utils.get_pylogger(__name__)


class DISK_Megadepth(BaseDataset):
    def __init__(
        self,
        root: str,
        max_scene_size: int,
        stage: str = "train",
        # condition: str = "rain",
        transforms: Optional[Callable] = None,
        positive_only: bool = False,
    ) -> None:
        super().__init__(positive_only=positive_only)

        self.root = root
        self.stage = stage
        self.transforms = transforms
        self.epoch = 0
        self.max_scene_size = max_scene_size

        if isinstance(self.root, str):
            self.root = Path(self.root)

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset not found at {self.root}")

        if transforms is None:
            self.transforms = Compose([])
        else:
            self.transforms = transforms

        if self.stage not in ["train"]:
            raise RuntimeError("Unknown option " + self.stage + " as training stage variable. Valid options: 'train'")

        self.sample()

        if positive_only:
            log.warning("Using only positive pairs!")

    def sample(self):
        json_path = self.root / "megadepth" / "dataset.json"
        with open(json_path) as json_file:
            json_data = json.load(json_file)

        scenes_base = self.root / "megadepth" / "scenes"

        scenes = []
        scene_rel_dirs = []

        for scene_key in json_data:
            scene = Scene(self.root / "megadepth", json_data[scene_key], self.max_scene_size)
            scenes.append(scene)
            # Precompute relative directory string once per scene (avoids Path ops per sample)
            rel_dir = str((self.root / "megadepth" / json_data[scene_key]["image_path"]).relative_to(scenes_base))
            scene_rel_dirs.append(rel_dir)

        tuples_per_scene = [len(scene) for scene in scenes]

        # Precompute cumulative sum for O(log n) scene lookup via searchsorted
        cumsum = np.cumsum(tuples_per_scene)

        list_image_path_1 = []
        list_image_path_2 = []
        list_label = []

        n_samples = sum(tuples_per_scene) if self.positive_only else 2 * sum(tuples_per_scene)

        for idx in range(n_samples):
            positive_sample = idx % 2 == 0 or self.positive_only
            if not self.positive_only:
                idx = idx // 2

            label = positive_sample

            # O(log n) scene lookup instead of O(n) accumulate scan
            i_scene = int(np.searchsorted(cumsum, idx, side="right"))
            i_image = idx - (int(cumsum[i_scene - 1]) if i_scene > 0 else 0)

            if positive_sample:
                name1, name2 = scenes[i_scene].get_image_names(i_image)
                rel_dir = scene_rel_dirs[i_scene]
                path_image1 = f"{rel_dir}/{name1}"
                path_image2 = f"{rel_dir}/{name2}"
            else:
                name1, _ = scenes[i_scene].get_image_names(i_image)
                rel_dir = scene_rel_dirs[i_scene]
                path_image1 = f"{rel_dir}/{name1}"

                scene_id_2, image_id_2 = self._get_other_random_scene_and_image_id(scenes, i_scene)
                name2, _ = scenes[scene_id_2].get_image_names(image_id_2)
                rel_dir_2 = scene_rel_dirs[scene_id_2]
                path_image2 = f"{rel_dir_2}/{name2}"

            list_image_path_1.append(path_image1)
            list_image_path_2.append(path_image2)
            list_label.append(label)

        self.image_path_1 = TorchSerializedList(list_image_path_1)
        self.image_path_2 = TorchSerializedList(list_image_path_2)
        self.label = TorchSerializedList(list_label)

        log.info(f"Sampled {len(self.label)} pairs from {len(scenes)} scenes")

    def resample(self):
        """Resample tuples from scenes with new random seed.

        This allows for fresh tuple sampling each epoch during training.
        """
        self.epoch += 1

        self.sample()  # Resample tuples in each scene

        log.info(f"Resampled DISK_Megadepth for epoch {self.epoch}")

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample: Any = {}

        src_path = str(self.root / "megadepth/scenes" / self.image_path_1[idx])
        trg_path = str(self.root / "megadepth/scenes" / self.image_path_2[idx])
        positive_sample = bool(self.label[idx])

        sample["label"] = positive_sample
        sample["src_path"] = src_path
        sample["trg_path"] = trg_path

        src_img = read_image(sample["src_path"]) / 255.0
        trg_img = read_image(sample["trg_path"]) / 255.0

        _, H_src, W_src = src_img.shape
        _, H_trg, W_trg = trg_img.shape

        src_mask = torch.ones((1, H_src, W_src), dtype=torch.uint8)
        trg_mask = torch.ones((1, H_trg, W_trg), dtype=torch.uint8)

        if self.transforms:
            src_img, src_mask = self.transforms(src_img, src_mask)
            trg_img, trg_mask = self.transforms(trg_img, trg_mask)

        sample["src_image"] = src_img
        sample["trg_image"] = trg_img
        sample["src_mask"] = src_mask.to(torch.bool)
        sample["trg_mask"] = trg_mask.to(torch.bool)

        return sample

    def _get_other_random_scene_and_image_id(self, scenes, scene_id_to_exclude: int) -> Tuple[int, int]:
        n = len(scenes)
        # Equivalent to: list(range(n)).remove(excluded); random.choice(remaining)
        # random.choice calls _randbelow(len(seq)), same as randrange(len(seq))
        idx_scene = random.randrange(n - 1)
        if idx_scene >= scene_id_to_exclude:
            idx_scene += 1
        idx_image = random.randint(0, len(scenes[idx_scene]) - 1)

        return idx_scene, idx_image


class Scene:
    def __init__(self, root_path, scene_data: Dict[str, Any], max_size_scene) -> None:
        self.root_path = root_path
        self.image_path = Path(scene_data["image_path"])
        self.image_names = np.array(scene_data["images"])

        # randomly sample tuples, store as numpy to avoid COW in forked dataloader workers
        if max_size_scene > 0:
            self.tuples = np.array(
                random.sample(scene_data["tuples"], min(max_size_scene, len(scene_data["tuples"]))),
                dtype=np.int32,
            )

    def __len__(self) -> int:
        return len(self.tuples)

    def get_image_names(self, idx: int) -> Tuple[str, str]:
        """Return (name1, name2) as strings — same random.sample logic as __getitem__."""
        idx_1, idx_2 = random.sample([0, 1, 2], 2)

        name1 = self.image_names[self.tuples[idx][idx_1]]
        name2 = self.image_names[self.tuples[idx][idx_2]]

        return name1, name2

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        idx_1, idx_2 = random.sample([0, 1, 2], 2)

        idx_1 = self.tuples[idx][idx_1]
        idx_2 = self.tuples[idx][idx_2]

        path_image_1 = self.root_path / self.image_path / self.image_names[idx_1]
        path_image_2 = self.root_path / self.image_path / self.image_names[idx_2]

        return path_image_1, path_image_2
