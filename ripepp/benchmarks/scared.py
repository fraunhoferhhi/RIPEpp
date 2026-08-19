"""
SCARED Benchmark for evaluating keypoint detection on surgical endoscopy data.

Similar to IMW_2020_Benchmark but uses the SCARED validation dataset with
overlap-filtered pairs.
"""

import os
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import poselib
import torch
from tqdm import tqdm

from ripepp import utils
from ripepp.data.data_transforms import Compose, DivisibleBy, Normalize
from ripepp.data.datasets.scared_val import SCARED_Val
from ripepp.utils.pose_error import AUCMetric, relative_pose_error
from ripepp.utils.utils import (
    cv2_matches_from_kornia,
    cv_resize_and_pad_to_shape,
    to_cv_kpts,
)

log = utils.get_pylogger(__name__)


class SCARED_Benchmark:
    """
    Benchmark for evaluating keypoint detection on SCARED surgical endoscopy data.

    Uses the SCARED_Val dataset with overlap-filtered random pairs.
    Evaluates relative pose estimation accuracy using AUC metrics.

    Args:
        num_pairs: Number of pairs to evaluate (default: 200)
        min_overlap: Minimum overlap ratio for pair filtering (default: 0.4)
        seed: Random seed for reproducibility (default: 42)
        conf_inference: Inference configuration for the model
        transforms: Optional image transformations
        matcher: Descriptor matcher function
        output_dir: Optional output directory for saving results
    """

    def __init__(
        self,
        num_pairs: int = 200,
        min_overlap: float = 0.4,
        seed: int = 42,
        conf_inference=None,
        transforms=None,
        matcher=None,
        output_dir: Optional[str] = "",
        dry_run: bool = False,
    ):
        data_dir = os.getenv("DATA_DIR")
        if data_dir is None:
            raise ValueError("Environment variable DATA_DIR is not set.")
        self.root_path = Path(data_dir) / "SCARED"

        if transforms is None:
            transforms = Compose(
                [
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    DivisibleBy(1, antialias=True),
                ]
            )

        if dry_run:
            num_pairs = 10
            log.info("Dry run enabled. Reducing number of pairs to 10.")

        self.data = SCARED_Val(
            str(self.root_path),
            num_pairs=num_pairs,
            min_overlap=min_overlap,
            seed=seed,
            transforms=transforms,
        )
        self.data_transforms = transforms
        self.results = None
        self.predictions = []
        self.conf_inference = conf_inference
        self.output_dir = output_dir
        self.matcher = matcher

    def evaluate_sample(self, model, sample, dev):
        """
        Evaluate a single sample.

        Args:
            model: The keypoint detection model with detectAndCompute method
            sample: A sample from SCARED_Val dataset
            dev: Device to run on

        Returns:
            Dict with evaluation results
        """
        img_1 = sample["src_image"].unsqueeze(0).to(dev)
        img_2 = sample["trg_image"].unsqueeze(0).to(dev)

        scale_h_1, scale_w_1 = (
            sample["orig_size_src"][0] / img_1.shape[2],
            sample["orig_size_src"][1] / img_1.shape[3],
        )
        scale_h_2, scale_w_2 = (
            sample["orig_size_trg"][0] / img_2.shape[2],
            sample["orig_size_trg"][1] / img_2.shape[3],
        )

        M = None
        info = {}
        kpts_1, desc_1, score_1 = None, None, None
        kpts_2, desc_2, score_2 = None, None, None
        match_dists, match_idxs = None, None

        try:
            kpts_1, desc_1, score_1 = model.detectAndCompute(img_1, **self.conf_inference)
            kpts_2, desc_2, score_2 = model.detectAndCompute(img_2, **self.conf_inference)

            if kpts_1.dim() == 3:
                assert kpts_1.shape[0] == 1 and kpts_2.shape[0] == 1, "Batch size must be 1"

                kpts_1, desc_1, score_1 = (
                    kpts_1.squeeze(0),
                    desc_1.squeeze(0),
                    score_1.squeeze(0),
                )
                kpts_2, desc_2, score_2 = (
                    kpts_2.squeeze(0),
                    desc_2.squeeze(0),
                    score_2.squeeze(0),
                )

            scale_1 = torch.tensor([scale_w_1, scale_h_1], dtype=torch.float).to(dev)
            scale_2 = torch.tensor([scale_w_2, scale_h_2], dtype=torch.float).to(dev)

            kpts_1 = kpts_1 * scale_1
            kpts_2 = kpts_2 * scale_2

            match_dists, match_idxs = self.matcher(desc_1, desc_2)

            matched_pts_1 = kpts_1[match_idxs[:, 0]]
            matched_pts_2 = kpts_2[match_idxs[:, 1]]

            camera_1 = sample["src_camera"]
            camera_2 = sample["trg_camera"]

            M, info = poselib.estimate_relative_pose(
                matched_pts_1.cpu().numpy(),
                matched_pts_2.cpu().numpy(),
                camera_1.to_cameradict(),
                camera_2.to_cameradict(),
                {
                    "max_epipolar_error": 0.5,
                },
                {},
            )
        except RuntimeError as e:
            if "No keypoints detected" in str(e):
                pass
            else:
                raise e

        success = M is not None
        if success:
            M = {
                "R": torch.tensor(M.R, dtype=torch.float),
                "t": torch.tensor(M.t, dtype=torch.float),
            }
            inl = info["inliers"]
        else:
            M = {
                "R": torch.eye(3, dtype=torch.float),
                "t": torch.zeros((3), dtype=torch.float),
            }
            inl = np.zeros((0,)).astype(bool)

        t_err, r_err = relative_pose_error(sample["s2t_R"].cpu(), sample["s2t_T"].cpu(), M["R"], M["t"])

        rel_pose_error = max(t_err.item(), r_err.item()) if success else np.inf
        ransac_inl = np.sum(inl)
        ransac_inl_ratio = np.mean(inl) if len(inl) > 0 else 0.0

        if success:
            assert match_dists is not None and match_idxs is not None, "Matches must be computed"
            cv_keypoints_src = to_cv_kpts(kpts_1, score_1)
            cv_keypoints_trg = to_cv_kpts(kpts_2, score_2)
            cv_matches = cv2_matches_from_kornia(match_dists, match_idxs)
            cv_mask = [int(m) for m in inl]
        else:
            cv_keypoints_src, cv_keypoints_trg = [], []
            cv_matches, cv_mask = [], []

        estimation = {
            "success": success,
            "M_0to1": M,
            "inliers": torch.tensor(inl).to(img_1),
            "rel_pose_error": rel_pose_error,
            "ransac_inl": ransac_inl,
            "ransac_inl_ratio": ransac_inl_ratio,
            "path_src_image": sample["src_path"],
            "path_trg_image": sample["trg_path"],
            "cv_keypoints_src": cv_keypoints_src,
            "cv_keypoints_trg": cv_keypoints_trg,
            "cv_matches": cv_matches,
            "cv_mask": cv_mask,
            "overlap": sample.get("overlap", None),
        }

        return estimation

    def evaluate(self, model, dev, progress_bar=False):
        """
        Evaluate the model on all samples.

        Args:
            model: The keypoint detection model
            dev: Device to run on
            progress_bar: Whether to show progress bar
        """
        model.eval()

        # reset results
        self.results = []

        for idx in tqdm(range(len(self.data)), disable=not progress_bar):
            sample = self.data[idx]
            self.results.append(self.evaluate_sample(model, sample, dev))

    def get_auc(self, threshold=5):
        """
        Get AUC at a specific threshold.

        Args:
            threshold: Error threshold in degrees (default: 5)

        Returns:
            AUC value at the given threshold
        """
        if self.results is None or len(self.results) == 0:
            raise ValueError("No results to log. Run evaluate first.")

        summary_results = self.calc_auc()

        return summary_results[f"scared_rel_pose_error@{threshold}°"]

    def calc_auc(self, auc_thresholds=None):
        """
        Calculate AUC metrics at multiple thresholds.

        Args:
            auc_thresholds: List of error thresholds in degrees (default: [5, 10, 20])

        Returns:
            Dict with AUC values at each threshold
        """
        if auc_thresholds is None:
            auc_thresholds = [5, 10, 20]

        if self.results is None or len(self.results) == 0:
            raise ValueError("No results to calculate auc. Run evaluate first.")

        rel_pose_errors = [r["rel_pose_error"] for r in self.results]

        pose_aucs = AUCMetric(auc_thresholds, rel_pose_errors).compute()
        assert isinstance(pose_aucs, list) and len(pose_aucs) == len(auc_thresholds)

        summary = {}
        for i, ath in enumerate(auc_thresholds):
            summary[f"scared_rel_pose_error@{ath}°"] = pose_aucs[i]

        return summary

    def calc_metric(self, key):
        """
        Calculate mean of a specific metric across all results.

        Args:
            key: Metric key to calculate mean for

        Returns:
            Mean value of the metric
        """
        if self.results is None or len(self.results) == 0:
            raise ValueError("No results to calculate metric. Run evaluate first.")

        values = [r[key] for r in self.results]
        values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]

        if len(values) == 0:
            return np.nan

        return np.mean(values)

    def log_results(self, logger=None, step=None):
        """
        Log results to a logger (e.g., WandB).

        Args:
            logger: Logger instance (e.g., WandB)
            step: Training step for logging
        """
        if self.results is None or len(self.results) == 0:
            raise ValueError("No results to log. Run evaluate first.")

        summary_results = self.calc_auc()
        summary_results["scared_ransac_inl_ratio"] = self.calc_metric("ransac_inl_ratio")

        if logger is not None:
            logger.log(summary_results, step=step)
        else:
            log.warning("No logger provided. Printing results instead.")
            print(self.calc_auc())

    def print_results(self):
        """Print results to console."""
        if self.results is None or len(self.results) == 0:
            raise ValueError("No results to print. Run evaluate first.")

        print(self.calc_auc())
        print(f"ransac_inl_ratio: {self.calc_metric('ransac_inl_ratio')}")

    def plot_results(self, num_samples=10, logger=None, step=None, use_plt=False):
        """
        Plot match visualizations for sample pairs.

        Args:
            num_samples: Number of samples to plot
            logger: Logger instance for WandB logging
            step: Training step for logging
            use_plt: If True, save to files instead of logging
        """
        if self.results is None or len(self.results) == 0:
            raise ValueError("No results to plot. Run evaluate first.")

        plot_data = []

        for result in self.results[:num_samples]:
            img1 = cv2.imread(result["path_src_image"])
            img2 = cv2.imread(result["path_trg_image"])

            # from BGR to RGB
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            plt_matches = cv2.drawMatches(
                img1,
                result["cv_keypoints_src"],
                img2,
                result["cv_keypoints_trg"],
                result["cv_matches"],
                None,
                matchColor=None,
                matchesMask=result["cv_mask"],
                flags=cv2.DrawMatchesFlags_DEFAULT,
            )
            file_name = (
                Path(result["path_src_image"]).parent.parent.parent.parent.name
                + "_"
                + Path(result["path_src_image"]).stem
                + "_"
                + Path(result["path_trg_image"]).stem
                + ".png"
            )
            # print rel_pose_error on image
            overlap_str = f" overlap: {result['overlap']:.2f}" if result.get("overlap") else ""
            plt_matches = cv2.putText(
                plt_matches,
                f"rel_pose_error: {result['rel_pose_error']:.2f} num_inliers: {result['ransac_inl']} "
                f"inl_ratio: {result['ransac_inl_ratio']:.2f} num_matches: {len(result['cv_matches'])} "
                f"num_keypoints: {len(result['cv_keypoints_src'])}/{len(result['cv_keypoints_trg'])}{overlap_str}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_8,
            )

            plot_data.append({"file_name": file_name, "image": plt_matches})

        if use_plt:
            log.info("No logger provided. Using plt to plot results.")
            for image in plot_data:
                plt.imsave(
                    image["file_name"],
                    cv_resize_and_pad_to_shape(image["image"], (1024, 2560)),
                )
                plt.close()
        else:
            import wandb

            # check logger
            if logger is None:
                raise ValueError("No logger provided for plotting results.")

            log.info(f"Logging SCARED images to wandb with step={step}")
            logger.log(
                {
                    "scared_examples": [
                        wandb.Image(cv_resize_and_pad_to_shape(image["image"], (1024, 2560))) for image in plot_data
                    ]
                },
                step=step,
            )
