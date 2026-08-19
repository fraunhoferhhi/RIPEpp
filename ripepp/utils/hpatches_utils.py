import os
from pathlib import Path

import cv2
import kornia.feature as KF
import numpy as np
import torch
from matplotlib import pyplot as plt

from ripepp import utils
from ripepp.data.data_transforms import Compose, Normalize
from ripepp.data.datasets.hpatches import HPatches
from ripepp.utils.utils import cv2_matches_from_kornia, to_cv_kpts

log = utils.get_pylogger(__name__)


def eval_hpatches(model, dev):
    log.info("Evaluating on HPatches dataset")

    data_dir = os.getenv("DATA_DIR")
    if data_dir is None:
        raise ValueError("Environment variable DATA_DIR is not set")
    data_path = Path(data_dir) / "hpatches"

    transforms = Compose(
        [
            Normalize(
                mean=np.array([0.485, 0.456, 0.406]),
                std=np.array([0.229, 0.224, 0.225]),
            )
        ]
    )

    ds = HPatches(data_path, "all", ["viewpoint", "illumination"], transforms, True)

    extractor = model

    thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    i_v = 0
    err_v = np.zeros(10)
    i_i = 0
    err_i = np.zeros(10)
    n_samples_skipped = 0

    for i in range(len(ds)):
        sample = ds[i]
        if "/v_" in sample["src_path"]:
            try:
                err_v += evaluate(extractor, sample, dev, thresholds=thresholds)
            # catch RuntimeError: No keypoints detected.
            except Exception as e:
                if e.args[0] == "No keypoints detected.":
                    n_samples_skipped += 1
                    continue
                else:
                    raise e
            i_v += 1
        else:
            try:
                err_i += evaluate(extractor, sample, dev, thresholds=thresholds)
            # catch RuntimeError: No keypoints detected.
            except Exception as e:
                if e.args[0] == "No keypoints detected.":
                    n_samples_skipped += 1
                    continue
                else:
                    raise e
            i_i += 1

    log.info("Evaluation on HPatches dataset completed")

    return err_v / i_v, err_i / i_i, n_samples_skipped


def evaluate(model, sample, dev, thresholds=None):
    if thresholds is None:
        thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    model.net.eval()

    err = []

    src_image = sample["src_image"].unsqueeze(0).to(dev)
    trg_image = sample["trg_image"].unsqueeze(0).to(dev)
    h = sample["homography"].to(dev)

    with torch.no_grad():
        kps_src, descs_src, scores_src = model.detectAndCompute(src_image, threshold=25.0, top_k=2048)
        kps_trg, descs_trg, scores_trg = model.detectAndCompute(trg_image, threshold=25.0, top_k=2048)

        kps_src = kps_src.float()
        kps_trg = kps_trg.float()

        # Match using vanilla opencv matcher
        matcher = KF.DescriptorMatcher("mnn", 0.8)  # threshold is not used with mnn
        dists, indexes = matcher(descs_src, descs_trg)

        if indexes.shape[0] == 0:
            dist = torch.tensor([float("inf")])
        else:
            matched_src_pts = kps_src[indexes[:, 0]]
            matched_trg_pts = kps_trg[indexes[:, 1]]

            # homogeneous coordinates
            matched_src_pts = torch.cat(
                [matched_src_pts, torch.ones(matched_src_pts.shape[0], 1, device=dev)],
                dim=1,
            )

            matched_src_pts_proj = torch.matmul(h, matched_src_pts.t()).t()
            matched_src_pts_proj = matched_src_pts_proj[:, :2] / matched_src_pts_proj[:, 2:]

            dist = torch.norm(matched_src_pts_proj - matched_trg_pts, dim=1)

            # DEBUG
            # for thr in thresholds:
            #     draw_match(src_image, trg_image, kps_src, kps_trg, scores_src, scores_trg, indexes, dists, dist, thr)
            # raise RuntimeError("Das war's sprach Lars.")
            # DEBUG

        for thr in thresholds:
            err.append(torch.mean((dist <= thr).float()).item())

    return np.array(err)


def draw_match(
    src,
    trg,
    kps1,
    kps2,
    scores1,
    scores2,
    indexes_matches,
    dists_matches,
    dist_proj,
    thr,
):
    kps1 = to_cv_kpts(kps1, scores1)
    kps2 = to_cv_kpts(kps2, scores2)

    matches = cv2_matches_from_kornia(dists_matches, indexes_matches)

    org_src_image = src.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    org_trg_image = trg.detach().cpu().numpy().squeeze().transpose(1, 2, 0)

    # denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    org_src_image = (((org_src_image * std) + mean) * 255).astype(np.uint8)
    org_trg_image = (((org_trg_image * std) + mean) * 255).astype(np.uint8)

    # convert from RGB to BGR
    org_src_image = org_src_image[..., ::-1]
    org_trg_image = org_trg_image[..., ::-1]

    mask = dist_proj < thr
    matchesMask = mask.int().ravel().tolist()

    result_match = cv2.drawMatches(
        org_src_image,
        kps1,
        org_trg_image,
        kps2,
        matches,
        None,
        matchColor=(0, 255, 0),
        matchesMask=matchesMask,
        singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_DEFAULT,
    )

    plt.figure(figsize=(15, 10))
    plt.imshow(result_match[..., ::-1])
    plt.tight_layout()
    plt.savefig(f"output_thr_{thr}.png")
    plt.close()
