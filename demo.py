import argparse

import cv2
import kornia.feature as KF
import kornia.geometry as KG
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.io import decode_image

from ripepp import load_model_from_checkpoint, resolve_variant_checkpoint
from ripepp.utils.utils import cv2_matches_from_kornia, resize_image, to_cv_kpts

parser = argparse.ArgumentParser(description="RIPE++ matching demo.")
parser.add_argument(
    "--weights",
    choices=["default", "aachen_day_night", "scared"],
    default="default",
    help="Which trained weight variant to use (downloaded if not present locally).",
)
args = parser.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_path = resolve_variant_checkpoint(args.weights)
model = load_model_from_checkpoint(ckpt_path).to(dev)
model.eval()

image1 = resize_image(decode_image("assets/all_souls_000013.jpg").float().to(dev) / 255.0)
image2 = resize_image(decode_image("assets/all_souls_000055.jpg").float().to(dev) / 255.0)

kpts_1, desc_1, score_1 = model.detectAndCompute(image1, threshold=0.5, top_k=2048)
kpts_2, desc_2, score_2 = model.detectAndCompute(image2, threshold=0.5, top_k=2048)

kpts_1, kpts_2 = kpts_1.squeeze(0), kpts_2.squeeze(0)
desc_1, desc_2 = desc_1.squeeze(0), desc_2.squeeze(0)
score_1, score_2 = score_1.squeeze(0), score_2.squeeze(0)

matcher = KF.DescriptorMatcher("mnn")  # threshold is not used with mnn
match_dists, match_idxs = matcher(desc_1, desc_2)

matched_pts_1 = kpts_1[match_idxs[:, 0]]
matched_pts_2 = kpts_2[match_idxs[:, 1]]

H, mask = KG.ransac.RANSAC(model_type="fundamental", inl_th=1.0)(matched_pts_1, matched_pts_2)
matchesMask = mask.int().ravel().tolist()

result_ransac = cv2.drawMatches(
    (image1.cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8),
    to_cv_kpts(kpts_1, score_1),
    (image2.cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8),
    to_cv_kpts(kpts_2, score_2),
    cv2_matches_from_kornia(match_dists, match_idxs),
    None,
    matchColor=(0, 255, 0),
    matchesMask=matchesMask,
    # matchesMask=None, # without RANSAC filtering
    singlePointColor=(0, 0, 255),
    flags=cv2.DrawMatchesFlags_DEFAULT,
)

plt.imshow(result_ransac)
plt.axis("off")
plt.tight_layout()

plt.show()
plt.savefig("result_ransac.png")
