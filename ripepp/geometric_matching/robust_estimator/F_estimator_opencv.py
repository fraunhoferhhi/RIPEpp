import cv2
import torch

from ripepp import utils

log = utils.get_pylogger(__name__)


class F_EstimatorOpenCV:
    def __init__(self):
        pass

    def __call__(self, pts0, pts1, inl_th, success_prob=0.99, max_iterations=10000):
        try:
            F, mask = cv2.findFundamentalMat(
                pts0.cpu().numpy(),
                pts1.cpu().numpy(),
                cv2.USAC_MAGSAC,  # decision based on: https://opencv.org/blog/evaluating-opencvs-new-ransacs/
                inl_th,
                success_prob,
                max_iterations,
            )
        except Exception as e:
            log.warning(f"cv2.findFundamentalMat failed with error: {e}")
            F, mask = None, None

        success = F is not None and mask is not None
        if success:
            inliers = torch.tensor(mask, dtype=torch.bool, device=pts0.device).squeeze()
        else:
            inliers = torch.zeros(pts0.shape[0], dtype=torch.bool, device=pts0.device)

        return F, inliers
