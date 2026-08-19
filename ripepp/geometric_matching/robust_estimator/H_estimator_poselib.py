import poselib
import torch


class H_EstimatorPoselib:
    def __init__(self):
        pass

    def __call__(self, pts0, pts1, inl_th):
        H, info = poselib.estimate_homography(
            pts0.cpu().numpy(),
            pts1.cpu().numpy(),
            {
                "max_reproj_error": inl_th,
            },
        )

        success = H is not None
        if success:
            inliers = info.pop("inliers")
            inliers = torch.tensor(inliers, dtype=torch.bool, device=pts0.device)
        else:
            inliers = torch.zeros(pts0.shape[0], dtype=torch.bool, device=pts0.device)

        return H, inliers
