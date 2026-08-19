import torch
import torch.nn as nn

from ripepp import utils

from .base_sampler import BaseSampler

log = utils.get_pylogger(__name__)


class NMSKeypointSampler(BaseSampler):
    def __init__(
        self,
        min_num_keypoints=4096,
        softmax_temperature=10.0,
        threshold=3.188775510204082e-6,
        top_k=2048,
    ):
        super().__init__()
        self.min_num_keypoints = min_num_keypoints
        self.softmax_temperature = softmax_temperature
        self.threshold = threshold  # 3.188775510204082e-6 -> approx 1/560^2
        self.top_k = top_k
        log.info(
            f"Using SequentialKeypointSampler with min_num_keypoints={min_num_keypoints} and softmax_temperature={softmax_temperature}"
        )

    def do_inference(self, x):
        B, C, H, W = x.shape

        assert B == 1, "Batch size should be 1"

        kpts = self.NMS(x[0], self.threshold)

        if self.top_k is not None:
            scores = x[0].squeeze(0)[kpts[:, 1].long(), kpts[:, 0].long()]
            sorted_idx = torch.argsort(-scores)
            kpts = kpts[sorted_idx[: self.top_k]]

        return kpts  # N, 2

    def forward(self, x, mask_padding=None):
        B, C, H, W = x.shape

        assert C == 1, "Input to SequentialKeypointSampler should have one channel"

        x = x.squeeze(1)  # B, H, W

        x_flat = x.view(B, -1)  # B, H*W
        probs = torch.softmax(x_flat / self.softmax_temperature, dim=1)  # B, H*W
        log_probs = torch.log_softmax(x_flat / self.softmax_temperature, dim=1)  # B, H*W
        x_log_prob = log_probs.view(B, H, W)  # B, H, W
        x_prob = probs.view(B, H, W)  # B, H, W

        kpts = []
        log_probs = []
        masks = []
        logits_selected = []

        for b in range(B):
            # non maximum suppression
            kpts_nms = self.NMS(x_prob[b], self.threshold)

            scores = x_prob[b][kpts_nms[:, 1].long(), kpts_nms[:, 0].long()]
            sorted_idx = torch.argsort(-scores)

            kpts_batch = torch.zeros(self.min_num_keypoints, 2, device=x.device)
            masks_batch = torch.zeros(self.min_num_keypoints, dtype=torch.bool, device=x.device)

            num_keypoints = min(kpts_nms.shape[0], self.min_num_keypoints)
            kpts_batch[:num_keypoints, :] = kpts_nms[sorted_idx[:num_keypoints], :]
            masks_batch[:num_keypoints] = 1

            kpts.append(kpts_batch)
            masks.append(masks_batch)
            log_probs.append(x_log_prob[b][kpts[b][:, 1].long(), kpts[b][:, 0].long()])
            logits_selected.append(x[b][kpts[b][:, 1].long(), kpts[b][:, 0].long()])

        kpts = torch.stack(kpts, dim=0)  # B, N, 2
        log_probs = torch.stack(log_probs, dim=0)  # B, N
        masks = torch.stack(masks, dim=0)  # B, N
        logits_selected = torch.stack(logits_selected, dim=0)  # B, N

        if mask_padding is not None:
            mask_padding = mask_padding.view(B, -1)
            mask_padding = torch.gather(mask_padding, 1, kpts[:, :, 1].long() * W + kpts[:, :, 0].long())

        return kpts.flip(-1), log_probs, masks, mask_padding, logits_selected

    def NMS(self, x, threshold=3.0, kernel_size=3):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        pad = kernel_size // 2
        local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)

        pos = (x == local_max) & (x > threshold)
        return pos.nonzero()[..., 1:].flip(-1)
