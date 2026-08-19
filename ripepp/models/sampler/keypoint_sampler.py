import torch
import torch.nn as nn

from ripepp import utils
from ripepp.utils.utils import extract_patches_from_inds, get_grid, gridify

from .base_sampler import BaseSampler

log = utils.get_pylogger(__name__)


class KeypointSampler(BaseSampler):
    """
    Sample keypoints according to a Heatmap
    Input
      x: [B, 1, H, W] heatmap

    Returns
      [list]:
        kps: [N, 2] - keypoint positions
        log_probs: [N] - logprobs for each kp
    """

    def __init__(
        self,
        window_size=8,
        threshold=0.5,
        top_k=2048,
        subpixel_sampling=False,
        sub_pixel_temp=0.5,
        nms_size=3,
    ):
        super().__init__()
        self.window_size = window_size
        self.idx_cells = None  # Cache for meshgrid indices
        self.threshold = threshold
        self.subpixel_sampling = subpixel_sampling
        self.sub_pixel_temp = sub_pixel_temp
        self.nms_size = nms_size
        self.top_k = top_k
        log.info(f"Using KeypointSampler with window size {window_size}")

    def sample(self, grid):
        """
        Sample keypoints given a grid where each cell has logits stacked in last dimension
        Input
          grid: [B, C, H//w, W//w, w*w]

        Returns
          log_probs: [B, H//w, W//w ] - logprobs of selected samples
          choices: [B, C, H//w, W//w] indices of choices
          accept_mask: [B, H//w, W//w] mask of accepted keypoints
          logits_selected: [B, H//w, W//w] - logits of selected samples

        """
        chooser = torch.distributions.Categorical(logits=grid)
        choices = chooser.sample()  # [B, C, H//w, W//w]
        logits_selected = torch.gather(grid, -1, choices.unsqueeze(-1)).squeeze(-1)  # [B, C, H//w, W//w]

        flipper = torch.distributions.Bernoulli(logits=logits_selected)
        accepted_choices = flipper.sample()

        # Sum log-probabilities is equivalent to multiplying the probabilities
        log_probs = chooser.log_prob(choices) + flipper.log_prob(accepted_choices)  # B, C, H//w, W//w

        accept_mask = accepted_choices.gt(0)  # [B, C, H//w, W//w]

        return (
            log_probs.squeeze(1),
            choices,
            accept_mask.squeeze(1),
            logits_selected.squeeze(1),
        )

    def precompute_idx_cells(self, H, W, device):
        idx_cells = gridify(
            torch.dstack(
                torch.meshgrid(
                    torch.arange(H, dtype=torch.float32, device=device),
                    torch.arange(W, dtype=torch.float32, device=device),
                )
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(1, -1, -1, -1),
            window_size=self.window_size,
        )

        return idx_cells

    def NMS(self, x, threshold=3.0, kernel_size=3):
        pad = kernel_size // 2
        local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)

        x = x * ((x == local_max) & (x > threshold))
        return x

    def do_inference(self, x):
        B, K, H, W = x.shape

        assert K == 1, "Only single channel heatmaps supported"

        x = x.squeeze(1)  # [B, H, W]
        scoremap = x.clone()

        grid = get_grid(B, H, W, x.device).reshape(B, H * W, 2)

        if self.nms_size > 0:
            x = self.NMS(x, self.threshold, self.nms_size)
        else:
            x = x * (x > self.threshold)

        if self.top_k is not None:
            inds = torch.topk(x.reshape(B, H * W), k=self.top_k).indices
        else:
            raise NotImplementedError("Only top_k is supported currently")

        kps = torch.gather(grid, dim=1, index=inds[..., None].expand(B, self.top_k, 2))
        scores = torch.gather(scoremap.reshape(B, H * W), dim=1, index=inds)

        if self.subpixel_sampling:
            offsets = get_grid(B, self.nms_size, self.nms_size, x.device).reshape(
                B, self.nms_size**2, 2
            )  # B x K_H x K_W x 2
            offsets[..., 0] = offsets[..., 0] * self.nms_size / W
            offsets[..., 1] = offsets[..., 1] * self.nms_size / H
            keypoint_patch_scores = extract_patches_from_inds(scoremap, inds, self.nms_size)
            keypoint_patch_probs = (keypoint_patch_scores / self.sub_pixel_temp).softmax(dim=1)  # B x K_H * K_W x N
            keypoint_offsets = torch.einsum("bkn, bkd ->bnd", keypoint_patch_probs, offsets)
            kps = kps + keypoint_offsets

        scores = scores / scoremap.max()

        return kps, scores  # B, N, 2 and B, N

    def forward(self, x, mask_padding=None):
        """
        Input
          x: [B, 1, H, W] heatmap
        Returns
          keypoints: [B, H//w, W//w, 2] - keypoint positions
          log_probs: [B, H//w, W//w] - logprobs for each kp
          mask: [B, H//w, W//w] - mask of selected keypoints
          mask_padding: [B, 1, H//w, W//w] - mask of padded keypoints
          logits_selected: [B, H//w, W//w] - logits of selected keypoints
        """
        B, C, H, W = x.shape

        # # standardize x between 0 and 1
        # min_vals = torch.min(x.view(B,-1),dim=1).values.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # max_vals = torch.max(x.view(B,-1),dim=1).values.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        # x = (x - min_vals) / (max_vals - min_vals)

        keypoint_cells = gridify(x, self.window_size)
        num_samples = keypoint_cells.shape[-2] * keypoint_cells.shape[-3]

        mask_padding = (
            (torch.min(gridify(mask_padding, self.window_size), dim=4).values) if mask_padding is not None else None
        )

        if self.idx_cells is None or self.idx_cells.shape[2:4] != (
            H // self.window_size,
            W // self.window_size,
        ):
            self.idx_cells = self.precompute_idx_cells(H, W, x.device)

        log_probs, in_patch_idx, mask, logits_selected = self.sample(keypoint_cells)

        grid = get_grid(B, H, W, x.device).reshape(B, H * W, 2)

        idx_all = torch.arange(0, H * W, device=x.device).unsqueeze(0).expand(B, -1).reshape(B, 1, H, W)
        idx_all = gridify(idx_all, self.window_size).reshape(B, num_samples, -1)
        idx = torch.gather(idx_all, dim=2, index=in_patch_idx.view(B, -1, 1).expand(B, num_samples, 1))

        keypoints = torch.gather(grid, dim=1, index=idx.expand(B, num_samples, 2))

        # old version with absolute coordinates
        # keypoints = (
        #     torch.gather(self.idx_cells.expand(B, -1, -1, -1, -1), -1, in_patch_idx.repeat(1, 2, 1, 1).unsqueeze(-1))
        #     .squeeze(-1)
        #     .permute(0, 2, 3, 1)
        # )
        # keypoints = keypoints.flip(-1)

        return (
            keypoints,
            log_probs.view(B, -1),
            mask.view(B, -1),
            mask_padding.view(B, -1) if mask_padding is not None else None,
            logits_selected.view(B, -1) if logits_selected is not None else None,
        )
