import torch
import torch.nn.functional as F

from ripepp import utils

from .base_sampler import BaseSampler

log = utils.get_pylogger(__name__)


class DeDoDeSampler(BaseSampler):
    def __init__(
        self,
        min_num_keypoints=4096,
        use_nms=True,
        increase_coverage=True,
        remove_borders=True,
    ):
        super().__init__()
        self.min_num_keypoints = min_num_keypoints
        self.use_nms = use_nms
        self.increase_coverage = increase_coverage
        self.remove_borders = remove_borders

        log.info(
            f"Using DeDoDeSampler with min_num_keypoints={min_num_keypoints}, use_nms={use_nms}, increase_coverage={increase_coverage}, remove_borders={remove_borders}"
        )

    def get_grid(self, B, H, W, device):
        ys, xs = torch.meshgrid(
            torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=device),
            torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=device),
            indexing="ij",
        )
        grid = torch.stack((xs, ys), dim=-1)  # (H, W, 2)
        grid = grid.view(1, H * W, 2).expand(B, -1, -1)  # (B, H*W, 2)
        return grid

    def do_inference(self, x):
        B, C, H, W = x.shape

        assert B == 1, "Batch size should be 1"

        output = self(x)

        kpts = output[0][0]

        return kpts  # N, 2

    def forward(self, x, mask_padding=None):
        """
        Args:
            x: (B, 1, H, W) logits (arbitrary real values)
            mask_padding: optional (B, H, W) boolean mask

        Returns:
            kpts: (B, K, 2) integer pixel coordinates (row, col)
            log_probs: (B, K) log-probabilities (masked entries undefined)
            masks: (B, K) validity mask
            mask_padding: (B, K) padding mask aligned with kpts (or None)
            logits_selected: (B, K) raw logits at selected locations
        """
        B, C, H, W = x.shape
        assert C == 1, "Input to DeDoDeSampler must have one channel"

        x = x.squeeze(1)  # (B, H, W)

        # --- probability space ---
        x_flat = x.reshape(B, -1)
        x_probs = torch.softmax(x_flat, dim=1).view(B, H, W)
        x_log_prob = torch.log(x_probs + 1e-10)

        # --- coverage reweighting (prob-space density, log-space update) ---
        if self.increase_coverage:
            weights = torch.exp(-(torch.linspace(-2, 2, steps=51, device=x.device) ** 2))
            weights = weights / weights.sum()
            weights = weights[None, None]

            local_density_x = F.conv2d(x_probs[:, None], weights[..., None, :], padding=(0, 25))
            local_density = F.conv2d(local_density_x, weights[..., None], padding=(25, 0))[:, 0]

            x_log_prob = x_log_prob - 0.5 * torch.log(local_density + 1e-8)

        # --- grid ---
        grid = self.get_grid(B, H, W, x.device)

        # --- NMS ---
        if self.use_nms:
            pooled = F.max_pool2d(x_log_prob, 3, 1, 1)
            x_log_prob = torch.where(
                x_log_prob >= pooled,
                x_log_prob,
                torch.full_like(x_log_prob, -float("inf")),
            )

        if self.remove_borders:
            frame = torch.zeros_like(x_log_prob)
            frame[:, 4:-4, 4:-4] = 1
            x_log_prob = x_log_prob + torch.log(frame + 1e-10)

        # --- top-k with validity guard ---
        flat = x_log_prob.reshape(B, -1)
        valid_flat = torch.isfinite(flat)
        num_valid = valid_flat.sum(dim=1)
        k = min(self.min_num_keypoints, int(num_valid.min().item()))

        inds = torch.topk(flat, k=k).indices  # (B, k)

        # --- pad indices ---
        inds_padded = torch.zeros(B, self.min_num_keypoints, dtype=torch.long, device=x.device)
        inds_padded[:, :k] = inds

        # --- masks ---
        masks = torch.zeros(B, self.min_num_keypoints, dtype=torch.bool, device=x.device)
        masks[:, :k] = True

        # --- gather outputs ---
        kpts = torch.gather(grid, 1, inds_padded[..., None].expand(B, self.min_num_keypoints, 2))
        log_probs = torch.gather(flat, 1, inds_padded)
        logits_selected = torch.gather(x_flat, 1, inds_padded)

        # --- convert to pixel coordinates (row, col) ---
        kpts = (kpts + 1) * torch.tensor([W - 1, H - 1], device=x.device)[None, None, :] / 2.0
        kpts = kpts.round().long()

        # --- padding mask ---
        if mask_padding is not None:
            mask_padding = mask_padding.view(B, -1)
            mask_padding = torch.gather(
                mask_padding,
                1,
                kpts[:, :, 1] * W + kpts[:, :, 0],
            )
            mask_padding = mask_padding & masks

        return kpts, log_probs, masks, mask_padding, logits_selected
