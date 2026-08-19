import torch
import torch.nn.functional as F

from ripepp import utils

from .base_sampler import BaseSampler

log = utils.get_pylogger(__name__)


class DaDSampler(BaseSampler):
    def __init__(
        self,
        num_samples=4096,
        nms_size=3,
        increase_coverage=True,
        coverage_size=51,
        coverage_pow=0.5,
        use_nms=True,
        remove_borders=True,
        sample_topk=True,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.nms_size = nms_size
        self.increase_coverage = increase_coverage
        self.coverage_size = coverage_size
        self.coverage_pow = coverage_pow
        self.use_nms = use_nms
        self.remove_borders = remove_borders
        self.sample_topk = sample_topk

        log.info(
            f"Using DaDSampler with num_samples={num_samples}, nms_size={nms_size}, increase_coverage={increase_coverage}, "
            f"coverage_size={coverage_size}, coverage_pow={coverage_pow}, use_nms={use_nms}, remove_borders={remove_borders}, sample_topk={sample_topk}"
        )

    def get_grid(self, B, H, W, device):
        x1_n = torch.meshgrid(
            *[torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=device) for n in (B, H, W)],
            indexing="ij",
        )
        x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
        return x1_n

    def do_inference(self, x):
        B, C, H, W = x.shape

        assert B == 1, "Batch size should be 1"

        output = self(x)

        kpts = output[0]
        scores = output[4]

        return kpts, scores  # N, 2

    def forward(self, x, mask_padding=None):
        B, C, H, W = x.shape
        assert C == 1, "Input to DeDoDeSampler must have one channel"

        x = x.squeeze(1)  # (B, H, W)

        # --- probability space ---
        x_flat = x.reshape(B, -1)
        keypoint_probs = torch.softmax(x_flat, dim=1).view(B, H, W)

        B, H, W = keypoint_probs.shape
        if self.increase_coverage:
            weights = (-(torch.linspace(-2, 2, steps=self.coverage_size, device=keypoint_probs.device) ** 2)).exp()[
                None, None
            ]
            # 10000 is just some number for maybe numerical stability, who knows. :), result is invariant anyway
            local_density_x = F.conv2d(
                (keypoint_probs[:, None] + 1e-6) * 10000,
                weights[..., None, :],
                padding=(0, self.coverage_size // 2),
            )
            local_density = F.conv2d(
                local_density_x,
                weights[..., None],
                padding=(self.coverage_size // 2, 0),
            )[:, 0]
            keypoint_probs = keypoint_probs * (local_density + 1e-8) ** (-self.coverage_pow)
        grid = self.get_grid(B, H, W, device=keypoint_probs.device).reshape(B, H * W, 2)
        if self.use_nms:
            keypoint_probs = keypoint_probs * (
                keypoint_probs == F.max_pool2d(keypoint_probs, self.nms_size, stride=1, padding=self.nms_size // 2)
            )
        if self.remove_borders:
            frame = torch.zeros_like(keypoint_probs)
            # we hardcode 4px, could do it nicer, but whatever
            frame[..., 4:-4, 4:-4] = 1
            keypoint_probs = keypoint_probs * frame
        if self.sample_topk:
            inds = torch.topk(keypoint_probs.reshape(B, H * W), k=self.num_samples).indices
        else:
            inds = torch.multinomial(
                keypoint_probs.reshape(B, H * W),
                num_samples=self.num_samples,
                replacement=False,
            )
        kps = torch.gather(grid, dim=1, index=inds[..., None].expand(B, self.num_samples, 2))

        log_probs = torch.gather(torch.log(keypoint_probs.reshape(B, H * W) + 1e-10), dim=1, index=inds)
        masks = torch.ones(B, self.num_samples, dtype=torch.bool, device=keypoint_probs.device)

        if mask_padding is not None:
            mask_padding = mask_padding.view(B, H * W)
            mask_padding = torch.gather(
                mask_padding,
                1,
                inds,
            )
            mask_padding = mask_padding & masks

        logits_selected = torch.gather(x.reshape(B, H * W), dim=1, index=inds)

        return kps, log_probs, masks, mask_padding, logits_selected
