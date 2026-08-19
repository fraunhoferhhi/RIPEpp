import numpy as np
import torch
from pathlib import Path

from hydra.utils import instantiate
import torch.nn as nn
import torch.nn.functional as F

from ripepp import utils
from ripepp.models.upsampler.direct_sample import DirectSample
from ripepp.models.upsampler.hypercolumn_features import HyperColumnFeatures
from ripepp.utils.utils import unnormalize_coords

log = utils.get_pylogger(__name__)


def load_model_from_checkpoint(checkpoint_path: Path = None, device="cpu"):
    """
    Load RIPE model from checkpoint with automatic configuration.

    This function automatically loads the model architecture from the configuration
    embedded in the checkpoint file. It requires checkpoints saved with the new
    format that includes configuration metadata.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to (default: 'cpu')

    Returns:
        RIPE model loaded with weights and configuration

    Raises:
        ValueError: If checkpoint doesn't contain config (old format)
        FileNotFoundError: If checkpoint file doesn't exist

    Example:
        >>> model = load_model_from_checkpoint("path/to/checkpoint.pth", device="cuda")
        >>> kpts, descs, scores = model.detectAndCompute(img)
    """
    from ripepp.utils.checkpoint import load_checkpoint, load_config_from_checkpoint, validate_checkpoint_path

    checkpoint_path = validate_checkpoint_path(checkpoint_path)

    # Load config from checkpoint
    config = load_config_from_checkpoint(checkpoint_path)

    if config is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path.name} does not contain configuration metadata. "
            f"This is an old-format checkpoint. Please use ripe_model_factory() "
            f"with config_file_path parameter to specify the model architecture manually."
        )

    log.info(f"Loading model from checkpoint: {checkpoint_path.name}")
    log.info(f"Checkpoint step: {load_checkpoint(checkpoint_path)['step']}")
    log.info(f"Backbone: {config.backbones._target_}")
    log.info(f"Sampler: {config.keypoint_sampler._target_}")

    # Instantiate model components from config
    backbone = instantiate(config.backbones)
    sampler = instantiate(config.keypoint_sampler)

    # Get model parameters from config
    window_size = config.get("window_size", 8)
    descriptor_dim = config.get("descriptor_dim", 256)

    # Handle non_linearity_dect
    non_linearity_dect = None
    if "network" in config and "non_linearity_dect" in config.network:
        non_linearity_dect = instantiate(config.network.non_linearity_dect)

    # Instantiate model
    model = RIPEPP(
        net=backbone,
        keypoint_sampler=sampler,
        window_size=window_size,
        non_linearity_dect=non_linearity_dect,
        descriptor_dim=descriptor_dim,
    )

    # Load weights
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    return model.to(device)


class RIPEPP(nn.Module):
    """
    Base class for extracting keypoints and descriptors
    Input
      x: [B, C, H, W] Images

    Returns
      kpts:
        list of size [B] with detected keypoints
      descs:
        list of size [B] with descriptors
    """

    def __init__(
        self,
        net,
        keypoint_sampler,
        window_size: int = 8,
        non_linearity_dect=None,
        descriptor_dim: int = 256,
    ):
        super().__init__()
        self.net = net

        self.is_combined_descriptors = self.net.mode == "dect+desc"

        self.detector = keypoint_sampler
        self.descriptor_upsampler = HyperColumnFeatures()
        self.descriptor_sampler = DirectSample()
        self.window_size = window_size
        self.non_linearity_dect = torch.nn.Identity() if non_linearity_dect is None else non_linearity_dect

        log.info(f"Training with window size {window_size}.")
        log.info(f"Use {self.non_linearity_dect} as final non-linearity before the detection heatmap.")

        self.conv_dim_reduction_coarse_desc = nn.Conv1d(
            sum(self.get_dim_raw_desc()),
            descriptor_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def train(self, mode=True):
        super().train(mode)
        return self

    def get_dim_raw_desc(self):
        return self.net.get_dim_layers_encoder()

    @torch.inference_mode()
    def detectAndCompute(self, img, output_aux=False, desc_blend_ratio=None, **inference_kwargs):
        """
        Detect keypoints and compute descriptors.

        Args:
            img: Input image [B, C, H, W]
            output_aux: Whether to return auxiliary outputs
            desc_blend_ratio: Blend weight for combining encoder/decoder descriptors
            **inference_kwargs: Sampler-specific parameters

        Returns:
            kpts: Keypoints [B, N, 2] in xy format
            descs: Descriptors [B, N, descriptor_dim]
            scores: Keypoint scores [B, N]
            (optional) aux: Dictionary with auxiliary outputs
        """
        self.train(False)

        # check if image is grayscale
        if img.shape[1] == 1:
            img = img.expand(-1, 3, -1, -1)

        out = self.net(img)
        out["heatmap"] = self.non_linearity_dect(out["heatmap"])

        B, K, H, W = out["heatmap"].shape

        # Pass kwargs directly to sampler
        kpts, scores = self.detector.inference_sampling(out["heatmap"], **inference_kwargs)

        if kpts.shape[1] == 0:
            raise RuntimeError("No keypoints detected")

        out["kpts"] = kpts

        desc_info = self.get_descriptors(
            out,
            blend_weight=desc_blend_ratio,
        )
        descs = desc_info["combined_desc"]

        # unnormalize keypoints to image size
        kpts = unnormalize_coords(kpts.float(), H, W)

        if output_aux:
            return (
                kpts.float(),
                descs,
                scores,
                {
                    "heatmap": out["heatmap"],
                    "encoder_layers": out["encoder_descs_layers"],
                    "decoder_desc_map": out["decoder_descs_map"] if self.is_combined_descriptors else None,
                    "conv": self.conv_dim_reduction_coarse_desc,
                },
            )

        return kpts.float(), descs, scores

    def dim_reduction_descriptors(self, descs):
        if isinstance(self.conv_dim_reduction_coarse_desc, nn.ModuleList):
            # individual descriptor convolutions for each layer
            desc_conv = []
            for desc, conv in zip(descs, self.conv_dim_reduction_coarse_desc, strict=False):
                desc_conv.append(conv(desc.permute(0, 2, 1)).permute(0, 2, 1))
            desc = torch.cat(desc_conv, dim=-1)
        else:
            if isinstance(descs, list):
                # concatenate descriptors from different layers
                desc = torch.cat(descs, dim=-1)
            else:
                desc = descs

            desc = self.conv_dim_reduction_coarse_desc(desc.permute(0, 2, 1)).permute(0, 2, 1)

        return desc

    def sample_descriptors(self, sampler_fn, feature_map, kpts, dim_reduction_fn=None):
        descs = sampler_fn(feature_map, kpts)

        if dim_reduction_fn is not None:
            descs = dim_reduction_fn(descs)

        descs = F.normalize(descs, dim=2)

        return descs

    def combine_descriptor_streams(self, encoder_desc, decoder_desc, weight):
        if decoder_desc is None or weight is None:
            return encoder_desc

        weight = float(weight)

        assert 0.0 <= weight <= 1.0, "Blend weight should be between 0 and 1"

        blended = (1.0 - weight) * encoder_desc + weight * decoder_desc
        blended = F.normalize(blended, dim=2)

        return blended

    def get_descriptors(self, out, blend_weight=None):
        desc_dict = {}

        desc_dict["encoder_descs"] = self.sample_descriptors(
            self.descriptor_upsampler,
            out["encoder_descs_layers"],
            out["kpts"],
            dim_reduction_fn=self.dim_reduction_descriptors,
        )
        if self.is_combined_descriptors:
            desc_dict["decoder_descs"] = self.sample_descriptors(
                self.descriptor_sampler,
                out["decoder_descs_map"],
                out["kpts"],
                dim_reduction_fn=None,
            )
        else:
            desc_dict["decoder_descs"] = None

        desc_dict["combined_desc"] = self.combine_descriptor_streams(
            desc_dict["encoder_descs"], desc_dict["decoder_descs"], blend_weight
        )

        return desc_dict

    def forward(self, x, mask_padding=None):
        B, C, H, W = x.shape
        out = self.net(x)
        out["heatmap"] = self.non_linearity_dect(out["heatmap"])

        kpts, log_probs, mask, mask_padding, logits_selected = self.detector(out["heatmap"], mask_padding)

        return {
            "kpts": kpts,
            "logprobs": log_probs,
            "mask_selected": mask,
            "mask_padding": mask_padding,
            "logits_selected": logits_selected if logits_selected is not None else None,
            **out,
        }


def output_number_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nb_params = sum([np.prod(p.size()) for p in model_parameters])

    log.info(f"Number of trainable parameters: {nb_params:d}")
