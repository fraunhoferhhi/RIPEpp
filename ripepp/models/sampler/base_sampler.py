from abc import abstractmethod

import torch.nn as nn


class BaseSampler(nn.Module):
    """
    Base class for all samplers.
    Defines the interface for all samplers.
    Methods to implement:
      - inference_sampling for inference-time sampling of keypoints with parameters top_k and threshold
      - forward for training-time sampling of keypoints with parameters x and mask_padding
    """

    def __init__(self):
        super().__init__()

    def check_overwrites_inference(self, inference_config):
        """
        Check if the inference config overwrites already set parameters.

        If so, save the old parameter and set the new one.

        Args:
            inference_config: Dict with inference parameters
        """

        self.old_inference_params = {}
        for key in inference_config:
            if hasattr(self, key):
                self.old_inference_params[key] = getattr(self, key)
                setattr(self, key, inference_config[key])

    def restore_inference_params(self):
        """
        Restore the old inference parameters.
        """

        for key in self.old_inference_params:
            setattr(self, key, self.old_inference_params[key])
        self.old_inference_params = {}

    @abstractmethod
    def do_inference(self, x):
        """
        Actual inference logic to be implemented by subclasses.

        Args:
            x: Input heatmap/logits

        Returns:
            Keypoints tensor of shape [N, 2] where N is the number of keypoints in xy format
        """
        raise NotImplementedError

    def inference_sampling(self, x, **kwargs):
        """
        Sample keypoints during inference.

        Args:
            x: Input heatmap/logits
            **kwargs: Sampler-specific parameters (e.g., top_k, threshold, num_samples, etc.)
                     Each sampler defines its own expected parameters.

        Returns:
            Keypoints tensor of shape [N, 2] where N is the number of keypoints in xy format
        """
        self.check_overwrites_inference(kwargs)
        try:
            keypoints = self.do_inference(x)
            return keypoints
        finally:
            self.restore_inference_params()

    @abstractmethod
    def forward(self, x, mask_padding=None):
        """
        Sample keypoints during training.

        Args:
            x: Input heatmap/logits
            mask_padding: Optional padding mask

        Returns:
            Keypoints tensor of shape [B, N, 2] where N is the number of keypoints in xy format
            log_probs: Log probabilities of the sampled keypoints of shape [B, N]
            mask: Mask of selected keypoints of shape [B, N]
            mask_padding: Mask for the padded area of shape [B, H, W]
            logits_selected: Logits of selected keypoints of shape [B, N]
        """
        raise NotImplementedError
