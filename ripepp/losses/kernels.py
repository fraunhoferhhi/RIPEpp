"""Kernel functions for converting Sampson distances to rewards."""

from ripepp import utils

log = utils.get_pylogger(__name__)


class HuberKernel:
    """Huber kernel for Sampson distance-based rewards.

    Returns rewards in [0, 1] for small distances, -1 for distances above threshold.
    """

    def __init__(self, delta=1.0, threshold=1.0):
        self.delta = delta
        self.threshold = threshold

        # Huber sanity check
        log.info(f"Huber kernel initialized with delta={self.delta}, threshold={self.threshold}")
        point_of_intersection = (2 * self.delta**2) ** 0.5
        log.info(f"Intersection with x-axis at {point_of_intersection}")
        assert point_of_intersection <= self.threshold, (
            "Huber kernel threshold too small! This would lead to positive rewards for outliers."
        )

    def __call__(self, d, outlier_penalty=-0.1):
        """
        Args:
            d: Sampson distances (any shape)
            outlier_penalty: Reward for outliers (distances > threshold)
        Returns:
            Rewards of the same shape, ranging from 1 (d=0) to outlier_penalty (d>threshold)
        """
        val = 1 - 0.5 * (d / (self.delta + 1e-8)) ** 2
        val[d > self.threshold] = outlier_penalty
        return val
