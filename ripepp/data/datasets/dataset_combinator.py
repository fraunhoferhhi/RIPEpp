import torch

from ripepp import utils

from .base_dataset import BaseDataset

log = utils.get_pylogger(__name__)

REQUIRED_DATA_KEYS = [
    "src_image",
    "trg_image",
    "src_mask",
    "trg_mask",
    "label",
    "src_path",
    "trg_path",
]


class DatasetCombinator(BaseDataset):
    """Combines multiple datasets into one. Length of the combined dataset is the length of the
    longest dataset. Shorter datasets are looped over.

    Args:
        datasets: List of datasets to combine.
        mode: How to sample from the datasets. Can be either "uniform" or "weighted".
            In "uniform" mode, each dataset is sampled with equal probability.
            In "weighted" mode, each dataset is sampled with probability proportional to its length.
            In "custom" mode, the user can specify custom weights for each dataset. Weights must sum to 1.
        weights: List of weights for each dataset in "custom" mode. Must sum to 1 and have the same length as the number of datasets. Ignored in "uniform" and "weighted" modes.
    """

    def __init__(self, datasets, mode="uniform", weights=None):
        super().__init__(positive_only=all(ds.positive_only for ds in datasets))

        self.datasets = datasets

        names_datasets = [type(ds).__name__ for ds in self.datasets]
        self.lengths = [len(ds) for ds in datasets]

        if mode == "weighted":
            self.probs_datasets = [length / sum(self.lengths) for length in self.lengths]
        elif mode == "uniform":
            self.probs_datasets = [1 / len(self.datasets) for _ in self.datasets]
        elif mode == "custom":
            assert weights is not None, "Weights must be provided in custom mode"
            assert len(weights) == len(datasets), "Number of weights must match number of datasets"
            assert sum(weights) == 1.0, "Weights must sum to 1"
            self.probs_datasets = weights
        else:
            raise ValueError(f"Unknown mode {mode}")

        log.info("Got the following datasets: ")

        for name, length, prob in zip(names_datasets, self.lengths, self.probs_datasets, strict=False):
            log.info(f"{name} with {length} samples and probability {prob}")
        log.info(f"Total number of samples: {sum(self.lengths)}")

        self.num_samples = max(self.lengths)

        self.dataset_dist = torch.distributions.Categorical(probs=torch.tensor(self.probs_datasets))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        dataset_idx = self.dataset_dist.sample().item()

        idx = torch.randint(0, self.lengths[dataset_idx], (1,)).item()
        sample = self.datasets[dataset_idx][idx]

        output = {key: sample[key] for key in REQUIRED_DATA_KEYS}

        return output
