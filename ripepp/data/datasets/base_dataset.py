from abc import abstractmethod

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Base class for datasets."""

    def __init__(self, **kwargs):
        super().__init__()

        # check if positive_only is in kwargs
        if "positive_only" in kwargs:
            self._positive_only = kwargs["positive_only"]
        else:
            raise ValueError(
                "positive_only argument is required for dataset initialization."
            )

    @property
    def positive_only(self) -> bool:
        return self._positive_only

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError
