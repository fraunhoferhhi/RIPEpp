import pickle
from typing import Any

import numpy as np
import torch

# from: https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/


class NumpySerializedList:
    def __init__(self, lst: list[Any]):
        lst = [np.frombuffer(pickle.dumps(x), dtype=np.uint8) for x in lst]
        self._addr = np.cumsum([len(x) for x in lst])
        self._lst = np.concatenate(lst)

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx: int):
        start = 0 if idx == 0 else self._addr[idx - 1]
        end = self._addr[idx]
        return pickle.loads(memoryview(self._lst[start:end]))


class TorchSerializedList(NumpySerializedList):
    def __init__(self, lst: list):
        super().__init__(lst)
        self._addr = torch.from_numpy(self._addr)
        self._lst = torch.from_numpy(self._lst)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr].numpy())
        return pickle.loads(bytes)
