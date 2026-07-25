"""
Seismic datasets.

Two backends:
  * SyntheticSeismicDataset  - generates plausible synthetic seismic samples
                               (Ricker-wavelet 1D traces, multi-trace gathers
                               with hyperbolic moveout, random 2D patches).
                               Useful for smoke-testing the model end-to-end.
  * NumpySeismicDataset      - loads .npy files from a directory.  One file per
                               sample; shape must match the configured
                               input_type:
                                   '1d'       -> (T,)        or (1, T)
                                   'multi_1d' -> (C, T)
                                   '2d'       -> (H, W)      or (1, H, W)

Selection is driven by the YAML config: set `data_path` to use the numpy
backend; omit it (or leave null) to use synthetic data.
"""
import math
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.normalize import normalize


def _as_hw(v: Union[int, Tuple[int, int], List[int]]) -> Tuple[int, int]:
    """Normalize a config value into an (H, W) tuple.

    Lets `img_size` in the YAML config be given as either a single int
    (square patch, H == W) or a 2-element (H, W) pair (rectangular patch).
    Used by `_gen_2d` (H = time samples, W = trace/offset samples) so the
    rest of that function doesn't need to branch on the config's shape.

    No validation: a non-int, non-length-2 input (e.g. a 3-element list)
    is not checked and will silently use only the first two entries.
    """
    if isinstance(v, int):
        return v, v
    return int(v[0]), int(v[1])


def _ricker(t: torch.Tensor, t0: float, f: float) -> torch.Tensor:
    a = (math.pi * f * (t - t0)) ** 2
    return (1.0 - 2.0 * a) * torch.exp(-a)


class SyntheticSeismicDataset(Dataset):
    def __init__(self, cfg: dict, num_samples: int = 1024, seed: int = 0, normalize_mode: str = "none"):
        self.cfg = cfg
        self.input_type = cfg["input_type"]
        self.num_samples = num_samples
        self.seed = seed
        self.normalize_mode = normalize_mode

    def __len__(self) -> int:
        return self.num_samples

    def _gen_1d(self, rng: torch.Generator) -> torch.Tensor:
        T = self.cfg["trace_length"]
        t = torch.linspace(0.0, 1.0, T)                                     # time axis in [0, 1] with equal spacing of T samples
        x = torch.zeros(T)                                                  # initialize trace of size T to zero
        n_events = int(torch.randint(2, 5, (1,), generator=rng).item())     # random number of events in [2, 4]
        for _ in range(n_events):
            t0 = float(torch.rand(1, generator=rng).item()) * 0.9 + 0.05    # random time shift in [0.05, 0.95)
            f = float(torch.rand(1, generator=rng).item()) * 20.0 + 10.0    # random frequency in [10, 30)
            amp = float(torch.rand(1, generator=rng).item()) * 0.8 + 0.2    # random amplitude in [0.2, 1.0)
            x = x + amp * _ricker(t, t0, f)                                 # add the Ricker wavelet to the trace at the random time shift
        x = x + 0.05 * torch.randn(T, generator=rng)                        # add Gaussian noise with std 0.05  
        return x.unsqueeze(0)  # (1, T)

    def _gen_multi(self, rng: torch.Generator) -> torch.Tensor:
        C = self.cfg["num_traces"]
        T = self.cfg["trace_length"]
        t = torch.linspace(0.0, 1.0, T)                                     # time axis in [0, 1] with equal spacing of T samples
        offsets = torch.linspace(-1.0, 1.0, C)                              # offsets for each trace in [-1, 1] with equal spacing of C traces
        gather = torch.zeros(C, T)                                          # initialize gather of size (C, T) to zero
        n_events = int(torch.randint(1, 4, (1,), generator=rng).item())     # random number of events in [1, 3]
        for _ in range(n_events):
            t0 = float(torch.rand(1, generator=rng).item()) * 0.6 + 0.2     # random time shift in [0.2, 0.8)
            v = float(torch.rand(1, generator=rng).item()) * 1.5 + 0.5      # apparent velocity in [0.5, 2.0)
            f = float(torch.rand(1, generator=rng).item()) * 20.0 + 10.0    # random frequency in [10, 30)
            amp = float(torch.rand(1, generator=rng).item()) * 0.8 + 0.2    # random amplitude in [0.2, 1.0)
            for ci in range(C):
                tau = math.sqrt(t0 * t0 + (offsets[ci].item() / v) ** 2)    # compute the time shift (moveout) for each trace based on the offset and apparent velocity
                gather[ci] = gather[ci] + amp * _ricker(t, tau, f)          # add the Ricker wavelet to the trace at the computed time shift
        gather = gather + 0.05 * torch.randn(C, T, generator=rng)           # add Gaussian (independent) noise with std 0.05
        return gather  # (C, T)

    def _gen_2d(self, rng: torch.Generator) -> torch.Tensor:
        H, W = _as_hw(self.cfg["img_size"])
        # Treat 2D patch as a small (W traces) x (H time) seismic section.
        t = torch.linspace(0.0, 1.0, H)                                     # time axis in [0, 1] with equal spacing of H samples
        offsets = torch.linspace(-1.0, 1.0, W)                              # offsets for each trace in [-1, 1] with equal spacing of W traces
        patch = torch.zeros(H, W)                                           # initialize patch of size (H, W) to zero
        n_events = int(torch.randint(1, 4, (1,), generator=rng).item())     # random number of events in [1, 3]
        for _ in range(n_events):
            t0 = float(torch.rand(1, generator=rng).item()) * 0.6 + 0.2     # random time shift in [0.2, 0.8)
            v = float(torch.rand(1, generator=rng).item()) * 1.5 + 0.5      # apparent velocity in [0.5, 2.0)
            f = float(torch.rand(1, generator=rng).item()) * 20.0 + 10.0    # random frequency in [10, 30)
            amp = float(torch.rand(1, generator=rng).item()) * 0.8 + 0.2    # random amplitude in [0.2, 1.0)
            for wi in range(W):
                tau = math.sqrt(t0 * t0 + (offsets[wi].item() / v) ** 2)    # compute the time shift (moveout) for each trace based on the offset and apparent velocity
                patch[:, wi] = patch[:, wi] + amp * _ricker(t, tau, f)      # add the Ricker wavelet to the trace at the computed time shift
        patch = patch + 0.05 * torch.randn(H, W, generator=rng)             # add Gaussian noise with std 0.05
        return patch.unsqueeze(0)  # (1, H, W)

    def __getitem__(self, idx: int) -> torch.Tensor:
        rng = torch.Generator().manual_seed(self.seed + idx)
        if self.input_type == "1d":
            x = self._gen_1d(rng)
        elif self.input_type == "multi_1d":
            x = self._gen_multi(rng)
        elif self.input_type == "2d":
            x = self._gen_2d(rng)
        else:
            raise ValueError(f"Unknown input_type: {self.input_type!r}")
        return normalize(x, self.normalize_mode)


class NumpySeismicDataset(Dataset):
    def __init__(self, root: str, input_type: str, normalize_mode: str = "none"):
        self.root = Path(root)
        self.input_type = input_type
        self.normalize_mode = normalize_mode
        self.files = sorted(self.root.glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"No .npy files found under {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        arr = np.load(self.files[idx]).astype(np.float32)
        x = torch.from_numpy(arr)
        if self.input_type == "1d" and x.dim() == 1:
            x = x.unsqueeze(0)              # (T,) -> (1, T)
        elif self.input_type == "2d" and x.dim() == 2:
            x = x.unsqueeze(0)              # (H, W) -> (1, H, W)
        return normalize(x, self.normalize_mode)


def build_dataloader(cfg: dict, split: str = "train") -> DataLoader:
    normalize_mode = cfg.get("normalize", "none")
    data_path = cfg.get("data_path")
    if data_path:
        dataset = NumpySeismicDataset(
            data_path, input_type=cfg["input_type"], normalize_mode=normalize_mode
        )
    else:
        dataset = SyntheticSeismicDataset(
            cfg, num_samples=cfg.get("num_samples", 1024), normalize_mode=normalize_mode,
        )
    return DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 32),
        shuffle=(split == "train"),                         # shuffle training data, but not validation/test data
        num_workers=cfg.get("num_workers", 0),              # number of subprocesses to use for data loading (0 means the data will be loaded in the main process)
        pin_memory=cfg.get("pin_memory", True),             # whether to copy tensors into CUDA pinned memory before returning them (improves GPU transfer speed)
        drop_last=(split == "train"),                       # drop the last incomplete batch if the dataset size is not divisible by the batch size (only for training)
    )
