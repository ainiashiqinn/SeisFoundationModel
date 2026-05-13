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
        t = torch.linspace(0.0, 1.0, T)
        x = torch.zeros(T)
        n_events = int(torch.randint(2, 5, (1,), generator=rng).item())
        for _ in range(n_events):
            t0 = float(torch.rand(1, generator=rng).item()) * 0.9 + 0.05
            f = float(torch.rand(1, generator=rng).item()) * 20.0 + 10.0
            amp = float(torch.rand(1, generator=rng).item()) * 0.8 + 0.2
            x = x + amp * _ricker(t, t0, f)
        x = x + 0.05 * torch.randn(T, generator=rng)
        return x.unsqueeze(0)  # (1, T)

    def _gen_multi(self, rng: torch.Generator) -> torch.Tensor:
        C = self.cfg["num_traces"]
        T = self.cfg["trace_length"]
        t = torch.linspace(0.0, 1.0, T)
        offsets = torch.linspace(-1.0, 1.0, C)
        gather = torch.zeros(C, T)
        n_events = int(torch.randint(1, 4, (1,), generator=rng).item())
        for _ in range(n_events):
            t0 = float(torch.rand(1, generator=rng).item()) * 0.6 + 0.2
            v = float(torch.rand(1, generator=rng).item()) * 1.5 + 0.5  # apparent velocity
            f = float(torch.rand(1, generator=rng).item()) * 20.0 + 10.0
            amp = float(torch.rand(1, generator=rng).item()) * 0.8 + 0.2
            for ci in range(C):
                tau = math.sqrt(t0 * t0 + (offsets[ci].item() / v) ** 2)
                gather[ci] = gather[ci] + amp * _ricker(t, tau, f)
        gather = gather + 0.05 * torch.randn(C, T, generator=rng)
        return gather  # (C, T)

    def _gen_2d(self, rng: torch.Generator) -> torch.Tensor:
        H, W = _as_hw(self.cfg["img_size"])
        # Treat 2D patch as a small (W traces) x (H time) seismic section.
        t = torch.linspace(0.0, 1.0, H)
        offsets = torch.linspace(-1.0, 1.0, W)
        patch = torch.zeros(H, W)
        n_events = int(torch.randint(1, 4, (1,), generator=rng).item())
        for _ in range(n_events):
            t0 = float(torch.rand(1, generator=rng).item()) * 0.6 + 0.2
            v = float(torch.rand(1, generator=rng).item()) * 1.5 + 0.5
            f = float(torch.rand(1, generator=rng).item()) * 20.0 + 10.0
            amp = float(torch.rand(1, generator=rng).item()) * 0.8 + 0.2
            for wi in range(W):
                tau = math.sqrt(t0 * t0 + (offsets[wi].item() / v) ** 2)
                patch[:, wi] = patch[:, wi] + amp * _ricker(t, tau, f)
        patch = patch + 0.05 * torch.randn(H, W, generator=rng)
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
        shuffle=(split == "train"),
        num_workers=cfg.get("num_workers", 0),
        pin_memory=cfg.get("pin_memory", True),
        drop_last=(split == "train"),
    )
