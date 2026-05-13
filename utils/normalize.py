"""
Sample-level normalization for seismic inputs.

All modes compute stats over **all elements** of the sample (joint normalization).
For multi-trace gathers (e.g. angle gathers), joint normalization preserves
relative amplitudes between traces -- which carry AVO information.

If you need per-trace normalization, do it in your own dataset before yielding.
"""
import torch


def normalize(x: torch.Tensor, mode: str, eps: float = 1e-6) -> torch.Tensor:
    """Normalize a single sample (any shape).

    mode:
        'none'     -> return x unchanged
        'zscore'   -> (x - mean) / std          (zero mean, unit variance)
        'minmax'   -> (x - min) / (max - min)   (scaled to [0, 1])
        'rms'      -> x / sqrt(mean(x^2))       (preserves sign, unit RMS)
        'max_abs'  -> x / max(|x|)              (scaled to [-1, 1], preserves zero)

    For seismic data, `rms` and `max_abs` are usually preferable to `minmax`
    because they preserve the zero point of the wavelet.
    """
    if mode == "none":
        return x
    x = x.float()
    if mode == "zscore":
        return (x - x.mean()) / (x.std() + eps)
    if mode == "minmax":
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + eps)
    if mode == "rms":
        rms = (x ** 2).mean().sqrt()
        return x / (rms + eps)
    if mode == "max_abs":
        return x / (x.abs().max() + eps)
    raise ValueError(
        f"Unknown normalize mode: {mode!r}.  "
        f"Expected one of: 'none', 'zscore', 'minmax', 'rms', 'max_abs'."
    )
