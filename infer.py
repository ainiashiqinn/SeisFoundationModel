"""
SeisFoundation inference entrypoint.

Loads a checkpoint, runs the model on the configured data, and writes
per-batch .pt files containing:
    input         - the input tensor
    latent        - encoder output (B, N+1, embed_dim)
    cls           - CLS token only  (B, embed_dim)
    pred_patches  - decoder output in patch space (B, N, patch_dim)
    recon         - decoder output in original input shape
    mask          - (B, N) or None  (only set when mask_ratio > 0)

Edit the config path at the bottom of this file and run:
    python infer.py
"""
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

# Ensure local-package imports work when run from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import build_dataloader  # noqa: E402
from models.foundation import SeisFoundation  # noqa: E402


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def pick_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("[infer] CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        print("[infer] MPS requested but unavailable; falling back to CPU.")
        return "cpu"
    return requested


def load_weights(model: SeisFoundation, ckpt_path: str, device: str):
    """Accepts either a full checkpoint (has 'model_state') or an encoder-only
    checkpoint (has 'encoder_state').  Encoder-only loads will leave the
    decoder randomly initialized -- decoder reconstructions will be garbage."""
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        print(f"[infer] loaded full model from {ckpt_path}")
    elif "encoder_state" in ckpt:
        missing, unexpected = model.load_state_dict(ckpt["encoder_state"], strict=False)
        print(
            f"[infer] loaded ENCODER ONLY from {ckpt_path} -- decoder is random. "
            f"({len(missing)} missing, {len(unexpected)} unexpected)"
        )
    else:
        raise KeyError(
            f"{ckpt_path} has neither 'model_state' nor 'encoder_state'; "
            f"keys: {list(ckpt.keys())}"
        )


def main(config_path: str):
    cfg = load_cfg(config_path)
    model_cfg = cfg["model"]
    data_cfg = {**model_cfg, **cfg.get("data", {})}
    infer_cfg = cfg["inference"]
    device = pick_device(cfg.get("train", {}).get("device", "cpu"))

    ckpt_path = infer_cfg["ckpt_path"]
    out_dir = Path(infer_cfg.get("out_dir", "./inference_out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_ratio = float(infer_cfg.get("mask_ratio", 0.0))
    max_batches = infer_cfg.get("max_batches")

    model = SeisFoundation(model_cfg).to(device)
    load_weights(model, ckpt_path, device)
    model.eval()

    loader = build_dataloader(data_cfg, split="val")

    n_seen = 0
    total_loss = 0.0
    for i, x in enumerate(loader):
        if max_batches is not None and i >= int(max_batches):
            break
        x = x.to(device, non_blocking=True)

        result = model.reconstruct(x, mask_ratio=mask_ratio)
        # Also compute reconstruction loss for reporting.
        target = model.patchify(x)
        loss = model.forward_loss(target, result["pred_patches"], result["mask"])

        torch.save(
            {
                "input":        x.detach().cpu(),
                "latent":       result["latent"].cpu(),
                "cls":          result["cls"].cpu(),
                "pred_patches": result["pred_patches"].cpu(),
                "recon":        result["recon"].cpu(),
                "mask":         None if result["mask"] is None else result["mask"].cpu(),
                "loss":         loss.item(),
            },
            out_dir / f"batch_{i:04d}.pt",
        )

        total_loss += loss.item() * x.size(0)
        n_seen += x.size(0)
        print(
            f"[infer] batch {i:4d}  x={tuple(x.shape)}  "
            f"latent={tuple(result['latent'].shape)}  "
            f"recon={tuple(result['recon'].shape)}  "
            f"loss={loss.item():.4f}"
        )

    if n_seen > 0:
        print(f"[infer] done.  avg_loss={total_loss / n_seen:.4f}  outputs -> {out_dir}")
    else:
        print("[infer] no batches processed.")


if __name__ == "__main__":
    CONFIG_PATH = "configs/config.yaml"

    main(config_path=CONFIG_PATH)
