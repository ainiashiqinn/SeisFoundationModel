"""
SeisFoundation pretraining entrypoint.

Edit the config path at the bottom of this file and run:
    python train.py
"""
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
from torch.optim import AdamW

# Ensure local-package imports work when run from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import build_dataloader  # noqa: E402
from models.foundation import SeisFoundation  # noqa: E402


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def pick_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        print("[train] MPS requested but unavailable; falling back to CPU.")
        return "cpu"
    return requested


def main(config_path: str, resume: Optional[str] = None):
    cfg = load_cfg(config_path)
    model_cfg = cfg["model"]
    data_cfg = {**model_cfg, **cfg.get("data", {})}
    train_cfg = cfg["train"]

    device = pick_device(train_cfg.get("device", "cpu"))

    loader = build_dataloader(data_cfg, split="train")

    model = SeisFoundation(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[train] input_type={model_cfg['input_type']} "
        f"use_mae={model_cfg.get('use_mae', True)} "
        f"mask_ratio={model_cfg.get('mask_ratio', 0.75)} "
        f"params={n_params/1e6:.2f}M device={device}"
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
        betas=tuple(train_cfg.get("betas", (0.9, 0.95))),
    )

    start_epoch = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[train] resumed from {resume} at epoch {start_epoch}")

    ckpt_dir = Path(train_cfg.get("ckpt_dir", "./checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_interval = int(train_cfg.get("log_interval", 10))
    grad_clip = train_cfg.get("grad_clip")

    model.train()
    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        running_loss = 0.0
        n_seen = 0
        for step, x in enumerate(loader):
            x = x.to(device, non_blocking=True)
            out = model(x)
            loss = out["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            n_seen += x.size(0)

            if step % log_interval == 0:
                print(f"[train] epoch {epoch} step {step:5d} loss {loss.item():.4f}")

        avg = running_loss / max(n_seen, 1)
        print(f"[train] epoch {epoch} avg_loss {avg:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg,
            },
            ckpt_dir / f"epoch_{epoch:03d}.pt",
        )
        # Encoder-only checkpoint -- the transferable part of the foundation model.
        torch.save(
            {
                "epoch": epoch,
                "encoder_state": model.encoder_state_dict(),
                "model_cfg": model_cfg,
            },
            ckpt_dir / "encoder_latest.pt",
        )


if __name__ == "__main__":
    CONFIG_PATH = "configs/config.yaml"
    RESUME = None  # set to a checkpoint path to resume

    main(config_path=CONFIG_PATH, resume=RESUME)
