"""
Unit 4 hands-on task: is a non-RoPE Block permutation-equivariant?

Run from the repo root:
    python claude/unit4_permutation_test.py
"""
import sys
from pathlib import Path

# make `models` importable when this file is run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models.transformer import Block


torch.manual_seed(0)

blk = Block(dim=256, num_heads=8).eval()

x = torch.randn(1, 5, 256)
perm = torch.tensor([3, 0, 4, 1, 2])

with torch.no_grad():
    y = blk(x)                       # run 1: original order
    yp = blk(x[:, perm, :])          # run 2: shuffled order

print("x        ", tuple(x.shape))
print("y        ", tuple(y.shape))
print("yp       ", tuple(yp.shape))
print()

# the claim: yp == y with its rows reordered by the same perm
print("yp == y[perm] :", torch.allclose(yp, y[:, perm, :], atol=1e-6))
print("  max abs diff:", (yp - y[:, perm, :]).abs().max().item())
print()

# control: compare against the UNreordered y
print("yp == y       :", torch.allclose(yp, y, atol=1e-6))
print("  max abs diff:", (yp - y).abs().max().item())
print()

# row-by-row, so you can see which output row matched which
for n in range(5):
    d = (yp[0, n] - y[0, perm[n]]).abs().max().item()
    print(f"  yp row {n}  vs  y row {perm[n].item()}   max diff {d:.2e}")
