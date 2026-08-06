"""
Unit 4 follow-up: what happens to permutation equivariance with RoPE?

Same experiment as unit4_permutation_test.py, but with RopeBlock instead of Block.
RopeBlock.forward takes (x, cos, sin), so shuffling x forces a choice:
  run 2a -- shuffle x, leave cos/sin alone   (position stays with the SLOT)
  run 2b -- shuffle x, cos and sin together  (position travels with the TOKEN)

Run from the repo root:
    python claude/unit4_permutation_test_rope.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models.rope import RopeBlock, RopeCache


torch.manual_seed(0)

DIM, HEADS = 256, 8
HEAD_DIM = DIM // HEADS          # 32

blk = RopeBlock(dim=DIM, num_heads=HEADS).eval()

# 5 toy tokens laid out like a tiny multi_1d gather:
#   angle 0 at times 0,1,2   and   angle 1 at times 0,1
pos_angle = torch.tensor([[0, 0, 0, 1, 1]])      # (1, 5)  axis 0
pos_time = torch.tensor([[0, 1, 2, 0, 1]])       # (1, 5)  axis 1

rope = RopeCache(head_dim=HEAD_DIM, num_axes=2)
cos, sin = rope.build([pos_angle, pos_time])     # each (1, 5, 32)

x = torch.randn(1, 5, DIM)
perm = torch.tensor([3, 0, 4, 1, 2])

with torch.no_grad():
    y = blk(x, cos, sin)                                        # run 1
    yp_slot = blk(x[:, perm, :], cos, sin)                      # run 2a
    yp_both = blk(x[:, perm, :], cos[:, perm, :], sin[:, perm, :])   # run 2b

target = y[:, perm, :]

print("cos      ", tuple(cos.shape))
print("cos[0, :, 0] =", cos[0, :, 0].tolist())   # first freq, one value per token
print()

print("run 2a  (cos/sin NOT shuffled) == y[perm] :",
      torch.allclose(yp_slot, target, atol=1e-6))
print("        max abs diff:", (yp_slot - target).abs().max().item())
print()

print("run 2b  (cos/sin shuffled too) == y[perm] :",
      torch.allclose(yp_both, target, atol=1e-6))
print("        max abs diff:", (yp_both - target).abs().max().item())
