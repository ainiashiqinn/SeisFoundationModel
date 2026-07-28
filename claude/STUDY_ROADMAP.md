# SeisFoundation — 2-Week Study Roadmap

**Goal:** explain every mathematical, deep-learning, and physical concept in this code and how it applies to seismic data — verbally, to an advisor. Not just run it.

**Your calibration (from the survey):**

- Target modality: **`multi_1d`** (angle gathers). The backbone is shared across all modalities, so most units transfer; modality-specific work concentrates in the patch-embed and position-encoding units.
- Position encodings: **all three** (`rope`, `sinusoidal`, `learnable`).
- Math scope: **strictly code-grounded.** No FFT, no PDE numerics, no linear inverse theory — the code implements none of them. If your advisor probes those, that's general geophysics, not this repo.
- Depth: **explain-and-use.** Working intuition, correct symbol↔variable mapping, know why each step exists, follow a derivation if shown. Not whiteboard-from-scratch.
- DL background: **new to deep learning** — a DL track runs underneath every unit.

**Budget:** 2 weeks × 15 h = **30 h.** Every unit lists an hour estimate and a week. Uncertain estimates are marked ⚠.

---

## What this roadmap cuts, and why

To fit 30 hours honestly:

- **No FFT / spectral analysis.** The repo never transforms to frequency. RoPE and sinusoidal encodings *use* sines, but that is not signal-domain Fourier. Skipped per your scope choice.
- **No inverse theory / PDE.** The only "inverse" here is MSE reconstruction. Skipped.
- **No from-scratch derivations.** You'll read and follow the attention and RoPE math, not re-derive it cold.
- **Light touch on the two non-target modalities** (`1d`, `2d`). You'll understand them by contrast in the patch-embed unit, not as separate study.
- **Trivia files** (`__init__.py`, `infer.py` plumbing) get one pass, no deep study.

If your advisor's real bar is "derive attention and RoPE from first principles," this plan is **~10 hours short** — tell me and I'll rebuild with fewer concepts covered more deeply.

---

## Three tracks, woven together

Each unit delivers three things at once:

- **DL** — the deep-learning concept, for a newcomer.
- **MATH** — the equation, symbols defined, assumptions stated.
- **SEISMIC** — what it means for angle-gather data physically.

The dependency ordering below is the *master* sequence; the DL, math, and seismic prerequisite chains from the survey all respect it.

```
U1 DL foundations + runnable skeleton
        │
        ▼
U2 Seismic data physics (Ricker + moveout + AVO)     ← independent, front-loaded for motivation
        │
        ▼
U3 Patchify + patch embedding (tokens, conv-as-projection)
        │
        ▼
U4 Self-attention + multi-head
        │
        ▼
U5 Position encodings — rope / sinusoidal / learnable
        │
        ▼
U6 Transformer block + encoder stack
        │
        ▼
U7 Masking + MAE mechanics (mask/CLS tokens, ids_restore)
        │
        ▼
U8 Reconstruction loss + normalization
        │
        ▼
U9 Training loop + optimizer + checkpoints + transfer
        │
        ▼
U10 Integration + exam prep (verification unit)
```

---

# WEEK 1 (15 h)

## Unit 1 — DL foundations + get it running (3 h)

The fastest path to a working skeleton, plus the DL vocabulary that makes the rest legible.

**Code in scope:** `train.py` (whole loop), `infer.py` (skim), `configs/config.yaml`. Set `input_type: 'multi_1d'`, leave `data_path: null` (synthetic).

**DL to learn:** tensors and shapes `(B, N, D)`; what `nn.Module`, a parameter, and a `forward` pass are; the shape of a training loop (epoch → step → forward → loss → `backward` → optimizer step); parameters vs hyperparameters (learned weights vs `config.yaml`). Intuition only — you'll meet each again in context.

**MATH to learn:** none beyond array bookkeeping. Confirm `num_patches = num_traces × (trace_length / patch_size) = 32 × (1024/16) = 2048` and token sequence length `= 2048 + 1` (CLS).

**Seismic connection:** the synthetic dataloader hands the model a `(B, 32, 1024)` batch — 32 traces (intended as reflection angles) × 1024 time samples per synthetic angle gather. Nothing physical happens to it yet; you're watching the plumbing move a gather from disk-shape to token-shape.

**Prerequisite reading:**

- Goodfellow, Bengio & Courville, *Deep Learning* (2016, free at deeplearningbook.org): Ch 5 (ML basics — the training-loop concept, generalization, hyperparameters).
- Skim `TUTORIAL.md` §1–§4 in this repo — it is accurate; use it.
- Optional, high-value for a newcomer: PyTorch official "Learn the Basics" tutorial (pytorch.org/tutorials) — tensors + a training loop end-to-end.

**Hands-on task:** run `python train.py` on synthetic `multi_1d`. Before you run, predict: (a) the token sequence length the encoder sees, (b) whether loss should decrease. Then add one print of `x.shape` and `out["loss"]` at step 0. Run 2–3 epochs. *Disagreement to watch:* if loss is flat or `nan`, you've mis-set a dim constraint — check `head_dim = embed_dim/num_heads = 256/8 = 32`, divisible by 4 (required for 2-axis RoPE).

**Checkpoint (cold-answerable):** "Walk me through what one training step does, in order, and name the tensor shape going into the model." You should answer without notes.

---

## Unit 2 — Seismic data physics: Ricker wavelet + moveout + AVO (3 h)

Front-loaded because it's independent of the ML and it's the part your advisor cares most about. This is where the "seismic" in the model actually lives — the *only* seismic physics in the repo is in this generator.

**Code in scope:** `data/dataset.py` — `_ricker` (line 36), `_gen_multi` (line 65, your target modality), and the joint-normalization comment. Contrast `_gen_1d`, `_gen_2d` briefly.

**DL to learn:** nothing new — this is a data unit. Note only that `SyntheticSeismicDataset` subclasses `torch.utils.data.Dataset` (`__len__`, `__getitem__`).

**MATH to learn (define every symbol):**

Ricker wavelet, as coded in `_ricker`:

$$ r(t) = \left(1 - 2\pi^2 f^2 (t-t_0)^2\right)\, e^{-\pi^2 f^2 (t-t_0)^2} $$

| symbol | code var | physical quantity | note |
|---|---|---|---|
| $t$ | `t` | (normalized) time axis, `linspace(0,1,T)` | **dimensionless here — not seconds** |
| $t_0$ | `t0` | wavelet center time | in [0,1] |
| $f$ | `f` | "frequency" 10–30 | **units are 1/normalized-time, not Hz** |
| $r$ | return value | amplitude | arbitrary units |

Assumption: zero-phase, symmetric wavelet (2nd derivative of a Gaussian). Real seismic wavelets are often *not* zero-phase; this synthetic one is.

Hyperbolic moveout, as coded in `_gen_multi`:

$$ \tau(x) = \sqrt{t_0^2 + \left(\tfrac{x}{v}\right)^2} $$

| symbol | code var | physical quantity |
|---|---|---|
| $\tau$ | `tau` | arrival time at trace/angle index | |
| $t_0$ | `t0` | zero-offset two-way time | |
| $x$ | `offsets[ci]` | offset proxy, `linspace(-1,1,C)` | **dimensionless** |
| $v$ | `v` | "apparent velocity" 0.5–2.0 | **dimensionless** |

**Flag the theory↔code gap:** the code labels the axis "angle" (angle gather) but the generator places events on a **hyperbolic moveout curve parameterized by offset**, which is the physics of a *common-midpoint / offset* gather, not a true angle gather (angle gathers are typically flat after migration). So the synthetic "angle gather" is really an offset gather in disguise. Worth saying out loud to your advisor. The noise term `0.05 * randn` and event counts are unjustified magic numbers.

**Seismic connection:** plotted, one `_gen_multi` sample is a small gather: 2–3 reflection events, each a Ricker wavelet smeared along a hyperbola that curves more at far offset (large `|x|`) and less for fast `v`. Joint normalization keeps the *relative* amplitude between near and far traces intact — that relative-amplitude-vs-offset behavior is the AVO signal a geophysicist would want preserved.

**Prerequisite reading:**

- Yilmaz, *Seismic Data Analysis* (2001): normal moveout / velocity analysis — I believe this is **Chapter 3**, but confirm the chapter title is "Velocity Analysis and Statics Corrections" in your edition rather than trusting the number.
- Sheriff & Geldart, *Exploration Seismology* (2nd ed., 1995): moveout and the Ricker wavelet (§4 on moveout, §9 on wavelets/processing — verify against your copy's TOC).
- AVO / angle gathers: Avseth, Mukerji & Mavko, *Quantitative Seismic Interpretation* (2005), the AVO chapter (Ch 4) — for why relative amplitude vs angle matters. Optional if AVO won't be examined.

**Hands-on task:** generate one `_gen_multi` sample and `plt.imshow` it. Before plotting, pick one event's `t0` and `v` and hand-compute `tau` at the near trace (`x≈0`) and the far trace (`x=1`). Predict which trace shows the event later. Compare to the image. *Disagreement means* you've mixed up the offset sign or the `t`-normalization.

**Checkpoint:** "Given `t0=0.3, v=0.5`, does the far-offset arrival come earlier or later than zero-offset, and by roughly how much in normalized time?" (Answer: later; $\tau=\sqrt{0.09+ (1/0.5)^2}=\sqrt{4.09}\approx2.02$ — clipped off the [0,1] axis, which itself tells you the synthetic `v` range is unphysically slow. Good thing to notice.)

---

## Unit 3 — Patchify + patch embedding (2.5 h)

How a gather becomes a sequence of tokens.

**Code in scope:** `models/patch_embed.py` — `PatchEmbedMulti1D` (forward, patchify, unpatchify); `foundation.py:patchify/unpatchify` delegates. Glance at `PatchEmbed1D`/`2D` for contrast only.

**DL to learn:** what "tokenizing" an input means; a **linear projection** (`nn.Linear`); **convolution as a layer** — here `nn.Conv2d(1, D, kernel=(1,16), stride=(1,16))` is a strided conv that acts as a per-patch linear projector, *not* a filter. Distinction between the **embedding** used by the encoder (`forward`, learned conv) and **patchify** used as the loss *target* (reshape, no weights).

**MATH to learn:** one patch of `patch_size=16` time samples on one trace → projected to a `D=256` vector: $z = W p + b$, where $p \in \mathbb{R}^{16}$ is the patch, $W \in \mathbb{R}^{256\times16}$. Token count for `multi_1d`: $N = A \cdot (T/P) = 32 \cdot 64 = 2048$; `patch_dim` $= P = 16$.

| symbol | code var | shape |
|---|---|---|
| $A$ | `num_traces` | 32 |
| $T$ | `trace_length` | 1024 |
| $P$ | `patch_size` | 16 |
| $N$ | `num_patches` | 2048 |
| token $k$ | — | angle $=k // 64$, time $=k \% 64$ (`foundation.py:_register_axis_positions`, lines 116–123) |

**Flag:** `patchify` (target) and `forward` (input embedding) are **two different operations** on the same data — one is a fixed reshape, the other a learned conv. The loss compares the decoder's output to the *reshape* target, never to the conv output.

**Seismic connection:** each token = one 16-sample time window on one trace/angle. Token grid is 32 angles × 64 time-patches. The model's finest resolution is 16 time samples; `patch_size` is the seismic "receptive field" per token.

**Prerequisite reading:**

- Dosovitskiy et al. 2021, *An Image Is Worth 16×16 Words* (ViT), arXiv:2010.11929, §3.1 (patch embedding). This code's patchify is the 1D/2D analog.
- Goodfellow Ch 9 (convolution) — only the "what stride and kernel do" part; ignore pooling.

**Hands-on task:** instantiate `PatchEmbedMulti1D(num_traces=32, trace_length=1024, patch_size=16, embed_dim=256)`, feed a `(2, 32, 1024)` tensor, and print the output of `forward` and of `patchify`. Predict both shapes first. (`forward` → `(2, 2048, 256)`; `patchify` → `(2, 2048, 16)`.)

**Checkpoint:** "Why does `patch_dim` equal 16 for `multi_1d` but `C·P` for `1d`? What does one token represent on the gather?"

---

## Unit 4 — Self-attention + multi-head (3.5 h)

The core mechanism. Budget the most time here — for a DL newcomer this is the hardest single idea.

**Code in scope:** `models/rope.py:RopeAttention.forward` (lines 77–89, your target since default is RoPE) and `models/transformer.py:Block` (`nn.MultiheadAttention`, the non-RoPE path). Ignore the `cos/sin` rotation lines for now — that's Unit 5.

**DL to learn:** query/key/value; **scaled dot-product attention**; **softmax** as a weighting; **multi-head** (split `D` into `H` heads of `head_dim`, attend independently, concatenate); why attention is **permutation-equivariant** without position info (motivates Unit 5). The `qkv` linear that produces all three at once.

**MATH to learn (follow, don't derive):**

$$ \text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V $$

| symbol | code | shape |
|---|---|---|
| $Q,K,V$ | `q,k,v` from `qkv(x)` unbind | each `(B, H, N, head_dim)` |
| $d_k$ | `head_dim` | 32 (=256/8) |
| $H$ | `num_heads` | 8 |
| softmax scaling | inside `F.scaled_dot_product_attention` | $1/\sqrt{d_k}$ applied internally |

Assumption: full (non-causal) attention — every token sees every token, no mask over the sequence (the MAE mask *drops* tokens before attention rather than masking the attention matrix).

**Flag:** `RopeAttention` uses `F.scaled_dot_product_attention` (fused, scaling internal) while `transformer.py:Block` uses `nn.MultiheadAttention`. Same math, two implementations — the model picks one based on `pos_embed_type`.

**Seismic connection:** attention lets a token at (far angle, time 0.4 s) directly weight a token at (near angle, same time) — the model can learn that an event's amplitude at one angle predicts its amplitude at another, i.e. it can learn AVO relationships across the gather without being told the gather geometry.

**Prerequisite reading:**

- Vaswani et al. 2017, *Attention Is All You Need*, arXiv:1706.03762, §3.2 (scaled dot-product + multi-head).
- Strongly recommended for explain-and-use depth: Jay Alammar, "The Illustrated Transformer" (jalammar.github.io). Not a citation, but the clearest visual for a newcomer.
- Optional hands-on: Karpathy, "Let's build GPT" (YouTube/nanoGPT) — builds attention in code.

**Hands-on task:** feed a `(1, 5, 256)` tensor to a `Block` (non-RoPE) and confirm output shape `(1, 5, 256)`. Then **shuffle the 5 tokens** and confirm the output is the same set of rows, reordered — demonstrating permutation-equivariance. *If it isn't,* you've accidentally left a position encoding on. This is the fact that makes Unit 5 necessary.

**Checkpoint:** "What does softmax over `QKᵀ/√d` produce, and what would happen to attention if you removed the `√d` scaling?" (Answer: a per-token probability distribution over other tokens; without scaling, large-`d` dot products push softmax into saturation and gradients vanish.)

---

## Unit 5 — Position encodings: all three (3 h)

Why the model needs position, and the three ways this code supplies it.

**Code in scope:** `models/rope.py` (whole file: `rotate_half`, `apply_rope`, `_rope_freqs`, `RopeCache.build`); `foundation.py:_sincos_1d`, `_build_sincos_pos_embed` (lines 163–191, sinusoidal); the `learnable` branch (`self.pos_embed` Parameter, lines 72–75); and how `forward_encoder` (lines 267–285) routes additive vs attention-time encodings.

**DL to learn:** absolute vs relative position; **additive** encodings (sinusoidal, learnable — added to tokens before the blocks) vs **attention-time** encodings (RoPE — rotates Q/K inside attention); learnable positional **parameters** vs fixed **buffers**; the factorized 2-axis idea (angle axis handled separately from time axis).

**MATH to learn (follow):**

Sinusoidal (`_sincos_1d`):
$$ PE(p, 2i)=\sin\!\big(p\,\omega_i\big),\quad PE(p,2i{+}1)=\cos\!\big(p\,\omega_i\big),\quad \omega_i = 10000^{-i/(\text{half})} $$

RoPE rotate-half (`apply_rope`): rotate each 2D sub-pair of a Q/K vector by angle $p\,\omega_i$:
$$ \tilde{x} = x\odot\cos(p\omega) + \text{rotate\_half}(x)\odot\sin(p\omega) $$
Key property (state it, don't prove): the dot product $\tilde{q}\cdot\tilde{k}$ depends only on the **relative** position $p_q - p_k$.

| symbol | code | meaning |
|---|---|---|
| $p$ | `pos` in `RopeCache.build` / `pos_axis_*` | token's index on an axis |
| $\omega_i$ | `_rope_freqs` output | per-dimension frequency |
| base | `rope_base=10000.0` | magic constant, inherited from RoFormer |
| axis 0 / axis 1 | `freqs_0`, `freqs_1` | **angle** axis / **time** axis (for `multi_1d`) |

**Flag:** for 2 axes, RoPE splits `head_dim` in half (`head_dim % 4 == 0`), sinusoidal splits `embed_dim` in half (`embed_dim % 4 == 0`). CLS token sits at position 0 on every axis; since $\cos 0=1, \sin 0=0$, RoPE is the identity there — CLS is deliberately unrotated (`foundation.py:_prepend_cls_positions`, line 153). The three modes produce **incompatible checkpoints** (TUTORIAL §8).

**Seismic connection:** RoPE's relative encoding means the model reacts to *differences* in angle and *differences* in time, not absolute indices — matching the physics, where the change in amplitude between angle 5° and 25° matters more than the absolute angle number. Learnable/absolute encodings lose that inductive bias.

**Prerequisite reading:**

- Vaswani et al. 2017, §3.5 (sinusoidal positional encoding).
- Su et al. 2024 (RoFormer), arXiv:2104.09864, §3 (RoPE). Read the *rotate-half formulation and the relative-position property*; you can skip the full theorem proof (explain-and-use).

**Hands-on task:** (a) call `RopeCache(head_dim=32, num_axes=2).build([pos0, pos1])` for a few positions and confirm at position 0 you get `cos=1, sin=0`. (b) Take two tokens, apply RoPE at positions (2,7) and again at (12,17) — same relative gap — and confirm their attention score is (nearly) identical. *Disagreement* means you've misunderstood relative invariance.

**Checkpoint:** "For `multi_1d`, which axis rotates the first half of `head_dim` and which the second? Why is CLS placed at position 0?"

---

# WEEK 2 (15 h)

## Unit 6 — Transformer block + encoder stack (2.5 h)

Assembling attention + MLP into the repeating unit, and stacking it.

**Code in scope:** `transformer.py:Block.forward`, `rope.py:RopeBlock.forward`; `foundation.py:forward_encoder` (lines 263–287), `_run_encoder_blocks`.

**DL to learn:** **LayerNorm** (and why it's not BatchNorm); **residual/skip connections**; **pre-norm** ordering (`x = x + attn(norm(x))`) vs post-norm and why pre-norm trains more stably; the **MLP** with GELU as the second half of the block; `depth` = number of stacked blocks.

**MATH to learn:** LayerNorm over the feature dim: $\hat{x}=\gamma\,\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$ with $\mu,\sigma^2$ computed per token over `D`. Residual: $x_{\ell+1}=x_\ell+F(x_\ell)$. No derivation needed.

**Seismic connection:** each of the 6 blocks refines the token representations; early blocks mix local time/angle structure, deeper blocks build gather-wide context. The stack is what turns raw patch projections into features that encode "this is a moveout event at this angle range."

**Prerequisite reading:**

- Goodfellow Ch 6 (feedforward nets, GELU-style activations) and the normalization discussion in Ch 8. LayerNorm itself: Ba, Kiros & Hinton 2016, arXiv:1607.06450 (skim the definition, §3).
- Vaswani 2017 §3.1 (the encoder block layout).

**Hands-on task:** count parameters with `sum(p.numel() for p in model.parameters())` and reconcile against the printed `params=…M` in `train.py`. Then set `depth: 1` vs `depth: 6` and confirm param count scales as expected. Predict the ratio before running.

**Checkpoint:** "In `x = x + self.attn(self.norm1(x), ...)`, what would break if you removed the `x +`? Why LayerNorm and not BatchNorm for sequence models?"

---

## Unit 7 — Masking + MAE mechanics (3.5 h)

The heart of the self-supervised paradigm and the trickiest bookkeeping in the repo.

**Code in scope:** `utils/masking.py:random_masking` (whole file); `foundation.py:forward_encoder` masking path (lines 271–277); `forward_decoder` mask-token reinsertion and `ids_restore` gather (lines 289–300); the `mask_token` and `cls_token` parameters.

**DL to learn:** **autoencoder** → **masked autoencoder**; **self-supervised / representation learning** (no labels — the data is its own target); the **mask token** (a learned placeholder for hidden patches) and **CLS token** (a learned summary); why the encoder sees only visible tokens (efficiency) and the decoder restores full length.

**MATH to learn:** masking is combinatorial, not analytic. `len_keep = max(1, int(N·(1−mask_ratio)))`; with `mask_ratio=0.75`, `N=2048` → keep 512. The `argsort(rand)` trick generates a random permutation per sample; `ids_restore = argsort(ids_shuffle)` is its inverse permutation. Understand: `gather(shuffled, ids_restore)` puts tokens back in canonical order.

| symbol | code | meaning |
|---|---|---|
| mask ratio | `mask_ratio` | 0.75 (magic, from MAE paper) |
| kept count | `len_keep` | 512 |
| `mask` | `mask` | `(B,N)`, 1=masked, 0=kept, in original order |
| `ids_restore` | `ids_restore` | inverse permutation |
| `ids_keep` | `ids_keep` | original indices of kept tokens (needed so RoPE rotates them by *true* position) |

**Flag:** the encoder is fed only kept tokens, so RoPE must rotate them by their **original** positions — that's why `ids_keep` is threaded through to `_kept_axes_positions` (lines 144–151). This coupling between masking and position encoding is the subtlest part of the codebase.

**Seismic connection:** the model hides 75% of the (angle, time) patches and learns to reconstruct them from the visible 25%. To succeed it must internalize gather structure — moveout continuity, wavelet shape, amplitude-vs-angle trends. That forced inference is *why* the pretrained encoder is a useful feature extractor.

**Prerequisite reading:**

- He et al. 2022, *Masked Autoencoders Are Scalable Vision Learners* (MAE), arXiv:2111.06377, §3 (masking strategy, encoder-on-visible, decoder with mask tokens). This repo is a near-direct 1D/gather port.

**Hands-on task:** call `random_masking` on a `(1, 8, 4)` tensor with `mask_ratio=0.5`. Verify: `mask.sum() == 4`, kept tokens = 4, and that gathering `[kept | mask_tokens]` by `ids_restore` returns canonical order. Then trace one masked token's index from `ids_shuffle` through `ids_restore` by hand. *If your hand-trace disagrees,* you've inverted the permutation.

**Checkpoint:** "Why does the code carry `ids_keep` separately from `ids_restore`? What goes wrong for RoPE if you rotate kept tokens by their *packed* index instead of their original index?"

---

## Unit 8 — Reconstruction loss + normalization (2.5 h)

What the model minimizes, and the two independent normalizations that confuse everyone.

**Code in scope:** `foundation.py:forward_loss` (lines 315–330); `utils/normalize.py` (whole file); the `norm_pix_loss` branch.

**DL to learn:** **loss function** (MSE); **reduction** (mean over patch dim, then masked mean over tokens); why loss is computed **only on masked patches**; **per-patch target normalization** (`norm_pix_loss`) vs **per-sample input normalization** (`normalize.py`) — two different operations at two different places.

**MATH to learn:**

$$ \mathcal{L} = \frac{\sum_{k} m_k \cdot \frac{1}{P}\sum_j (\hat{p}_{kj}-p_{kj})^2}{\sum_k m_k} $$

| symbol | code | meaning |
|---|---|---|
| $\hat p$ | `pred` | decoder output patches |
| $p$ | `target` | patchify(x), the reshape target |
| $m_k$ | `mask` | 1 on masked tokens only |
| $P$ | patch_dim | 16 |

`norm_pix_loss`: standardize each target patch to zero-mean/unit-var before the MSE (`(target-mean)/sqrt(var+1e-6)`).

Sample norm modes (`normalize.py`): `zscore`, `minmax`, `rms`, `max_abs`, `none` — all computed **jointly over the whole sample**.

**Flag the two normalizations (advisor bait):**

| | `data.normalize` | `model.norm_pix_loss` |
|---|---|---|
| where | at load, `normalize.py` | inside the loss, `forward_loss` |
| scope | whole sample | one patch |
| affects | what the model sees | gradient signal only |

`eps=1e-6` in both — magic constant.

**Seismic connection:** joint (not per-trace) sample normalization is a deliberate seismic choice — it preserves relative amplitude between near and far angles, i.e. **AVO information survives normalization**. `norm_pix_loss` makes the loss insensitive to per-patch contrast, which for gathers can wash out the very amplitude trends you care about — a reason `false` is the safe default for `multi_1d`.

**Prerequisite reading:**

- He et al. 2022 (MAE), the "reconstruction target" and "normalized pixels" paragraphs in §3.
- Goodfellow Ch 5.5 (MSE / maximum likelihood framing of squared error) — optional, for *why* MSE.

**Hands-on task:** build a `(1, 2, 4)` `pred` and `target` with a known difference and a `mask` of `[1, 0]`; hand-compute the masked MSE, then confirm `forward_loss` returns it. Toggle `norm_pix_loss` and predict whether loss goes up or down.

**Checkpoint:** "A colleague says `data.normalize: 'rms'` and `norm_pix_loss: true` are redundant. Correct them." (They're not — different scope, different stage, one changes inputs, the other changes only the gradient target.)

---

## Unit 9 — Training loop, optimizer, checkpoints, transfer (2.5 h)

Closing the loop and understanding why a "foundation model" is worth the trouble.

**Code in scope:** `train.py` (optimizer, loop, checkpointing, `encoder_state_dict`); `foundation.py:_init_weights`, `encode`, `reconstruct`; `infer.py:load_weights`; TUTORIAL §8–§9.

**DL to learn:** **autograd/backprop** (conceptual — `loss.backward()` fills `.grad`); **AdamW** (Adam + decoupled weight decay), learning rate, betas; **gradient clipping**; **weight init** (truncated-normal, Xavier); `state_dict` and **checkpointing**; **transfer learning** — freeze the encoder, attach a head, linear-probe or fine-tune; `train()`/`eval()` and `torch.no_grad()`.

**MATH to learn (intuition only):** AdamW update in words — per-parameter adaptive step from first/second gradient-moment estimates, with weight decay applied separately. No derivation. Grad clip: rescale the gradient vector if its norm exceeds `1.0`.

| symbol | code | meaning |
|---|---|---|
| lr | `lr=1e-4` | step size |
| betas | `[0.9, 0.95]` | moment decay (MAE values) |
| weight_decay | `0.05` | L2-ish regularization (MAE value) |
| grad_clip | `1.0` | max gradient norm |

**Flag:** two checkpoints saved per epoch — full (`epoch_NNN.pt`) and **encoder-only** (`encoder_latest.pt`). The encoder-only file is the whole point: after pretraining you discard the decoder and reuse the encoder. Loading encoder-only into `infer.py` leaves the decoder random → garbage reconstructions but valid `latent`/`cls` (documented, `infer.py:45–63`).

**Seismic connection:** the payoff. After label-free pretraining on many gathers, `encode(x)["cls"]` is a single vector summarizing a gather (for classification/regression — e.g. facies, fluid flag) and `encode(x)["tokens"]` gives per-(angle,time) features (for picking, denoising, segmentation). This is what you'd actually hand to a downstream seismic task.

**Prerequisite reading:**

- Goodfellow Ch 8 (optimization; SGD → momentum → Adam) and Ch 7.1 (weight decay as regularization). Kingma & Ba 2015 (Adam), arXiv:1412.6980 — skim §2 only. Loshchilov & Hutter 2019 (AdamW), arXiv:1711.05101 — the one-paragraph "decoupled decay" idea.
- TUTORIAL §8–§9 in this repo (accurate, use it).

**Hands-on task:** train 3 epochs, then load `encoder_latest.pt` and list its keys; confirm no `decoder_*` keys are present. Call `model.encode(x)` and check `cls` is `(B, 256)` and `tokens` is `(B, 2048, 256)`. Predict both shapes first.

**Checkpoint:** "Why does the repo save an encoder-only checkpoint separately? What exactly is transferred to a downstream task, and what is thrown away?"

---

## Unit 10 — Integration + exam prep (VERIFICATION UNIT) (4 h)

Prove you can trace the whole pipeline cold, and stress-test your understanding against the code's real flaws.

**Code in scope:** all of it, as one path.

**Task 1 — full-shape trace (1.5 h).** For a single `multi_1d` batch `(4, 32, 1024)`, write the tensor shape at every stage: after `normalize`, after patch-embed `forward`, after masking, after CLS prepend, through the encoder, after `decoder_embed`, after mask-token reinsertion, after decoder blocks, after `decoder_pred`, after dropping CLS, and the `patchify` target it's compared to. Then verify each by adding prints. Any mismatch is a gap in your mental model.

**Task 2 — confront the known issues (1 h).** Explain, out loud and in code terms: (a) the **broken train/val split** — `build_dataloader` only changes `shuffle`/`drop_last`; synthetic `seed=0` is fixed and a real `.npy` `train`/`val` would read the *same* directory, so validation re-sees training data. (b) The **undefined `2d` axis semantics** and the **offset-vs-angle mislabel** in the synthetic gather (Unit 2). (c) The **absence of `dt`/units** — the model has no notion of real seconds or Hz.

**Task 3 — magic-number census (0.5 h).** List every hard-coded constant with no derivation: `mask_ratio=0.75`, `rope_base=10000`, init std `0.02`, noise `0.05`, `eps=1e-6`, betas `[0.9,0.95]`, wd `0.05`. For each, state: inherited convention or arbitrary? (Most are inherited from MAE/RoFormer; noise `0.05` is arbitrary.)

**Task 4 — mock viva (1 h).** Answer cold, no notes: *What is this model, in one sentence? Why masked autoencoding and not supervised learning? What does one token represent physically? Why RoPE for gathers? Where is the only seismic physics in the repo? What would you fix before training on real data?*

**Checkpoint:** you can deliver Task 1's shape trace and Task 4's answers without looking at the code. If you can, you've met your advisor's bar.

---

# Question list for the PhD author (batch and send)

Things that cannot be recovered from the code or standard references:

1. Is `multi_1d` intended for **true angle gathers** (flat after migration) or **offset/CMP gathers**? The synthetic generator uses hyperbolic moveout in offset, which suggests offset gathers — was that deliberate?
2. What is the intended **`dt` / sampling interval and physical units** for real data? Nothing in the code fixes them. Are the synthetic "frequencies" (10–30) meant to map to Hz?
3. Is real data expected **pre- or post-migration**, NMO-corrected or raw? What preprocessing does the `.npy` loader assume is already done?
4. The **train/val split does nothing** (same data, only `shuffle`/`drop_last` differ). Is a real split planned, or was validation never intended in pretraining?
5. For `2d`, is axis 0 **time and axis 1 space**, or the reverse? The generator and the position code disagree in spirit — does it matter for your use?
6. Was this built to **reproduce a specific paper**, or is it an original recombination of ViT + MAE + RoPE? No source paper is referenced anywhere.
7. Any recommended **`num_traces` / `trace_length` / `patch_size`** for the real survey, or are the config defaults placeholders?
8. Is `norm_pix_loss=false` the intended default for `multi_1d`, given AVO amplitude concerns?

---

# Risk list — where estimates may double

- **Unit 4 (attention)** ⚠ — the single hardest concept for a DL newcomer. If Q/K/V doesn't click, everything downstream stalls. If you're shaky after 3.5 h, spend a full extra session here before Unit 5; it's the best-spent overtime in the plan.
- **Unit 7 (masking / `ids_restore`)** ⚠ — the permutation-inverse-and-gather bookkeeping is fiddly and easy to *think* you understand. The hand-trace task is the real test; budget extra if it disagrees.
- **Unit 5 (RoPE)** ⚠ — the relative-position property and 2-axis factorization are subtle. Explain-and-use keeps this bounded, but the interaction with masking (Unit 7) can force a re-visit.
- **Unit 2 (seismic physics)** — *lower* risk technically, but this is what your advisor weighs most, so the risk is under-preparing the *explanation*, not the code. Practice saying it aloud.
- **Environment setup** — you have no real data yet; if the PyTorch/synthetic run doesn't work in Unit 1, don't debug for hours. The synthetic path is self-contained; a failure is almost always a dim-constraint mismatch (Unit 1 hands-on note).

---

# Weekly review ritual

Run this prompt at the end of each week, against **everything covered so far**, not just the newest unit. Answer cold, then check against the code.

> **Weekly retention check.** Without opening the files:
> 1. Draw the pipeline from raw `(B, 32, 1024)` gather to loss, naming every stage and its output shape.
> 2. For each concept covered this week, give the one-sentence DL meaning, the equation (symbols defined), and the seismic meaning.
> 3. Pick one hard-coded constant and say whether it's principled or arbitrary.
> 4. State one thing about the code you now believe is wrong, sloppy, or undocumented — and why.
> 5. Name the single concept you're least sure of. That's next week's first re-read.
>
> Then open the code and grade yourself. Anything you got wrong goes to the top of next week.

End of Week 2, replace step 1 with the full Unit 10 shape-trace from memory — if you can produce it cold, you're exam-ready.

---

*Uncertainties flagged in-line with ⚠ and "verify" notes. Chapter numbers for Yilmaz / Sheriff & Geldart / Avseth are approximate — confirm against your edition's table of contents before relying on them. Everything about the code itself was read directly from the repo, not inferred, except where the survey marked otherwise.*
