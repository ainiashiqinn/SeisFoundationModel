# SeisFoundation — Tutorial

A unified ViT-style foundation model for seismic data. One backbone, three input
modalities (`1d` trace / `multi_1d` gather / `2d` patch), optional masked-autoencoder
pretraining — everything controlled from a single YAML.

---

## Contents

1. [Quickstart](#1-quickstart)
2. [Project layout](#2-project-layout)
3. [Config reference](#3-config-reference)
4. [Switching modality and MAE](#4-switching-modality-and-mae)
   - [Position embeddings — three options](#41-position-embeddings--three-options)
   - [CLS token](#42-cls-token)
5. [Architecture: making the model deeper / wider / smaller](#5-architecture-making-the-model-deeper--wider--smaller)
6. [Training hyperparameters](#6-training-hyperparameters)
7. [Using real data](#7-using-real-data)
   - [Normalization](#71-normalization)
8. [Checkpoints — full vs encoder-only](#8-checkpoints--full-vs-encoder-only)
9. [Using the trained encoder downstream](#9-using-the-trained-encoder-downstream)
10. [Extending the project](#10-extending-the-project)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Quickstart

**Requirements:** PyTorch, PyYAML, NumPy.

**Pretrain:**
```bash
python train.py
```
Reads `configs/config.yaml`, builds the model + a (synthetic) dataloader, writes
checkpoints to `./checkpoints/`.

**Inference:**
```bash
python infer.py
```
Loads `inference.ckpt_path` from the same YAML, runs encoder + decoder on the
dataloader, writes per-batch `.pt` files under `inference_out/`.

Both scripts have the config path hard-coded near the bottom of the file:
```python
if __name__ == "__main__":
    CONFIG_PATH = "configs/config.yaml"
    ...
```
Edit that line if you move the YAML or want to maintain multiple configs.

---

## 2. Project layout

```
SeisFoundation/
├── configs/
│   └── config.yaml             # single config -- switches and hyperparams
├── models/
│   ├── transformer.py          # pre-norm Transformer block (learnable-pos path)
│   ├── rope.py                 # Rotary Position Embedding + RopeAttention block
│   ├── patch_embed.py          # PatchEmbed1D / PatchEmbedMulti1D / PatchEmbed2D
│   └── foundation.py           # SeisFoundation -- encoder + decoder + MAE
├── data/
│   └── dataset.py              # SyntheticSeismicDataset, NumpySeismicDataset
├── utils/
│   ├── masking.py              # random_masking (MAE-style)
│   └── normalize.py            # per-sample input normalization (zscore / rms / max_abs / ...)
├── train.py                    # pretraining entrypoint
└── infer.py                    # inference entrypoint
```

The Transformer encoder and decoder are **shared across all three modalities**. The
only modality-specific piece is the patch embedding in `models/patch_embed.py`.

---

## 3. Config reference

All keys live in `configs/config.yaml`. Below is every key with its meaning and
which modality / mode uses it.

### `model:` — architecture and switches

| key                  | type        | used by         | meaning                                                                 |
|----------------------|-------------|-----------------|-------------------------------------------------------------------------|
| `input_type`         | str         | all             | `'1d'` / `'multi_1d'` / `'2d'`                                          |
| `use_mae`            | bool        | all             | `true` = masked autoencoder; `false` = vanilla autoencoder              |
| `mask_ratio`         | float (0–1) | MAE             | fraction of patches hidden from the encoder (typical: 0.5–0.85)         |
| `norm_pix_loss`      | bool        | all             | per-patch normalize the reconstruction target (recommended for `2d`)   |
| `pos_embed_type`     | str         | all             | `'rope'` (default) / `'sinusoidal'` (fixed sin/cos) / `'learnable'`     |
| `rope_base`          | float       | RoPE            | RoPE base frequency (default `10000.0`, rarely worth tuning)            |
| `trace_length`       | int         | 1d, multi_1d    | number of time samples per trace; must be divisible by `patch_size`     |
| `num_traces`         | int         | multi_1d        | number of traces per sample (e.g. shot gather)                          |
| `img_size`           | int or [H,W]| 2d              | spatial size; each dim divisible by the matching `patch_size` dim       |
| `patch_size`         | int or [ph,pw] | all          | patch length (1d/multi_1d) or `(ph, pw)` (2d)                           |
| `in_channels`        | int         | 1d, 2d          | input channels (usually 1 for seismic)                                  |
| `embed_dim`          | int         | all             | encoder token width                                                     |
| `depth`              | int         | all             | number of encoder Transformer blocks                                    |
| `num_heads`          | int         | all             | encoder attention heads (must divide `embed_dim`)                       |
| `mlp_ratio`          | float       | all             | MLP hidden dim = `embed_dim * mlp_ratio`                                |
| `decoder_embed_dim`  | int         | all             | decoder token width (often `embed_dim/2`)                               |
| `decoder_depth`      | int         | all             | number of decoder blocks                                                |
| `decoder_num_heads`  | int         | all             | decoder attention heads (must divide `decoder_embed_dim`)               |

Keys for modalities you aren't using are simply ignored — leave them in the file.

### `data:`

| key            | type          | meaning                                                                  |
|----------------|---------------|--------------------------------------------------------------------------|
| `data_path`    | str or null   | directory of `.npy` files. `null` → use the synthetic dataset.           |
| `num_samples`  | int           | synthetic dataset size                                                   |
| `batch_size`   | int           | batch size                                                               |
| `num_workers`  | int           | DataLoader workers                                                       |
| `pin_memory`   | bool          | pin host memory (good for GPU)                                           |
| `normalize`    | str           | per-sample normalization at load time. See [§7.1](#71-normalization).    |

### `train:`

| key             | type           | meaning                                                                 |
|-----------------|----------------|-------------------------------------------------------------------------|
| `epochs`        | int            | number of epochs                                                        |
| `lr`            | float          | AdamW base learning rate                                                |
| `weight_decay`  | float          | AdamW weight decay (MAE paper uses 0.05)                                |
| `betas`         | [β1, β2]       | AdamW betas (MAE uses `[0.9, 0.95]`)                                    |
| `grad_clip`     | float or null  | gradient-norm clip; `null` to disable                                   |
| `device`        | str            | `'cpu'` / `'cuda'` / `'mps'` (falls back to CPU if unavailable)         |
| `log_interval`  | int            | print loss every N steps                                                |
| `ckpt_dir`      | str            | where to save checkpoints                                               |

### `inference:`

| key            | type           | meaning                                                                  |
|----------------|----------------|--------------------------------------------------------------------------|
| `ckpt_path`    | str            | path to a full checkpoint (`epoch_NNN.pt`) or encoder-only (`encoder_latest.pt`) |
| `out_dir`      | str            | where to write per-batch `.pt` outputs                                   |
| `mask_ratio`   | float          | 0 → full reconstruction; >0 → emulate MAE at inference time              |
| `max_batches`  | int or null    | cap number of batches processed; `null` = all                            |

---

## 4. Switching modality and MAE

Flip two lines:
```yaml
model:
  input_type: 'multi_1d'    # '1d' | 'multi_1d' | '2d'
  use_mae:    true          # true | false
```

Each modality expects a specific input tensor shape from the dataloader:

| `input_type` | input shape          | tokens from encoder        | what each axis represents              |
|--------------|----------------------|----------------------------|----------------------------------------|
| `1d`         | `(B, 1, T)`          | `(B, T/patch + 1, D)`      | time                                   |
| `multi_1d`   | `(B, A, T)`          | `(B, A·T/patch + 1, D)`    | **angle** (or trace) × time            |
| `2d`         | `(B, 1, H, W)`       | `(B, (H/ph)·(W/pw) + 1, D)`| row × col                              |

`multi_1d` is designed for **angle gathers**: same subsurface location, different
reflection angles. `num_traces` is the size of the angle axis. With RoPE on, the
angle axis and time axis rotate independently — the model sees relative angle
differences naturally (small angle ↔ near offset, large angle ↔ far offset).

`use_mae=false` turns off masking entirely: the encoder sees every patch and the
decoder reconstructs every patch (plain ViT autoencoder).

---

## 4.1 Position embeddings — three options

`model.pos_embed_type` selects how positions are encoded:

| mode               | what it is                                                                                                                    | params | resolution-flex | when to prefer                                                  |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------|--------|-----------------|-----------------------------------------------------------------|
| `'rope'` (default) | Rotary embeddings — position is encoded by rotating Q/K inside attention. **Relative**.                                       | 0      | yes             | variable trace/image sizes; physical signals where Δposition matters more than absolute index |
| `'sinusoidal'`     | Fixed sin/cos absolute embedding (Vaswani et al. 2017; factorized 2D for `multi_1d`/`2d`). **Added** to tokens before encoder. | 0      | partial (interpolation needed) | parameter-free baseline; reproducing classic Transformer setups |
| `'learnable'`      | Learnable absolute embedding (`(1, N+1, embed_dim)` parameter, separate for decoder). **Added** to tokens.                    | ~2·N·D | no              | reproducing MAE-paper numbers; fixed-resolution data            |

### How each handles 2 axes (`multi_1d`, `2d`)

| mode          | 2-axis behavior                                                                                       |
|---------------|-------------------------------------------------------------------------------------------------------|
| `'rope'`      | Factorized: head_dim split in half — first half rotates with axis 0, second half with axis 1.         |
| `'sinusoidal'`| Factorized: embed_dim split in half — first half is sin/cos of axis 0 index, second half of axis 1.   |
| `'learnable'` | Flat 1D over all `N` tokens — model learns the 2D layout from a 1D index. No inductive bias for axes. |

For `multi_1d` axis 0 is **angle**, axis 1 is **time**. For `2d` axis 0 is **row**, axis 1 is **column**.

### Dim constraints

| mode          | 1d                       | multi_1d / 2d            |
|---------------|--------------------------|--------------------------|
| `'rope'`      | `head_dim % 2 == 0`      | `head_dim % 4 == 0`      |
| `'sinusoidal'`| `embed_dim % 2 == 0`     | `embed_dim % 4 == 0`     |
| `'learnable'` | (none — works for any)   | (none — works for any)   |

Same constraints apply to the decoder pair (`decoder_embed_dim / decoder_num_heads` for RoPE, `decoder_embed_dim` for sinusoidal).

## 4.2 CLS token

`CLS` is a single **learnable** vector (one slot, prepended to the patch sequence at index 0) used as a *summary* of the whole sample. It carries no input data; through self-attention it aggregates information from every patch, and its final encoded state is the model's sample-level feature.

- Defined at [foundation.py](models/foundation.py): `self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))`
- Returned by `model.encode(x)["cls"]` with shape `(B, embed_dim)`
- With RoPE, CLS is placed at position 0 on every axis. Since `cos(0)=1, sin(0)=0`, RoPE is the identity at position 0 — CLS is unrotated, which preserves its "summary" role.

Use CLS for tasks that want one vector per sample (classification, regression, retrieval). Use the patch tokens (`encode(x)["tokens"]`) when you need per-location predictions.

## 5. Architecture — making the model deeper / wider / smaller

Five knobs control model capacity:

```yaml
embed_dim:         256   # token width (encoder)
depth:             6     # encoder blocks
num_heads:         8     # encoder heads  (must divide embed_dim)
decoder_embed_dim: 128   # token width (decoder)
decoder_depth:     2     # decoder blocks
decoder_num_heads: 4     # decoder heads  (must divide decoder_embed_dim)
mlp_ratio:         4.0   # MLP hidden = embed_dim * mlp_ratio
```

**Make it deeper** → raise `depth` (and optionally `decoder_depth`).
Deeper models help when you have lots of data; compute and memory grow linearly in depth.

**Make it wider** → raise `embed_dim` (and `decoder_embed_dim`).
Wider models help when patches are information-rich (large `patch_size` / 2D).
Memory grows ~linearly, compute grows ~quadratically in width.

**Make it smaller (e.g. for CPU smoke tests)** → cut all four:
```yaml
embed_dim: 128
depth: 4
num_heads: 4
decoder_embed_dim: 64
decoder_depth: 2
decoder_num_heads: 2
```

**Rules of thumb (MAE-style sizing):**
- Decoder is intentionally smaller than encoder: `decoder_embed_dim ≈ embed_dim / 2`,
  `decoder_depth ≈ depth / 4`. The decoder is discarded after pretraining.
- `num_heads` must divide `embed_dim`. Same for the decoder pair.
- `mlp_ratio=4.0` is the standard ViT choice; rarely worth changing.
- If using **RoPE**: `head_dim = embed_dim/num_heads` must be divisible by 2 (`1d`) or 4 (`multi_1d`/`2d`).
- If using **sinusoidal**: `embed_dim` must be divisible by 2 (`1d`) or 4 (`multi_1d`/`2d`). Same for `decoder_embed_dim`. See [§4.1](#41-position-embeddings--three-options).

**Canonical ViT / MAE presets (legacy reference sizes):**

The encoder side follows the established ViT presets from
Dosovitskiy et al. 2020 (ViT) / Touvron et al. 2020 (DeiT) / He et al. 2022 (MAE).
Almost every ViT-style paper since 2020 reports numbers against these.

| preset      | `depth` | `embed_dim` | `num_heads` | head_dim | `mlp_ratio` | encoder params |
|-------------|---------|-------------|-------------|----------|-------------|----------------|
| ViT-Pico    | 6       | 192         | 3           | 64       | 4.0         | ~3 M           |
| ViT-Tiny    | 12      | 192         | 3           | 64       | 4.0         | ~5.7 M         |
| ViT-Small   | 12      | 384         | 6           | 64       | 4.0         | ~22 M          |
| ViT-Base    | 12      | 768         | 12          | 64       | 4.0         | ~86 M          |
| ViT-Large   | 24      | 1024        | 16          | 64       | 4.0         | ~307 M         |
| ViT-Huge    | 32      | 1280        | 16          | 80       | 4.0         | ~632 M         |
| (shipped default) | 6 | 256       | 8           | 32       | 4.0         | ~5 M           |

All canonical presets satisfy the position-embedding dim constraints from
[§4.1](#41-position-embeddings--three-options) (`head_dim` divisible by 4,
`embed_dim` divisible by 4) — so they work with `rope`, `sinusoidal`, or
`learnable` without modification.

**Matching decoder (MAE-paper recipe).** The MAE paper uses the *same* small
decoder regardless of encoder size — the decoder is intentionally lightweight
and is discarded after pretraining:

| `decoder_depth` | `decoder_embed_dim` | `decoder_num_heads` |
|-----------------|---------------------|---------------------|
| 8               | 512                 | 16                  |

You can scale the decoder down further for CPU experiments
(`decoder_depth: 2, decoder_embed_dim: 128, decoder_num_heads: 4` — the current default).

**Drop a preset into the YAML.** For example, ViT-Base + MAE decoder:

```yaml
model:
  embed_dim:         768
  depth:             12
  num_heads:         12
  mlp_ratio:         4.0
  decoder_embed_dim: 512
  decoder_depth:     8
  decoder_num_heads: 16
```

Or the (smaller) DeiT-Tiny preset for CPU smoke tests:

```yaml
model:
  embed_dim:         192
  depth:             12
  num_heads:         3
  mlp_ratio:         4.0
  decoder_embed_dim: 96
  decoder_depth:     4
  decoder_num_heads: 3
```

### Tuning the *receptive field*: `patch_size`

`patch_size` controls how many raw samples each token covers — independent of model width/depth.

- **Smaller `patch_size`** → more tokens → finer detail, more compute (attention is
  O(N²) in token count). For 2D, halving `patch_size` quadruples token count.
- **Larger `patch_size`** → fewer tokens → coarser but cheap. For long traces or
  big 2D patches, raise `patch_size` before raising `embed_dim`.

Constraint: `trace_length % patch_size == 0` (1D / multi_1d) and
`img_size[i] % patch_size[i] == 0` (2D).

---

## 6. Training hyperparameters

### Learning rate
- Pretraining default `lr: 1e-4` works for the tiny / small presets on CPU/GPU.
- For larger batch sizes use the linear-scaling rule: `lr ∝ batch_size`.
- The MAE paper uses `lr = base_lr * batch_size / 256` with `base_lr = 1.5e-4`.

### Batch size
Pick the largest your device fits. CPU is fine at 4–32; a single GPU usually
handles 64–256 for the tiny / small presets.

### Epochs
- Synthetic data: 20–50 epochs is plenty to see loss go down.
- Real data: MAE typically wants long pretraining (~800 epochs in the paper).
  For smaller datasets, 100–400 is a reasonable start.

### `mask_ratio`
- `0.75` (default) follows the MAE paper for 2D images and works well for `2d`.
- For 1D seismic traces, try `0.5–0.75`.
- For very dense data (high-redundancy gathers), higher (`0.85–0.9`) can work.

### `norm_pix_loss`
- `true` for `2d`: makes the loss invariant to per-patch contrast — usually helps.
- `false` for `1d` / `multi_1d` is a safe default unless you see amplitude drift.

### `grad_clip`
- `1.0` is a safe default. Disable (`null`) if you never see large gradient spikes.

### `weight_decay` and `betas`
- `weight_decay: 0.05`, `betas: [0.9, 0.95]` — copy from MAE; rarely worth tuning.

---

## 7. Using real data

Drop one `.npy` file per sample in a directory and set:
```yaml
data:
  data_path: /path/to/dir
```

Each file's shape must match the configured `input_type`:

| `input_type` | accepted shape per file       |
|--------------|-------------------------------|
| `1d`         | `(T,)` or `(1, T)`            |
| `multi_1d`   | `(A, T)` — A = angle/trace axis (must match `num_traces`) |
| `2d`         | `(H, W)` or `(1, H, W)`       |

Other formats (SEG-Y, HDF5, MAT) are not built in. The easiest path is to
write a one-time conversion script that dumps your data to `.npy`. If you'd
rather plug in a custom loader, subclass `torch.utils.data.Dataset` and edit
`data/dataset.py:build_dataloader` to use it.

## 7.1 Normalization

Set `data.normalize` to apply per-sample normalization at load time:

```yaml
data:
  normalize: 'rms'   # 'none' | 'zscore' | 'minmax' | 'rms' | 'max_abs'
```

| mode        | formula                                       | output range / property            | preserves zero? |
|-------------|-----------------------------------------------|------------------------------------|-----------------|
| `'none'`    | identity                                      | unchanged                          | yes             |
| `'zscore'`  | `(x - mean) / std`                            | mean 0, std 1                      | no              |
| `'minmax'`  | `(x - min) / (max - min)`                     | `[0, 1]`                           | no              |
| `'rms'`     | `x / sqrt(mean(x^2))`                         | unit RMS                           | yes             |
| `'max_abs'` | `x / max(|x|)`                                | `[-1, 1]`                          | yes             |

**All modes compute stats jointly over the whole sample**, so for `multi_1d`
angle gathers the **relative amplitudes between traces are preserved** —
AVO information is not destroyed. (If you want per-trace normalization
instead, do it in your own dataset before yielding.)


**Not the same as `model.norm_pix_loss`.** Two independent normalizations:

| where                  | when applied                  | scope            | affects        |
|------------------------|-------------------------------|------------------|----------------|
| `data.normalize`       | per sample, at load time      | whole sample     | what the model sees |
| `model.norm_pix_loss`  | per patch, inside the loss    | one patch        | gradient signal only |

You can combine them: e.g. `data.normalize: 'rms'` + `model.norm_pix_loss: true`
gives RMS-scaled inputs with per-patch-normalized reconstruction targets.

---

## 8. Checkpoints — full vs encoder-only

Each epoch, `train.py` writes two files into `ckpt_dir`:

| file                       | contents                                                | use for                       |
|----------------------------|---------------------------------------------------------|-------------------------------|
| `epoch_NNN.pt`             | `model_state`, `optimizer_state`, `epoch`, `config`     | resume training, full inference |
| `encoder_latest.pt`        | `encoder_state` — patch_embed + cls_token + encoder_blocks + encoder_norm + position-encoding state | downstream transfer            |

The encoder-only file is the transferable artifact — after pretraining, throw
the decoder away.

The exact position-encoding keys saved depend on `pos_embed_type`:

| pos_embed_type | keys included in `encoder_state`                                            |
|----------------|------------------------------------------------------------------------------|
| `'rope'`       | `pos_axis_0`, `pos_axis_1` (buffers), `encoder_rope.freqs_*` (non-persistent — re-derived on construct) |
| `'sinusoidal'` | `pos_embed` (buffer, precomputed), `pos_axis_0`, `pos_axis_1`                |
| `'learnable'`  | `pos_embed` (Parameter), `pos_axis_0`, `pos_axis_1`                          |

When loading, instantiate `SeisFoundation` with the *same* `pos_embed_type` as
the training config — the keys don't translate across modes.

`infer.py` accepts either: it autodetects the key (`model_state` vs
`encoder_state`). If you pass an encoder-only file, the decoder will be random
and the reconstruction will look like noise — but the `latent` / `cls` outputs
are still meaningful.

**Resume training:**
```python
if __name__ == "__main__":
    CONFIG_PATH = "configs/config.yaml"
    RESUME = "checkpoints/epoch_004.pt"     # was None
    main(config_path=CONFIG_PATH, resume=RESUME)
```

---

## 9. Using the trained encoder downstream

Two encoder outputs cover most tasks:

| You want…                                       | Use                                    | Shape              |
|-------------------------------------------------|----------------------------------------|--------------------|
| one prediction per sample (classify / regress)  | `encoder.encode(x)["cls"]`             | `(B, embed_dim)`   |
| one prediction per patch / region / time-window | `encoder.encode(x)["tokens"]`          | `(B, N, embed_dim)`|

**Load the encoder:**
```python
import torch, yaml
from models.foundation import SeisFoundation

cfg   = yaml.safe_load(open("configs/config.yaml"))
model = SeisFoundation(cfg["model"])

ckpt = torch.load("checkpoints/encoder_latest.pt", map_location="cpu")
model.load_state_dict(ckpt["encoder_state"], strict=False)   # decoder keys missing -- expected
encoder = model.eval()

# Freeze for linear-probe / feature extraction:
for p in encoder.parameters():
    p.requires_grad = False
```

**Global feature (classification, regression):**
```python
import torch.nn as nn
head = nn.Linear(cfg["model"]["embed_dim"], num_classes)
feat = encoder.encode(x)["cls"]    # (B, embed_dim)
logits = head(feat)
```

**Per-patch feature (denoising, segmentation, picking):**
```python
tokens = encoder.encode(x)["tokens"]   # (B, N, embed_dim)
head   = nn.Linear(cfg["model"]["embed_dim"], out_dim_per_patch)
out    = head(tokens)                  # (B, N, out_dim)
```

**Reshape patch features back to the input grid** (for FCN / U-Net heads):
```python
tokens = encoder.encode(x)["tokens"]
B, N, D = tokens.shape

# 1d:
feat = tokens.transpose(1, 2)                                       # (B, D, n_patches)

# multi_1d:
nt = encoder.patch_embed.num_traces
tp = encoder.patch_embed.time_patches
feat = tokens.reshape(B, nt, tp, D).permute(0, 3, 1, 2)             # (B, D, nt, tp)

# 2d:
gh, gw = encoder.patch_embed.grid_h, encoder.patch_embed.grid_w
feat = tokens.transpose(1, 2).reshape(B, D, gh, gw)                 # (B, D, gh, gw)
```

Downsampling factor between the input grid and the token grid is `patch_size`.

---

## 10. Extending the project

### Add a new input modality
1. Add a `PatchEmbedXXX` class to `models/patch_embed.py` exposing
   `forward(x) -> (B, N, embed_dim)`, `patchify(x) -> (B, N, patch_dim)`,
   `unpatchify(p) -> input_shape`, and attributes `num_patches`, `patch_dim`.
2. Register a new branch in `build_patch_embed()`.
3. Add a generator branch in `data/dataset.py:SyntheticSeismicDataset` and
   (if needed) adapt `NumpySeismicDataset` for the new shape.

### Replace the decoder
The decoder lives entirely in `models/foundation.py` (`decoder_*` attributes
and `forward_decoder`). You can replace it with a CNN, U-Net, or anything
that maps `(B, N+1, decoder_embed_dim)` to `(B, N, patch_dim)`.
Nothing else in the codebase touches the decoder.

### Add a downstream head
Build a separate model (don't edit `SeisFoundation`):
```python
class Downstream(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head
    def forward(self, x):
        feat = self.encoder.encode(x)["cls"]   # or ["tokens"]
        return self.head(feat)
```

### Add a hard bottleneck (fixed K-dim per sample)
Bolt a linear layer onto the CLS token:
```python
self.bottleneck = nn.Linear(embed_dim, K)
z = self.bottleneck(encoder.encode(x)["cls"])     # (B, K)
```
This is a downstream-task design, not a foundation-model change.

### Add a new position-encoding mode
There are already three modes (`rope`, `sinusoidal`, `learnable`) and switching
between them is a YAML edit — see [§4.1](#41-position-embeddings--three-options).
To add a fourth (e.g. ALiBi, relative bias, learned-1D-per-axis):

1. Add a branch in `SeisFoundation.__init__` for your new `pos_embed_type` value.
   Build whatever buffers / parameters / cache the scheme needs.
2. If the scheme is **additive** (like `sinusoidal`/`learnable`), populate
   `self.pos_embed` and `self.decoder_pos_embed`; the existing forward path
   already handles them.
3. If the scheme is **attention-time** (like `rope`), reuse the `RopeBlock`
   pattern: build a cache that produces per-token modulation tensors and pass
   them into a custom attention block.
4. Update the assertion in `__init__` to accept the new name.
5. Add the new option to `pos_embed_type` in `configs/config.yaml`.

---

## 11. Troubleshooting

**`AssertionError: trace_length must be divisible by patch_size`**
Pick a `patch_size` that divides your `trace_length` (or `img_size`). For 2D,
both spatial dims must be divisible.

**`embed_dim must be divisible by num_heads`** (from `nn.MultiheadAttention` / `RopeAttention`)
Set `num_heads` so it divides `embed_dim`. Same constraint for the decoder
pair. Typical: `embed_dim=256, num_heads=8` or `embed_dim=384, num_heads=12`.

**`head_dim must be divisible by 2 / 4 for RoPE`**
Triggered when `pos_embed_type: 'rope'` and `head_dim = embed_dim / num_heads` doesn't divide cleanly. Need:
- `head_dim % 2 == 0` for `1d`
- `head_dim % 4 == 0` for `multi_1d` / `2d`

Either pick a different `num_heads`, or switch to `pos_embed_type: 'learnable'` (no head_dim constraint).

**`sinusoidal 2-axis encoding needs dim % 4 == 0`**
Triggered when `pos_embed_type: 'sinusoidal'` on `multi_1d`/`2d` with `embed_dim` (or `decoder_embed_dim`) not divisible by 4. Bump `embed_dim` to the next multiple of 4 or switch to RoPE/learnable.

**Loaded checkpoint has keys like `pos_axis_0`, `encoder_rope.freqs_0` that aren't recognized**
You're loading a checkpoint trained with a *different* `pos_embed_type`. The three modes have incompatible state-dict layouts (see [§8](#8-checkpoints--full-vs-encoder-only)). Match the config's `pos_embed_type` to the one used at training time.

**CUDA OOM**
- Lower `batch_size`
- Raise `patch_size` (fewer tokens → much less attention memory)
- Lower `embed_dim` or `depth`
- For `multi_1d`, lower `num_traces` or `trace_length`

**Loss is `nan` immediately**
- Check input normalization — extreme outliers can blow up attention.
- Try `grad_clip: 1.0` (the default).
- Lower the learning rate by 10×.

**Encoder-only checkpoint warns about missing keys**
Expected. The "missing" keys are the decoder (`decoder_*`, `mask_token`).
Reconstruction quality through `infer.py` will be poor with random decoder;
use a full checkpoint for visual reconstruction. The encoder latent is still
correct.

**Inference reconstruction looks wrong even with full checkpoint**
- If `norm_pix_loss: true` was used during training, reconstructions live in
  the per-patch *normalized* space — they won't visually match the raw input.
  Disable `norm_pix_loss` if you want pixel-faithful reconstructions.
- Ensure `model.eval()` is called (`infer.py` already does this; only matters
  if you call the model manually).
