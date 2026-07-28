# Unit 1 Self-Test — Training Loop, Parameters vs Hyperparameters, Math Intuition

Cover the answer key. Answer each cold, then scroll down to check.

---

## A. Epoch, step, batch

1. Define epoch, step, and batch in terms of `train.py`'s loop structure.
2. Computational: a dataset has 400 samples, `batch_size=32`. How many steps per epoch (ignore `drop_last`)?
3. In `train.py`, which line starts the epoch loop, and which line starts the step loop?

## B. One training step, in order

4. List the stages of one training step in `train.py` (lines 81–89), in order, each with its line number.
5. What would break if `optimizer.zero_grad()` were skipped entirely across steps?
6. Where does `grad_clip` happen relative to `backward()` and `step()` (line numbers), and what tensor does it act on?

## C. Forward + loss — textbook vs. this codebase

7. In `loss = out["loss"]` (line 83), is the loss being *computed* on that line or *retrieved*? Justify from the code.
8. Where is the loss actually computed, and why doesn't `train.py` itself call something like `criterion(pred, target)`?
9. This is a masked autoencoder — self-supervised. What plays the role of the "label" that predictions are compared against?

## D. Backprop and gradients

10. What does `loss.backward()` compute? What does it explicitly *not* decide?
11. True/false: `loss.backward()` updates the model's weights. Explain.
12. `optimizer.step()` needs two things to do its job. Name them and say where each comes from.

## E. Parameters vs hyperparameters

13. Define "parameter" and "hyperparameter" in one sentence each.
14. Classify each as parameter or hyperparameter: `cls_token`, `lr`, `weight_decay`, `betas`, `embed_dim`, `decoder_embed_dim`, `pos_embed_type`, `.grad`, the `Conv2d` weight in `PatchEmbedMulti1D`.
15. Is `.grad` a parameter? What is it exactly, and what two events control when it's populated vs. cleared?
16. `embed_dim` and `decoder_embed_dim` never get a `.grad`. Why not, mechanically — what Python type are they, and what do they do instead of being learned?
17. Where in the code does `decoder_embed_dim` get its default value if the config doesn't set it, and what is that default in terms of `embed_dim`?

## F. Math intuition — num_patches, sequence length, head_dim

18. Formula and computation: `num_patches` for `multi_1d` with `num_traces=32`, `trace_length=1024`, `patch_size=16`.
19. What is the actual token sequence length fed into the encoder, and why is it different from `num_patches`?
20. Formula and computation: `head_dim` for the encoder (`embed_dim=256`, `num_heads=8`) and for the decoder (`decoder_embed_dim=128`, `decoder_num_heads=4`).
21. What constraint does 2-axis RoPE place on `head_dim`, and do both the encoder's and decoder's `head_dim` satisfy it?

---

# ANSWER KEY

1. Epoch = one full pass through the entire dataset. Step = processing one batch (one iteration of the inner loop). Batch = the group of samples processed together in one step.
2. `400 / 32 = 12.5` → 12 full steps of 32, plus one partial step of 16 (or 13 steps total if the loader doesn't drop the remainder).
3. Epoch loop: line 77, `for epoch in range(start_epoch, int(train_cfg["epochs"]))`. Step loop: line 80, `for step, x in enumerate(loader)`.
4. (81) `x = x.to(device, non_blocking=True)` — move batch to device. (82) `out = model(x)` — forward pass. (83) `loss = out["loss"]` — unpack the already-computed loss. (85) `optimizer.zero_grad(set_to_none=True)` — clear old gradients. (86) `loss.backward()` — backprop, populate `.grad`. (87–88) `torch.nn.utils.clip_grad_norm_(...)` if `grad_clip` is set — rescale gradients. (89) `optimizer.step()` — update parameters using `.grad`.
5. Gradients accumulate by default in PyTorch (`+=`, not overwrite). Without `zero_grad()`, each step's `.grad` would add on top of every previous step's gradient, corrupting the update — not a clean per-step gradient.
6. Between `backward()` (line 86) and `step()` (line 89) — specifically lines 87–88, `if grad_clip: torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))`. It acts on the gradients (`.grad`) just populated by `backward()`, rescaling them if their combined norm exceeds `grad_clip` (1.0 in this config) — not on the weights themselves.
7. Retrieved, not computed. `out` is a dict already returned by `model(x)` on the previous line; `out["loss"]` is a plain dictionary key lookup, no computation happens on line 83 itself.
8. Inside `SeisFoundation.forward`, which internally calls `forward_loss` (`foundation.py:315–330`) as part of the same forward pass — encoder, decoder, and loss all run in one `model(x)` call. `train.py` never sees raw predictions or targets separately; it only receives the finished scalar loss.
9. The input itself, reshaped. `forward_loss` compares the decoder's predictions against `patchify(x)` (the same input, chopped into raw patches) — restricted to the masked patches. No external label array exists; the data is its own target.
10. It computes the gradient of the loss with respect to every parameter (`∂L/∂θ`), via the chain rule, and stores each in that parameter's `.grad`. It does **not** decide the step size or the update rule — that's what `optimizer.step()` does with the gradients afterward.
11. False. `backward()` only fills in `.grad` on each parameter. No `.data` (the actual weight values) changes until `optimizer.step()` runs.
12. (1) The gradients — `.grad` on each parameter, populated by `loss.backward()` the line before. (2) The hyperparameters bound at construction — `lr`, `betas`, `weight_decay` — set in the `AdamW(...)` call, `train.py:56–61`.
13. Parameter: a learned tensor, updated by the optimizer using gradients (e.g. weights, biases). Hyperparameter: a fixed value chosen before training that controls model shape or the training process, never updated by gradient descent.
14. Parameters: `cls_token`, the `Conv2d` weight in `PatchEmbedMulti1D`. Hyperparameters: `lr`, `weight_decay`, `betas`, `embed_dim`, `decoder_embed_dim`, `pos_embed_type`. Neither (an attribute of a parameter, not a category of its own): `.grad`.
15. No — `.grad` is not a parameter, it's an attribute attached to a parameter tensor (alongside `.data`). It's populated by `loss.backward()` each step and typically cleared by `optimizer.zero_grad()` at the start of the *next* step (line 85) — it doesn't persist as learned content the way `.data` does.
16. Both are plain Python `int`s read straight from the config dict (`foundation.py:42`, `embed_dim = cfg["embed_dim"]`; line 47, `decoder_embed_dim = cfg.get(...)`), not `torch.Tensor`s — so there's no `.grad` slot to populate at all. Instead of being learned, they *size* the tensors that are learned, e.g. `nn.Parameter(torch.zeros(1, 1, embed_dim))` (line 62).
17. `foundation.py:47`: `decoder_embed_dim = cfg.get("decoder_embed_dim", embed_dim // 2)`. Default is half of `embed_dim` if not explicitly set in the config (and in `configs/config.yaml`, it *is* explicitly set to 128, which matches `256 // 2` anyway).
18. `num_patches = num_traces × (trace_length / patch_size) = 32 × (1024/16) = 32 × 64 = 2048`.
19. 2049 — `num_patches` (2048) plus one CLS token, prepended once per sample regardless of gather size (`self.cls_token`, `foundation.py:62`).
20. Encoder: `head_dim = embed_dim / num_heads = 256 / 8 = 32`. Decoder: `head_dim = decoder_embed_dim / decoder_num_heads = 128 / 4 = 32`.
21. `head_dim % 4 == 0`, because RoPE splits `head_dim` into 2 axis-halves (angle, time) and rotates each half in 2D pairs — so each half must itself be evenly divisible by 2, meaning the whole `head_dim` must be divisible by 4. Both encoder (32) and decoder (32) satisfy it.
