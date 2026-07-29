# Unit 3 Self-Test — Patchify + Patch Embedding (`models/patch_embed.py`)

Cover the answer key. Answer each cold, then scroll down to check.

Reference config throughout unless stated otherwise:
`input_type: multi_1d`, `num_traces=32`, `trace_length=1024`, `patch_size=16`, `embed_dim=256`.

---

## A. Vocabulary — patches, tokens, and the four counters

1. What is the difference between a *patch* and a *token*? Give the shape of each.
2. Is `num_patches` the same as the number of tokens the encoder sees? If not, what is the difference and where in the code does it come from?
3. Define each from memory, and give its value for the reference config: `patch_size`, `time_patches`, `num_traces`, `num_patches`, `patch_dim`.
4. Which two of those live only on `PatchEmbedMulti1D`, and which lives only on `PatchEmbed1D`?

## B. Convolution as the embedding layer

5. Which line of `PatchEmbed1D` is the embedding layer, and what does it map from and to?
6. In `nn.Conv1d` / `nn.Conv2d`, what do "1d" and "2d" actually count?
7. State the output-length formula for a conv with kernel K and stride S, no padding.
8. Does a single output value of a conv depend on all input channels or just one? State the general rule.
9. `nn.Conv1d(1, 256, kernel_size=16, stride=16)` on input `(4, 1, 1024)`. Give the output shape, the weight shape, the bias shape, and the total parameter count.
10. Same layer but `stride=1`, `out_channels=8`. Give the output shape. What is this operation called in geophysics, and how does it differ in *purpose* from the stride-16 version?
11. Why is `stride == kernel_size` the special case that makes a conv equivalent to a per-patch linear projection? Write the equation.
12. `nn.Conv1d(3, 64, kernel_size=8, stride=4)` on `(2, 3, 100)`. Output shape? Weight shape?
13. `nn.Conv2d(1, 256, kernel_size=(1,16), stride=(1,16))` on `(4, 1, 32, 1024)`. Output shape and weight shape?
14. **Key question.** `PatchEmbedMulti1D`'s conv kernel has height 1 — only one axis does anything. So why does line 58 use `Conv2d` instead of `Conv1d`? What would break?
15. Compare the parameter count of `Conv1d(1, 128, kernel_size=2)` against `Conv2d(1, 128, kernel_size=(1,2))`. Does the parameter count depend on the number of traces?
16. Hand-compute: patch `p = [1.0, -0.5]`, filter `w = [0.3, -0.7]`, bias `b = 0.1`. What is the output? Is `b` re-used at every patch position or is there one bias per position?
17. What does `embed_dim` control — and what does it *not* control? Is the seismic embedding a compression or an expansion?

## C. `forward` vs `patchify` — two different operations

18. Give the output shape of `forward` and of `patchify` for the reference config with `B=4`.
19. Which of the two has learnable weights? Which is a pure reshape?
20. Does `forward` call `patchify` internally? Explain how the patch-cutting happens in `forward`.
21. Which of the two produces the reconstruction *target* for the MAE loss, and at which line of `foundation.py`?
22. If you change `embed_dim` from 256 to 512, which of these change: `num_patches`, `patch_dim`, `forward` output shape, `patchify` output shape? Why?
23. Why is `decoder_pred` defined as `nn.Linear(decoder_embed_dim, patch_dim)` rather than something involving `embed_dim`?
24. What invariant must `forward` and `patchify` share for the loss to be meaningful? Is it enforced anywhere in the code?

## D. `patchify` — shape walkthroughs

25. `PatchEmbed1D` with `in_channels=2, trace_length=12, patch_size=4`, batch of 2. Give the shape after each line: input → line 30 → line 31 permute → line 31 reshape.
26. Same setup, with channel 0 = `a0..a11` and channel 1 = `b0..b11`. Write out the three tokens element by element.
27. `PatchEmbedMulti1D` with `num_traces=3, trace_length=12, patch_size=4`, batch of 2. Shape after line 67 and line 68.
28. Same setup, angles `a`, `b`, `c`. Write out all nine tokens element by element.
29. `PatchEmbed2D` with `img_size=(128,64)`, `patch_size=(16,16)`, `in_channels=1`, batch of 2. Give `grid_h`, `grid_w`, `num_patches`, `patch_dim`, and the shape after each line: input → line 113 → line 114 permute → line 114 reshape.
30. Why do `PatchEmbed1D` and `PatchEmbed2D` need a `permute` but `PatchEmbedMulti1D` does not? State the general rule about `reshape`.
31. Could line 31's trailing `reshape(B, num_patches, C*P)` be replaced by `.flatten(2)`? Could it be replaced by `.view(B, num_patches, C*P)`? Explain both.
32. Would `x.flatten(2)` applied directly to the line-30 output `(B, C, N, P)` give the same result? What shape would it give?

## E. Token index arithmetic

33. Write the formula that converts `(angle_idx, time_idx)` to token number k, and the two that invert it. Which counter is the multiplier?
34. Why is the multiplier `time_patches` and not `num_traces`?
35. Token 100: give `angle_idx`, `time_idx`, and the sample range on the original trace.
36. Which token holds samples 320–335 of angle 12?
37. Which token contains sample 700 of angle 5? (Note 700 is not a patch boundary.)
38. Token 2047: give `angle_idx`, `time_idx`, sample range. What is special about it?
39. Given `time_idx`, how do you get the first and last sample index of that patch?
40. `PatchEmbed2D` with `grid_w = 4`: which token is at row 3, col 2? What is the general formula?
41. In `multi_1d`, are consecutive tokens neighbours in time or in angle? What about tokens 64 apart?

## F. Why the three classes differ

42. State the one rule that decides whether an axis becomes *more tokens* or *more numbers inside a token*.
43. Why is `patch_dim = patch_size` for `multi_1d` but `in_channels * patch_size` for `1d`? Answer in terms of the physics, not the shapes.
44. Give a concrete seismic example of the kind of second axis that belongs *inside* a token, and one that belongs in *separate* tokens.
45. For the same 16 input numbers arranged as 2 rows of 8 with `patch_size=4`: how many tokens and how wide, under `PatchEmbed1D(in_channels=2)` vs `PatchEmbedMulti1D(num_traces=2)`?
46. What does one token cover physically in each of the three classes?
47. What is the computational cost of `PatchEmbedMulti1D`'s choice? Quantify it against the alternative.
48. `PatchEmbedMulti1D` and `PatchEmbed2D` both produce a 2D token grid. What actually distinguishes them? Look at the conv kernels.

## G. `unpatchify`

49. Give the input and output shapes of `unpatchify` for `PatchEmbed1D` and for `PatchEmbedMulti1D`.
50. True/false: `unpatchify` turns patches into one long sequence. Correct it if false.
51. Is `unpatchify(patchify(x))` exactly equal to `x`, or approximately? Justify from the code.
52. Where is `unpatchify` called in `foundation.py`? Is it in the training path? Where is the loss actually computed — patch space or gather space?
53. What array is passed into `unpatchify` at that call site, and what does the result represent?

## H. Position bookkeeping (`_register_axis_positions`)

54. What does `_register_axis_positions` build, and what are the shapes of what it stores for `multi_1d`?
55. Where is it called and how many times per training run?
56. Write out the first 10 entries of `angle_idx` and of `time_idx` for `time_patches=3, num_traces=3`.
57. Why is `register_buffer` used rather than `nn.Parameter`? Name two consequences.
58. Why does the model need these at all — what does attention *not* know without them?
59. Which components consume them? Name at least two call sites.

## I. Traps and integration

60. At line 66, `B, C, T = x.shape` — what does `C` mean there, and why is that a trap?
61. `pos_axis_0` means one thing for `multi_1d` and another for `2d`. What, and why? Where is the axis convention documented?
62. Trace the full shape for `multi_1d`, batch of 4: raw gather → `patchify` → `forward` → after CLS is prepended.
63. In `(4, 2048, 256)`, which single number changes if you (a) double the batch, (b) halve `patch_size`, (c) double `num_traces`, (d) set `embed_dim=512`?
64. Name the three assertions/constraints in this file and what each protects against.

---

# ANSWER KEY

1. A patch is raw data — a contiguous slice of the signal, shape `(patch_dim,)`, in amplitude units. A token is a `D`-dimensional vector in the model's learned space, shape `(embed_dim,)`, with no physical units. `forward` maps patch → token.
2. No. `num_patches` = 2048, but the encoder sees 2049: `foundation.py:62` prepends a learned `cls_token`, and `pos_embed` is sized `num_patches + 1` at line 73.
3. `patch_size` = 16, samples per patch. `time_patches` = 64 = `trace_length // patch_size`, patches per single trace. `num_traces` = 32, the angle axis. `num_patches` = 2048 = `num_traces * time_patches`, total tokens. `patch_dim` = 16, raw numbers per patch.
4. `time_patches` and `num_traces` exist only on `PatchEmbedMulti1D` (lines 52, 55). `in_channels` is used by `PatchEmbed1D` and `PatchEmbed2D` but not `PatchEmbedMulti1D`.
5. Line 18, `self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)`. Maps `(B, C, T)` → `(B, D, N)`; line 24 transposes to `(B, N, D)`.
6. The number of **spatial axes** the window slides over — not the number of tensor dimensions. `Conv1d` input is 3D, `Conv2d` input is 4D, because both carry batch and channel axes on top.
7. `L_out = (L - K) // S + 1`.
8. All of them. The kernel always spans the full `C_in` axis and collapses it — the channel axis is never slid over.
9. Output `(4, 256, 64)`; weight `(256, 1, 16)`; bias `(256,)`; params = 256×1×16 + 256 = **4352**.
10. `L_out = (1024-16)//1 + 1 = 1009`, output `(4, 8, 1009)`. That is **filtering** — eight FIR filters of length 16 applied at every sample. Purpose differs: stride 1 produces a filtered trace of nearly the original length; stride = kernel produces one summary vector per non-overlapping window, i.e. tokenization.
11. Because the windows tile the input without overlap, so window n sees exactly patch n and nothing else. The layer reduces to `z_n = W p_n + b` applied independently per patch, with `W` of shape `(embed_dim, C*P)` and `b` of shape `(embed_dim,)`.
12. `(100-8)//4 + 1 = 24` → output `(2, 64, 24)`; weight `(64, 3, 8)`.
13. `H_out = (32-1)//1+1 = 32`, `W_out = (1024-16)//16+1 = 64` → output `(4, 256, 32, 64)`; weight `(256, 1, 1, 16)`.
14. Because a `Conv1d` would have to treat the 32 traces as *input channels* — and the kernel always collapses the channel axis. `nn.Conv1d(32, 256, 16, stride=16)` on `(B, 32, 1024)` gives `(B, 256, 64)`: only 64 tokens, with all 32 angles mixed together in the very first layer. The angle axis would be destroyed before attention could ever compare angles. `unsqueeze(1)` at line 61 promotes the trace axis to a *spatial* axis, and kernel height 1 means the kernel never spans it — 32 traces in, 32 out, independent.
15. Both are 128×2 + 128 = **384**. A kernel height of 1 adds no parameters. And the count is independent of the number of traces — the same weights are reused on every trace (weight sharing).
16. `0.3*1.0 + (-0.7)*(-0.5) + 0.1 = 0.3 + 0.35 + 0.1 = 0.75`. There is **one bias per output channel**, re-used at every patch position — not one per position.
17. `embed_dim` controls how many different filters are run, i.e. how many numbers describe each patch. It does **not** control `num_patches`, `patch_dim`, or `patch_size`. For seismic it is an expansion: 16 raw numbers → 256.
18. `forward` → `(4, 2048, 256)`. `patchify` → `(4, 2048, 16)`.
19. `forward` (the conv `proj`) has weights. `patchify` is a pure reshape with no parameters at all.
20. No — `forward` never calls `patchify`. The cutting is implicit in `stride == kernel_size`: the conv's windows land on exactly the same non-overlapping segments.
21. `patchify`, at `foundation.py:333` (`target = self.patchify(x)`).
22. Only `forward`'s output shape changes, to `(4, 2048, 512)`. `patchify` has no weights, so it cannot depend on `embed_dim`; its width is `patch_dim`, fixed by `patch_size` and `in_channels`. `num_patches` depends only on the grid.
23. Because the decoder's output has to land in the same space as the loss target, which is `patchify`'s space — width `patch_dim`, not `embed_dim`.
24. Both must enumerate patches in the **same order**, since the loss at `foundation.py:326` subtracts `pred[k]` from `target[k]` elementwise. Nothing in the code asserts this — it is an implicit invariant.
25. `num_patches = 3`, `patch_dim = 8`. `(2,2,12)` → line 30 `(2,2,3,4)` → permute `(2,3,2,4)` → reshape `(2,3,8)`.
26. token 0 = `a0 a1 a2 a3 b0 b1 b2 b3`; token 1 = `a4 a5 a6 a7 b4 b5 b6 b7`; token 2 = `a8 a9 a10 a11 b8 b9 b10 b11`.
27. `time_patches = 3`, `num_patches = 9`, `patch_dim = 4`. `(2,3,12)` → line 67 `(2,3,3,4)` → line 68 `(2,9,4)`.
28. tokens 0–2 = `a0..a3`, `a4..a7`, `a8..a11`; tokens 3–5 = `b0..b3`, `b4..b7`, `b8..b11`; tokens 6–8 = `c0..c3`, `c4..c7`, `c8..c11`.
29. `grid_h = 8`, `grid_w = 4`, `num_patches = 32`, `patch_dim = 256`. `(2,1,128,64)` → line 113 `(2,1,8,16,4,16)` → permute `(2,8,4,1,16,16)` → reshape `(2,32,256)`.
30. `reshape` can only merge axes that are already **adjacent and already in the desired order**. In `PatchEmbed1D` you must merge `C` with `P`, but `N` sits between them → permute first. In `PatchEmbedMulti1D` you merge `num_traces` with `time_patches`, which are already adjacent and in the right order → plain reshape.
31. `.flatten(2)` — yes, identical. It merges axes 2 and 3 of `(B,N,C,P)` into `(B,N,C*P)`, same order, and like `reshape` it silently copies a non-contiguous tensor. `.view()` — no, it would raise, because `permute` leaves the tensor non-contiguous and `view` requires contiguity.
32. No. `(B, C, N, P).flatten(2)` merges `N` and `P`, giving `(B, C, N*P)` — a different tensor and the wrong one. The patch index must be on axis 1.
33. `k = angle_idx * time_patches + time_idx`; inverses `angle_idx = k // time_patches`, `time_idx = k % time_patches`. The multiplier is `time_patches`.
34. Because `time_idx` is the fast axis — it takes a full `time_patches` steps of `time_idx` before `angle_idx` advances by one. The multiplier is always the size of the faster axis. (`k = time_idx * num_traces + angle_idx` would describe a different, time-major memory layout, which is not what line 68 produces.)
35. `100 // 64 = 1`, `100 % 64 = 36` → angle 1, time_idx 36 → samples `36*16 = 576` through 591.
36. `320 // 16 = 20` → time_idx 20. `k = 12*64 + 20 = 788`.
37. `700 // 16 = 43` → time_idx 43 (covering samples 688–703, which contains 700). `k = 5*64 + 43 = 363`.
38. `2047 // 64 = 31`, `2047 % 64 = 63` → angle 31, time_idx 63 → samples 1008–1023. It is the **last** token: both indices are at their maximum, one below `num_traces` and `time_patches`.
39. `first = time_idx * patch_size`, `last = first + patch_size - 1`.
40. `k = row * grid_w + col = 3*4 + 2 = 14`. General: multiply the slow index by the size of the fast axis, add the fast index. Same arithmetic as `multi_1d` with `grid_w` in place of `time_patches`. (`foundation.py:128–129`.)
41. Consecutive tokens are neighbours in **time**, same angle (time is the fast axis). Tokens 64 apart are the same time patch on adjacent angles.
42. Attention compares *tokens*, and cannot compare two numbers already merged inside one token. So: things the model should treat as one joint observation → same token; things the model should compare against each other → separate tokens.
43. `multi_1d`'s second axis is reflection angle, and how amplitude varies with angle (AVO) is exactly what you want the model to reason about — so angles must be separate tokens, and a token holds only `patch_size` numbers. `1d`'s `in_channels` are co-located sensors (e.g. 3C geophone x/y/z, or dual-sensor pressure + velocity) measuring the same instant at the same place — one joint observation, nothing to compare — so they sit inside one token, width `C*P`.
44. Inside a token: the three components of a 3C geophone, or hydrophone + geophone on a dual-sensor streamer. Separate tokens: reflection angles in an angle gather; offsets in a CMP gather.
45. `PatchEmbed1D(in_channels=2)`: 2 tokens of width 8. `PatchEmbedMulti1D(num_traces=2)`: 4 tokens of width 4. Same 16 numbers, different packaging.
46. `PatchEmbed1D`: one time patch, all `C` channels. `PatchEmbedMulti1D`: one trace × `patch_size` samples — a horizontal sliver. `PatchEmbed2D`: a rectangular tile of `ph` traces × `pw` samples.
47. Attention is O(N²). Folding angle into features would give 64 tokens (~4k pairs); keeping them separate gives 2048 tokens (~4.2M pairs) — roughly 1000× more. The author paid that deliberately to keep angle resolvable.
48. The conv kernel. `PatchEmbedMulti1D` uses kernel `(1, patch_size)` — height 1, so a patch never spans more than one trace. `PatchEmbed2D` uses kernel `(ph, pw)` — the patch is a genuine 2D tile spanning several traces *and* several time samples. The token grids look alike; what a token *contains* differs.
49. `PatchEmbed1D`: `(B, N, C*P)` → `(B, C, T)`. `PatchEmbedMulti1D`: `(B, num_traces*time_patches, P)` → `(B, num_traces, T)`. For the reference config, `(4, 2048, 16)` → `(4, 32, 1024)`.
50. **False.** `patchify` makes the sequence; `unpatchify` destroys it and rebuilds the original gather. The output is a 3D array with the trace/angle axis restored, not a sequence.
51. **Exactly** equal. Line 37 applies `permute(0, 2, 1, 3)`, the same permutation as line 31, and that permutation is its own inverse (it swaps axes 1 and 2 both times). No floating-point tolerance needed.
52. Only at `foundation.py:357`, inside `reconstruct`. It is **not** in the training path — `forward` (lines 332–338) computes the loss at line 326 entirely in patch space, with `pred` and `target` both `(B, N, patch_dim)`. It is a viewing utility: it produces something plottable as a gather.
53. `pred` — the decoder's *predicted* patches, not real data. The result is the model's reconstruction of the gather, not the gather itself. `reconstruct` returns both `pred_patches` and `recon` (lines 363–364) so you can compare against the input.
54. Per-token coordinates on the original grid. For `multi_1d` it stores two integer buffers of length 2048: `pos_axis_0` = `angle_idx`, `pos_axis_1` = `time_idx` (`foundation.py:120–123`). It is the k → (angle, time) arithmetic, precomputed for all k at once.
55. `foundation.py:59`, inside `__init__` — **once**, at model construction. Never during `forward`; the values never change.
56. `angle_idx = [0,0,0, 1,1,1, 2,2,2]` (`repeat_interleave(3)`); `time_idx = [0,1,2, 0,1,2, 0,1,2]` (`repeat(3)`). Nine tokens total, so only nine entries exist.
57. They are fixed integers, not learned. Consequences: (a) they travel with `.to(device)` and are written into the checkpoint (listed at `foundation.py:226`); (b) they get no gradient and are never touched by the optimizer.
58. Attention is permutation-invariant — without position information it cannot tell that token 100 and token 164 are the same time patch on neighbouring angles, or that tokens 100 and 101 are adjacent in time. These buffers hand the grid geometry back to the model.
59. `_axes_buffers` (line 135) → `_expand_positions` (line 142) → `RopeCache`; and `_build_sincos_pos_embed` reads them directly at lines 181 and 187–188. All three `pos_embed_type` settings consume them.
60. At line 66 `C` is the **angle axis** (`num_traces`), not `in_channels`. The author reused the letter from `PatchEmbed1D` (line 29) where it genuinely means channels. Same name, different physical meaning, 37 lines apart.
61. For `multi_1d`, input is `(B, A, T)` so `pos_axis_0` = angle and `pos_axis_1` = time. For `2d`, input is `(B, H, W)` with **H = time, W = trace/offset** (documented at `dataset.py:35` and `:95–97`), so `pos_axis_0` = time and `pos_axis_1` = trace. Axis 0 means different things in different modalities. Nothing breaks — RoPE rotates whatever it is given — but don't assume they correspond.
62. `(4, 32, 1024)` → `patchify` `(4, 2048, 16)` → `forward` `(4, 2048, 256)` → after CLS `(4, 2049, 256)`.
63. (a) `(8, 2048, 256)`; (b) `(4, 4096, 256)`; (c) `(4, 4096, 256)`; (d) `(4, 2048, 512)`.
64. Line 12 (`PatchEmbed1D`) and line 51 (`PatchEmbedMulti1D`): `trace_length % patch_size == 0` — protects against a ragged final patch, since the conv would silently drop the remainder and `unpatchify` could not rebuild the original length. Line 92 (`PatchEmbed2D`): the same divisibility check on both image axes. There is **no** assertion that `forward` and `patchify` order patches identically (see Q24), and none that `in_channels` matches the input.
