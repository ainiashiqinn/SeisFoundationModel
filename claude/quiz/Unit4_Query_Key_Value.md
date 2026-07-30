# Unit 4 Self-Test — Query, Key, Value (`models/rope.py:RopeAttention`)

Cover the answer key. Answer each cold, then scroll down to check.

Reference config throughout unless stated otherwise:
`input_type: multi_1d`, `num_traces=32`, `trace_length=1024`, `patch_size=16`,
`embed_dim=256`, `num_heads=8`, `depth=6`, `batch_size=16`, `mask_ratio=0.75`,
`decoder_embed_dim=128`, `decoder_num_heads=4`.

**Scope note:** this covers Q/K/V, softmax, scaling, the score matrix, and what
happens to the attention output. Multi-head mechanics (the `reshape`/`permute`/
`unbind` on `rope.py:79`), `self.proj`, and permutation equivariance are *not*
covered here — separate sheet.

### Toy example used by several questions

```
x[0] = [1, 0]      x[1] = [0, 1]      x[2] = [1, 1]

W_Q = [[1, 1],     W_K = [[1, 0],     W_V = [[2,  0],
       [0, 1]]            [0, 1]]            [0, -1]]
```

---

## A. Why attention exists at all

1. What does MLP stand for, and where does it live in `RopeBlock`?
2. State the one limitation of an MLP that attention exists to fix.

## B. Notation

3. In `score(i, j)`, what does `i` refer to and what does `j` refer to? Which of the two contributes its `q`, and which its `k`?
4. Compute by hand: `[0.9, -0.4, 1.2] . [1.1, -0.6, 0.8]`.
5. A dot product of two vectors of length 32 produces how many numbers?

## C. Softmax

6. What two properties does softmax guarantee about its output?
7. Write out the two steps of softmax on the list `[1.0, 0.0, 1.0]`, showing the intermediate values.
8. Why raise `e` to each number instead of simply dividing each number by the total?
9. Apply softmax to `[1.0, 0.9]`, then to `[6.0, 1.0]`. What property of softmax do the two results demonstrate?
10. In the score matrix, does each *row* sum to 1, each *column*, or both?

## D. Q, K, V

11. Where do `q`, `k`, and `v` for a given token come from? Are they three separate inputs, or something else?
12. State the one-line job of each of the three.
13. Which of `q`, `k`, `v` appear in the attention output, and which are discarded after use?
14. Using the toy example, compute `q`, `k`, and `v` for `x[2] = [1, 1]`.
15. Must `q` and `k` have the same length? Must `v` be the same length as them? Which of those is a mathematical requirement and which is an implementation choice in this repo?
16. Which line of `rope.py` creates all three projection matrices, and what are its input and output dimensions?
17. Why is it one `nn.Linear` of size `dim*3` rather than three separate `nn.Linear(dim, dim)` layers?
18. In `self.qkv(x)` the output has 768 columns. Which columns belong to Q, which to K, which to V?

## E. Why three matrices and not one

19. Suppose you skipped `W_Q` and `W_K` entirely and scored tokens with `x[i] . x[j]` directly. Name the two flaws this has.
20. What property does the score matrix have if you tie `W_Q = W_K`? Can training overcome it?
21. Give a seismic example of a token pair where one-directional attention matters — where token A should attend strongly to token B but not the reverse.
22. Which is the more damaging modification and why: freezing `W_V` to the identity matrix, or tying `W_Q = W_K`?
23. Using the toy example, verify that `score(0,1)` is not equal to `score(1,0)`.

## F. The scaling factor

24. What is the full chain of consequences from "vectors are long" to "the model does not learn"? Five links.
25. If `q` and `k` have entries roughly centred on 0 with spread about 1, how does the typical size of `q . k` grow with vector length `d`?
26. Why divide by `sqrt(d)` rather than by `d`?
27. At `head_dim = 256` with no scaling, roughly what weight does the single best-matching token receive out of 50 candidates? What is it with scaling?
28. Explain in terms of gradients why a saturated softmax stops `W_Q` and `W_K` from learning.
29. Nudging one score by +0.1 moves the weights by 1.19e-03 at `head_dim=4` unscaled, and by 0.0 at `head_dim=32` unscaled. What does the second number mean for training?
30. Where in this repository is the `1/sqrt(head_dim)` division written? What is the honest answer if your advisor asks you to point at the line?

## G. head_dim

31. Define `head_dim` in one sentence. Which line of `rope.py` computes it?
32. Compute `head_dim` for the encoder and for the decoder using the reference config. What do you notice, and what is the mechanical reason for it?
33. Is `head_dim` or `embed_dim` the quantity that controls how sharp attention is? Justify.
34. What constraint does 2-axis RoPE place on `head_dim`, and which line asserts it? Does the reference config satisfy it?

## H. The score matrix

35. Give the four axes of the score matrix in order and say what each indexes.
36. Why does `N` appear twice? Are those two axes different sets of tokens?
37. Write the formula for a single entry `scores[b, h, i, j]` in words and in symbols.
38. Do tokens in batch element 3 ever attend to tokens in batch element 7? Do head 2 and head 5 share a score matrix?
39. For `mask_ratio = 0.0`, give N and the full score-matrix shape.
40. For `mask_ratio = 0.75` (training default), give N and the full score-matrix shape. Show how you get N.
41. Compute the number of entries and the float32 memory for both of the above.
42. Masking 75% of tokens reduces the score matrix by what factor? Why is it not 4x?
43. Under masking, does row index `i` refer to the original token number? If not, what does it refer to, and which variable keeps track of the mapping?
44. Can you print the score matrix in this codebase? Explain.
45. Token 1049 is (angle 16, time-patch 25) and token 2024 is (angle 31, time-patch 40). With CLS prepended and `mask_ratio=0`, which rows do they occupy? State in plain words what `scores[3, 5, 1050, 2025]` means.

## I. Convex combination

46. Define a convex combination. Which two conditions must hold?
47. Why does softmax guarantee both conditions automatically?
48. Given `v[0]=[2,0]`, `v[1]=[0,-1]`, `v[2]=[2,-1]`, what range must each coordinate of any attention output fall in?
49. Can the output ever land exactly on `v[1]`? Answer mathematically, then answer for float32.

## J. What happens to the output

50. What is the shape of the attention output relative to its input? Why does that matter?
51. Write the two lines of `RopeBlock.forward`. In `x = x + self.attn(self.norm1(x), cos, sin)`, which `x` is fed to attention and which `x` is added back?
52. Toy example: token 0 has `x[0] = [1, 0]` and attention output `[1.604, -0.599]`. What is its value after the residual line?
53. Is that post-residual value still inside the triangle formed by the three `v` vectors? What does that tell you about the role of the residual?
54. How many times does a token receive an attention update before leaving the encoder? Which config key sets it?
55. What does attention do that the MLP cannot, and what does the MLP do that attention cannot?

## K. Integration

56. Two implementations of attention exist in this repo. Name both, name their files, and say what selects between them.
57. Trace the toy example end to end: from `x[1] = [0, 1]` to its post-residual value, naming every step.

---

# ANSWER KEY

1. Multi-layer perceptron. `rope.py:101-107` — `nn.Linear(256, 1024)` → `nn.GELU()` → `nn.Linear(1024, 256)`, with dropout.
2. An MLP processes one token at a time. It has no way to look at any other token. Attention is the mechanism that lets tokens exchange information.
3. `i` = the token doing the looking (the one an output is being computed for); `j` = the token being looked at. `i` contributes its `q`, `j` contributes its `k`.
4. `(0.9*1.1) + (-0.4*-0.6) + (1.2*0.8) = 0.99 + 0.24 + 0.96 = 2.19`.
5. One.
6. Every output is positive, and they sum to exactly 1.
7. Step 1: `e^1.0 = 2.718`, `e^0.0 = 1.000`, `e^1.0 = 2.718`, sum `= 6.437`. Step 2: `2.718/6.437 = 0.422`, `1.000/6.437 = 0.155`, `2.718/6.437 = 0.422`. Result `[0.422, 0.155, 0.422]`.
8. Two reasons: `e^x` is always positive even for negative `x`, so weights stay valid; and it exaggerates differences, which makes attention selective rather than a flat average.
9. `[1.0, 0.9] → [0.525, 0.475]`; `[6.0, 1.0] → [0.993, 0.007]`. A small gap stays nearly tied, a large gap becomes winner-take-all. Softmax amplifies score gaps.
10. Each row sums to 1. Row `i` is token `i`'s attention budget split across all candidates. Columns sum to nothing in particular.
11. All three come from the *same* input vector, pushed through three different learned matrices: `q = W_Q x`, `k = W_K x`, `v = W_V x`. Three views of one vector, not three separate inputs.
12. `q` — what this token is looking for. `k` — what this token offers when others look at it. `v` — the content that gets passed along.
13. Only `v` appears in the output. `q` and `k` are consumed entirely in producing the weights and never appear in the result. Q/K are the routing circuit, V is the payload.
14. `q = [1*1 + 1*1, 0*1 + 1*1] = [2, 1]`; `k = [1, 1]`; `v = [2*1 + 0*1, 0*1 + (-1)*1] = [2, -1]`.
15. `q` and `k` must match — they are dotted together. `v` need not, mathematically. In this repo `nn.Linear(dim, dim*3)` forces all three to 32, which is an implementation convenience.
16. `rope.py:73`, `self.qkv = nn.Linear(dim, dim * 3)` — 256 in, 768 out.
17. One fused matmul is faster than three, and the three matrices are always applied to the same input anyway. It is purely an efficiency packing; mathematically identical.
18. Columns 0–255 are Q, 256–511 are K, 512–767 are V. The `3` axis in the `reshape` on line 79 performs that split.
19. (a) It is symmetric — `x[i].x[j]` always equals `x[j].x[i]`, so one-directional dependence is impossible. (b) "Relevant" is hard-coded to mean "similar," but a token often needs something complementary rather than similar.
20. The score matrix becomes symmetric for all inputs. Training cannot overcome it — it is structural, not a matter of weight values.
21. A masked far-offset token needs to draw heavily on a clear, high-amplitude near-offset token containing the same event. The near-offset token, already unmasked and unambiguous, has little to gain from the masked one. Symmetric weights cannot express that.
22. Tying `W_Q = W_K` is far more damaging, because it removes asymmetry permanently. Freezing `W_V = I` still leaves `v` a full-rank view of `x`, and `self.proj` immediately afterwards can recover much of what `W_V` would have done.
23. `score(0,1) = q[0].k[1] = [1,0].[0,1] = 0`. `score(1,0) = q[1].k[0] = [1,1].[1,0] = 1`. Not equal.
24. Long vectors → big dot products (more terms in the sum) → softmax saturates (one weight ≈ 1, the rest ≈ 0) → gradient through the softmax ≈ 0 → `W_Q` and `W_K` receive no update and never learn.
25. Like `sqrt(d)`. The terms have mixed signs and largely cancel, so the sum grows as the square root of the number of terms, not linearly.
26. Because the quantity being cancelled grows like `sqrt(d)`. Dividing by `d` would over-correct, shrinking all scores toward zero and flattening attention back into the useless uniform average.
27. About 0.97 without scaling (out of 50 candidates, where an even split would be 0.02). With scaling, about 0.12 — and that value stays roughly constant as `head_dim` changes, which is the whole point.
28. Training asks "if I change this score slightly, how much does the output change?" When softmax is saturated the answer is essentially zero, so the gradient flowing back to `W_Q` and `W_K` is essentially zero. They stay frozen near their random initial values.
29. It means the gradient has underflowed to exactly zero in float32. No update at all reaches `W_Q` or `W_K` — the model is permanently stuck with whatever random attention pattern initialization gave it.
30. Nowhere in this repository. `F.scaled_dot_product_attention` (`rope.py:84`) applies it internally, as does `nn.MultiheadAttention` in `transformer.py`. The honest answer is that it happens inside PyTorch and is not written in this codebase.
31. The length of each individual `q`, `k`, and `v` vector — i.e. the length of the vectors actually being dotted together. `rope.py:71`: `self.head_dim = dim // num_heads`.
32. Encoder `256 // 8 = 32`; decoder `128 // 4 = 32`. Both are 32, so the divisor `sqrt(32) = 5.657` is identical in both. Mechanically, the decoder halves both `embed_dim` and `num_heads`, and halving numerator and denominator leaves the ratio unchanged. Whether the author intended `head_dim = 32` or simply halved everything is not recoverable from the code.
33. `head_dim`. The decoder is half the width of the encoder, but because `head_dim` matches, its typical dot-product size, softmax sharpness, and saturation risk are identical. `embed_dim` does not enter the scaling at all.
34. `head_dim % 4 == 0`, asserted at `rope.py:44` (each of the two axis subspaces needs an even size). `32 % 4 == 0`, so both encoder and decoder pass.
35. `(B, H, N, N)`. `B` = which gather in the batch; `H` = which head; first `N` = the token doing the looking (rows); second `N` = the token being looked at (columns).
36. Because every token plays two roles — it has a `q` (looking) and a `k` (being looked at). The matrix covers every pairing of those roles. It is the *same* set of tokens on both axes, not two different sets.
37. In words: in gather `b`, according to head `h`, how strongly token `i` wants to pull information from token `j`. In symbols: `(q of token i) . (k of token j) / sqrt(32)`, both taken from gather `b`, head `h`.
38. No and no. Batch elements never interact — that is what makes `B` a separate axis. Each head computes its own completely independent score matrix.
39. `N = 2048 + 1 CLS = 2049`. Shape `(16, 8, 2049, 2049)`.
40. `2048 × (1 − 0.75) = 512` kept, plus 1 CLS, so `N = 513`. Shape `(16, 8, 513, 513)`.
41. `mask_ratio=0`: `16*8*2049*2049 = 537,395,328` entries = 2,150 MB. `mask_ratio=0.75`: `16*8*513*513 = 33,685,632` entries = 135 MB.
42. About 16x, because cost scales with `N` squared and `N` drops roughly 4x. Not exactly 16x because the `+1` CLS token does not scale with `mask_ratio`.
43. No. Under masking, `i` indexes a position in the packed 513-long kept-token sequence. Row 7 might be original token 1583. `ids_keep` carries the mapping back to original positions, which is what lets RoPE rotate kept tokens by their true positions (Unit 7).
44. No. `F.scaled_dot_product_attention` is fused — it computes scores, softmaxes, and multiplies by `v` in chunks, so the full `(B, H, N, N)` array never exists as a tensor. `transformer.py` separately passes `need_weights=False`. Getting attention maps out requires editing the code.
45. Rows 1050 and 2025 (CLS occupies row 0, so gather token `k` sits at row `k+1`). `scores[3, 5, 1050, 2025]` means: in gather 3, head 5, how much the near-offset patch at time-patch 25 should draw from the far-offset patch at time-patch 40.
46. A weighted sum where every weight is ≥ 0 and the weights sum to exactly 1. Drop either condition and it is only a linear combination.
47. `e^x` is always positive, giving condition one. Dividing by the sum forces the total to 1, giving condition two.
48. Coordinate 0 must fall in `[0, 2]`; coordinate 1 must fall in `[-1, 0]`. Geometrically the output lies inside the triangle with those three corners.
49. Mathematically no — softmax weights are strictly positive, so every `v` contributes something and the output is strictly inside, never on a corner or edge. In float32 yes: when saturated, the largest weight rounds to exactly 1 and the others to exactly 0, so the output lands on a corner numerically. That rounding is the same event as the vanishing gradient.
50. Identical — `(B, N, 256)` in, `(B, N, 256)` out. That shape preservation is what allows the block to be stacked repeatedly.
51. `x = x + self.attn(self.norm1(x), cos, sin)` then `x = x + self.mlp(self.norm2(x))`. Attention is fed the *normalized* `x`; the *original, un-normalized* `x` is what gets added back. Attention produces a correction, not a replacement.
52. `[1 + 1.604, 0 + (-0.599)] = [2.604, -0.599]`.
53. No — coordinate 0 is 2.604, outside the `[0, 2]` range. The residual escapes the convex hull, which is precisely how the representation can move somewhere new rather than only interpolating among existing values. `self.proj` and the MLP do the same.
54. Six. `depth: 6` in `config.yaml:32`, built at `foundation.py:69`.
55. Attention lets tokens exchange information with each other. The MLP lets each token process what it received, on its own — it cannot mix across tokens.
56. `RopeAttention` using `F.scaled_dot_product_attention` (`models/rope.py`), and `Block` using `nn.MultiheadAttention` (`models/transformer.py`). `pos_embed_type` selects which — `rope` gets the first, `sinusoidal`/`learnable` get the second.
57. `x[1] = [0,1]` → `q = [1,1]`, `k = [0,1]`, `v = [0,-1]` → scores against the three `k` vectors = `[1, 1, 2]` → divide by `sqrt(2) = 1.414` → `[0.707, 0.707, 1.414]` → softmax → `[0.248, 0.248, 0.503]` → weighted sum of `v` = `0.248*[2,0] + 0.248*[0,-1] + 0.503*[2,-1] = [1.502, -0.751]` → residual `x + out = [0,1] + [1.502,-0.751] = [1.502, 0.249]`.
