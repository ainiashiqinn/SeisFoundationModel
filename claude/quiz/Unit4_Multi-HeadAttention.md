# Unit 4 Self-Test — Multi-Head Attention (`models/rope.py:RopeAttention`)

Cover the answer key. Answer each cold, then scroll down to check.

Reference config throughout unless stated otherwise:
`input_type: multi_1d`, `num_traces=32`, `trace_length=1024`, `patch_size=16`,
`embed_dim=256`, `num_heads=8`, `depth=6`, `mlp_ratio=4.0`, `batch_size=16`,
`pos_embed_type: rope`, `rope_base=10000.0`.

**Scope note:** this covers the head split, the `reshape`/`permute`/`unbind` shape
trace on `rope.py:79-80`, `nn.Linear`, weight initialization, where weights are
actually updated, the concatenation on line 88, and `self.proj` on line 89.
Q/K/V, softmax, and score scaling are on the Query-Key-Value sheet. Permutation
equivariance and RoPE itself are not covered here.

### Toy example A — two heads

`B=1, N=4, D=8, num_heads=2, head_dim=4`.

```
q[0] = [2, 0, 1, 3,  5, 1, 0, 2]
k[1] = [1, 4, 2, 0,  3, 3, 1, 1]
       \__head 0__/ \__head 1__/
```

### Toy example B — nn.Linear

```
nn.Linear(3, 4)                        nn.Linear(2, 3)

weight = [[ 0.5, -1.0,  2.0],          weight = [[ 1.0, -1.0],
          [ 1.0,  0.0,  0.5],                    [ 0.0,  2.0],
          [-0.5,  0.5,  1.0],                    [ 3.0,  1.0]]
          [ 2.0,  1.0, -1.0]]
                                       bias   = [1.0, 0.0, -2.0]
bias   = [0, 0, 0, 0]
```

---

## A. What actually gets split

1. `rope.py:71` computes `self.head_dim = dim // num_heads`. With `embed_dim=256` and `num_heads=8`, what is `head_dim`?
2. When we say "8 heads of 32," what is being divided into 8 pieces — the set of tokens, or the numbers inside one token's vector? How many tokens does head 3 see?
3. `embed_dim` is 256 and, for the reference config, the pre-mask token count `N` is also a power of two. Which of the two does `head_dim` divide, and why is the other one irrelevant to the head split?
4. True/false: increasing `num_heads` from 8 to 16 increases the number of parameters in `self.qkv`. Justify.

## B. Per-head scores

5. Using toy example A, compute `score_0(0,1)` — head 0's unscaled score between token 0 and token 1.
6. Same example, compute `score_1(0,1)`.
7. Same example, compute the score if there were a single head over all 8 dimensions. What is its relationship to your answers to 5 and 6?
8. In one sentence, state what is lost by summing all 256 products into one number instead of keeping 8 partial sums.
9. Where in `rope.py` is the variable holding `score_h` defined? Name the line, or explain why there isn't one.
10. `q`, `k`, `v` enter `F.scaled_dot_product_attention` with shape `(B, H, N, head_dim)`. Which of those four axes does the function treat as batch, and what does that imply about how the 8 heads relate to each other?

## C. The shape trace — `rope.py:79-80`

11. `x` of shape `(4, 320, 256)` enters `RopeAttention.forward` with `num_heads=8`. Give the shape after each of: `self.qkv(x)`, `.reshape(B, N, 3, num_heads, head_dim)`, `.permute(2, 0, 3, 1, 4)`, and `q` after `.unbind(0)`.
12. Write the meaning of each axis of `(3, B, H, N, head_dim)` in words.
13. `.reshape(...)` on line 79 moves no data. What does it actually change?
14. For one token, `self.qkv` produces 768 numbers. Give the index range occupied by q, by k, and by v.
15. Which of those 768 numbers holds v, head 3, dimension 5? Show the arithmetic.
16. State the general rule `flat = f(t, h, d)` where `t` is 0/1/2 for q/k/v, `h` is the head, and `d` is the dimension within the head.
17. Is the q-then-k-then-v, head-major layout in question 14 a mathematical fact about `nn.Linear`, or a convention? What makes it work either way?
18. Why is `permute` needed at all — what does putting `H` before `N` accomplish, and why does `qkv` go to axis 0?
19. What does `.unbind(0)` do, and what shape does each of `q`, `k`, `v` have afterwards for the reference config with `N` tokens?
20. `rope.py:81` is `cos_b = cos.unsqueeze(1)`, turning `(B, N, head_dim)` into `(B, 1, N, head_dim)`. What does the `1` do when it meets `q` of shape `(B, H, N, head_dim)`, and what does that mean physically about the 8 heads?

## D. `nn.Linear`

21. `nn.Linear(in_features, out_features)` stores two tensors. Name them and give both shapes.
22. Write the formula for a single output number `y[o]` in terms of the weight, the input, and the bias.
23. In plain words: what is row `o` of `.weight`, and what does it produce?
24. Using toy example B, compute `y` for `nn.Linear(3, 4)` with `x = [1, 2, 0]`.
25. Using toy example B, compute `y` for `nn.Linear(2, 3)` with `x = [3, 1]`.
26. The forward pass is written `x @ weight.T + bias`. Why the transpose? Is it a mathematical step or a storage-layout step?
27. `nn.Linear` acts on which axis of a `(B, N, D)` input? What happens to the other two axes?
28. A tensor of shape `(2, 100, 256)` passes through `nn.Linear(256, 768)`. Give the output shape, and state how many times row 613 of the weight is used in that single call.
29. Following from 28: state the consequence for whether `nn.Linear` can let token 3 influence token 40. Which component of `RopeBlock` is the only one that can?
30. Count the parameters, including biases, in each of: `self.qkv`, `self.proj`, `self.mlp[0]`, `self.mlp[3]`. Give the four numbers and the total for one `RopeBlock`. Ignore the LayerNorms.
31. `self.mlp` (rope.py:101-107) is `Linear(256, 1024) → GELU → Dropout → Linear(1024, 256) → Dropout`. What would the two Linear layers collapse into if the `GELU` were deleted?

## E. Where the numbers come from and when they change

32. Are `W` and `b` random at the start? Name the file, the method, and the two `nn.init` calls that set every `nn.Linear` in this model.
33. Xavier uniform draws each weight from a uniform distribution on `[-a, +a]`. Write the formula for `a`, then evaluate it for `self.qkv` and for `self.proj`.
34. The biases start at exactly zero while the weights start random. Why is it safe for the bias to be zero but not the weight?
35. If every entry of `self.qkv.weight` were initialized to the same constant, what would happen to the 768 rows over training, and what is the name for the property that random init provides?
36. Which lines in `rope.py` change the value of `self.qkv.weight`?
37. Name the file and line that actually writes new values into `self.qkv.weight`. Name the line that computes the gradients, and the line that constructs the optimizer.
38. `optimizer.step()` is a single call, yet it updates weights buried six blocks deep. What was passed to `AdamW` that makes this possible?
39. `rope.py:110-111` are `x = x + self.attn(...)` and `x = x + self.mlp(...)`. These reassign `x`. Why does that not count as a weight update? State the distinction between what persists and what is discarded each batch.
40. Nothing in the code labels row 613 as "v, head 3, dim 5." So what makes that row specialize to that job during training?

## F. The concatenation — `rope.py:88`

41. `F.scaled_dot_product_attention` returns shape `(B, H, N, head_dim)`. Write out line 88 and give the shape after each of its two operations, for `B=1, N=4, H=2, head_dim=4`.
42. Which function performs the concatenation — `torch.cat`, `torch.stack`, or something else?
43. After `.transpose(1, 2)` token 0's data is a 2×4 block. Write out how flattening it orders the 8 numbers.
44. What would go wrong if you called `.reshape(B, N, D)` directly on `(B, H, N, head_dim)` without the transpose?
45. At the reference config, which dimensions of the 256-long output vector hold head 5's contribution?

## G. `self.proj` — `rope.py:89`

46. Immediately after line 88, how many heads does the number sitting in output dimension 100 depend on?
47. `self.proj` is `nn.Linear(256, 256)`. Using the row-dot-product picture, explain in one or two sentences why applying it changes the answer to question 46.
48. If you wanted `proj_out[0]` to depend only on head 2, which entries of `self.proj.weight[0]` would have to be zero, and which could be nonzero?
49. Describe what would happen across the 6 stacked blocks if `self.proj` were removed entirely, given that `RopeBlock.forward` adds a residual.
50. In the textbook form `MultiHead(x) = Concat(o_0, ..., o_7) W^O`, which code object is `W^O`? And where are the per-head `W_i^Q, W_i^K, W_i^V` in this implementation?
51. Concat-then-project is equal to a sum over heads. Write that identity and say what shape each head's effective output matrix has.
52. Honest-limits question: is `self.proj` the only place in `RopeBlock` where information crosses head boundaries? What is the precise claim you can make from reading the code alone?

## H. Coupling to RoPE

53. `RopeCache.__init__` asserts `head_dim % 4 == 0` when `num_axes=2` (rope.py:45). For `num_heads=8` and `embed_dim=256`, does it pass?
54. With `head_dim=32` and `num_axes=2`, what is `axis_dims`, and how many rotation frequencies are built per axis?
55. You set `num_heads=256` with `embed_dim=256`. Give `head_dim`, write what `score_h(i,j)` reduces to arithmetically, and say whether the model builds.
56. State the general coupling in one sentence: changing `num_heads` silently changes what, in `RopeCache`?

## I. Seismic meaning

57. For `multi_1d`, one token is a patch of what, indexed by which two coordinates?
58. Name two different notions of "relevant neighbouring token" that exist simultaneously in an angle gather, and explain why a single head is forced to compromise between them.
59. Is the claim "head 0 learns the time axis and head 1 learns the angle axis" something you can read off this code? What would you have to do to check it?

---

# ANSWER KEY

1. `head_dim = 256 // 8 = 32`.

2. The **numbers inside one token's vector** — the 256-long embedding is cut into 8 contiguous slices of 32. The token set is untouched. Head 3 sees **all** `N` tokens, just through a 32-dimensional view of each.

3. `head_dim` divides `embed_dim` (the feature width), enforced by `assert dim % num_heads == 0` on line 69. `N` is the number of tokens and lives on a different axis entirely; no head ever gets a subset of tokens, so `N` plays no part in the split.

4. **False.** `self.qkv` is `nn.Linear(dim, dim*3)` = `nn.Linear(256, 768)` regardless of `num_heads` (line 73). The head count only affects how line 79 *reads* the 768 outputs. Parameter count is identical; FLOP count is essentially identical too, since `8 × (N·N·32)` equals `1 × (N·N·256)`.

5. Head 0 uses dims 0-3:
   `2*1 + 0*4 + 1*2 + 3*0 = 2 + 0 + 2 + 0 = 4`.

6. Head 1 uses dims 4-7:
   `5*3 + 1*3 + 0*1 + 2*1 = 15 + 3 + 0 + 2 = 20`.

7. Single head over all 8: `4 + 20 = 24`. It is exactly the **sum** of the per-head scores. Multi-head forms the same products; it simply declines to add them all into one pile.

8. Head 1 rated this token pair highly (20) while head 0 was nearly indifferent (4), and once summed to 24 that disagreement is unrecoverable — one number means one softmax, one ranking, and all 256 feature dimensions are forced to agree on it.

9. **There is no such variable.** The scores are computed inside `F.scaled_dot_product_attention` on line 85 and never materialize in Python. The fused kernel does q·kᵀ, scaling, softmax, and the multiply by `v` internally.

10. `B` and `H` are both treated as batch axes; only the last two (`N`, `head_dim`) participate in the attention math. The implication: the 8 heads are **fully independent** — no head's scores or softmax touch another's. The head axis is a batch axis, which is precisely why the reshape/permute is all that "multi-head" requires.

11.
```
x                 (4, 320, 256)
self.qkv(x)       (4, 320, 768)
.reshape          (4, 320, 3, 8, 32)
.permute          (3, 4, 8, 320, 32)
q after unbind    (4, 8, 320, 32)
```

12. Axis 0 selects q, k, or v. Axis 1 is the batch (which gather). Axis 2 is the head. Axis 3 is the token. Axis 4 is the dimension within that head's slice.

13. It changes only how the flat run of 768 numbers is **indexed** — the labelling, not the storage. `3 * 8 * 32 = 768`, so nothing is copied or moved.

14. `q = 0..255`, `k = 256..511`, `v = 512..767`. (Watch the boundary: a 256-long block starting at 256 ends at **511**, not 512.)

15. `flat = 512 + 3*32 + 5 = 512 + 96 + 5 = 613`.

16. `flat = t*(num_heads*head_dim) + h*head_dim + d`, i.e. `t*256 + h*32 + d` at the reference config. Rightmost index varies fastest.

17. A **convention**, imposed by the argument order in the reshape. `self.qkv` emits 768 unlabelled numbers; line 79 declares which slot is which. Any consistent convention works, because the weights are learned — the network adapts to whatever assignment you fix. A different repo may use `(B, N, num_heads, 3, head_dim)` and be equally correct. What matters is that it is identical on every forward and backward pass.

18. `F.scaled_dot_product_attention` treats everything except the final two axes as batch, so `N` and `head_dim` must occupy the last two positions and `H` must sit in front of `N` for the heads to run independently. `qkv` goes to axis 0 so that the next line can peel it off with a single `unbind(0)`.

19. It slices along axis 0 and returns three tensors (views, no copy). Each has shape `(B, 8, N, 32)`.

20. The `1` **broadcasts** across the head axis, so `cos_b` is applied identically to all 8 heads. Physically: every head receives the same positional rotation angles. Position is shared; only the content projections differ per head.

21. `weight` of shape `(out_features, in_features)`, and `bias` of shape `(out_features,)`.

22. `y[o] = sum over i of weight[o][i] * x[i] + bias[o]`, i.e.

    $$y_o = \sum_{i} W_{o,i}\,x_i + b_o$$

23. Row `o` is a learned direction of length `in_features`. It is dotted with the input vector and produces exactly one output number, `y[o]`. `out_features` rows → `out_features` output numbers.

24.
```
y[0] =  0.5*1 + (-1.0)*2 +  2.0*0 = -1.5
y[1] =  1.0*1 +   0.0*2  +  0.5*0 =  1.0
y[2] = -0.5*1 +   0.5*2  +  1.0*0 =  0.5
y[3] =  2.0*1 +   1.0*2  + (-1.0)*0 =  4.0
```
`y = [-1.5, 1.0, 0.5, 4.0]`

25.
```
y[0] = 1.0*3 + (-1.0)*1 + 1.0  = 3
y[1] = 0.0*3 +   2.0*1  + 0.0  = 2
y[2] = 3.0*3 +   1.0*1  - 2.0  = 8
```
`y = [3, 2, 8]`  — the bias is easy to drop; it is part of every output.

26. `weight` is stored as `(out, in)`. To dot all its rows against `x` in one matrix product the rows must stand as columns, which is `weight.T` of shape `(in, out)`; then `(…, in) @ (in, out) = (…, out)`. It is **storage-layout bookkeeping**, not a mathematical step.

27. Only the **last** axis (`D`). Everything to its left is treated as batch — the same weight matrix is applied independently to each of the `B*N` token vectors, and `B` and `N` pass through unchanged.

28. Output `(2, 100, 768)`. Row 613 is used `2 * 100 = 200` times — once per token vector, always with the same 256 numbers.

29. It **cannot**. Weight sharing across tokens means the layer transforms each token in isolation; there is no path from token 3's values to token 40's output. `self.attn` is the only component in `RopeBlock` that crosses the token axis. This is the entire reason attention exists.

30.
```
self.qkv     nn.Linear(256, 768)    256*768 + 768   = 197,376
self.proj    nn.Linear(256, 256)    256*256 + 256   =  65,792
self.mlp[0]  nn.Linear(256, 1024)   256*1024 + 1024 = 263,168
self.mlp[3]  nn.Linear(1024, 256)   1024*256 + 256  = 262,400
                                            total   = 788,736
```

31. Into a single `nn.Linear(256, 256)`. Two consecutive affine maps compose to one affine map (`W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)`), so the hidden width of 1024 would buy nothing. The `nn.GELU()` on line 103 is what prevents the collapse.

32. Random, and set explicitly by this repo rather than left to PyTorch's default. `models/foundation.py`, method `_init_weights`, lines 204-208:
```python
for m in self.modules():
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
```

33. $a = \sqrt{6/(\text{fan\_in} + \text{fan\_out})}$.
    `self.qkv`: `sqrt(6/(256+768)) = sqrt(6/1024) ≈ 0.0765`.
    `self.proj`: `sqrt(6/(256+256)) = sqrt(6/512) ≈ 0.1083`.
    The scale is a heuristic chosen so output variance roughly matches input variance through a deep stack — too large and activations explode over 6 blocks, too small and they vanish. It is not an optimum.

34. The weight matrix alone already makes the 768 rows different from each other, which is what matters. A zero bias just means every output starts as a pure projection with no offset; gradients will move the biases apart immediately.

35. All 768 rows would compute the same output, receive the same gradient, and update identically — they would remain identical forever, making the layer effectively one row wide. Random init provides **symmetry breaking**.

36. **None.** No line in `rope.py` ever changes a weight. `rope.py` only describes the computation performed on a forward pass.

37. `train.py:89`, `optimizer.step()`, is the only line that writes new values into the weights. `train.py:86`, `loss.backward()`, computes and stores `.grad` on every parameter. `train.py:56-61` constructs the `AdamW` optimizer.

38. `model.parameters()` (line 57). That call walks the whole module tree and collects every `nn.Parameter` — including `self.qkv.weight` and `.bias` in all 6 blocks — so the optimizer holds direct references and can update them in place.

39. Those lines rebind the local name `x` to a new **activation**: a value computed for one forward pass and discarded once gradients have flowed through it. Weights are `nn.Parameter`s that persist across batches and are edited only by the optimizer. Same distinction that decides what goes into a checkpoint — `model_state` holds parameters, never activations.

40. The reshape/permute/unbind convention is **identical on every pass**, and those ops are pure index relabellings, so during backprop the gradient routes back to exactly the flat positions it came from. The gradient arriving at row 613 was computed only through the path where that number acted as v, head 3, dim 5. That row therefore only ever receives pressure to be good at that one job. Consistency of an arbitrary convention is what turns a slot into a meaning.

41.
```python
out = out.transpose(1, 2).reshape(B, N, D)
```
```
(1, 2, 4, 4)  --transpose(1,2)-->  (1, 4, 2, 4)  --reshape-->  (1, 4, 8)
(B, H, N, hd)                      (B, N, H, hd)               (B, N, D)
```

42. Neither `cat` nor `stack` — it is a **reshape**, flattening the head axis into the feature axis. The concatenation is implicit in the memory layout.

43. Flattening reads row by row, so `[h0 h0 h0 h0 | h1 h1 h1 h1]` — head 0's four numbers, then head 1's four.

44. You would be flattening `H` into `N` rather than into `head_dim`, gluing together numbers belonging to different tokens. The result has the right shape but is meaningless.

45. Head 5 occupies dims `32*5 = 160` through `191`.

46. **One.** Dimension 100 falls in `96..127`, which is head 3's slice, and nothing else contributes to it.

47. `proj_out[o]` is row `o` of `self.proj.weight` dotted with the entire 256-long vector, so it sums contributions from dims 0-31 (head 0) through 224-255 (head 7). Every output number now reads all 8 heads at once — this is the first place their conclusions meet in a single number.

48. Entries **64 through 95** may be nonzero (head 2's slice); all other 224 entries must be zero.

49. Output dims 0-31 would be head 0's work alone, 32-63 head 1's alone, and so on. Because `RopeBlock.forward` does `x = x + self.attn(...)`, each head would be permanently wired into the same fixed 32-dim strip of the residual stream, in every one of the 6 blocks. Eight heads that never combine.

50. `W^O` is `self.proj` (line 74). The per-head `W_i^Q, W_i^K, W_i^V` are **not separate objects** — all 24 of them are fused into the single `nn.Linear(256, 768)` on line 73, and the per-head split is recovered by slicing in the reshape on line 79. Same math, one matmul instead of 24. This is the main place where the implementation departs in form (not in content) from the textbook equation.

51. Because `out` is a concatenation, the product splits:

    $$\text{out}\,W^{O\top} = \sum_{h=0}^{7} o_h\, W^{O\top}_{[32h:32h+32,\ :]}$$

    Each head's effective output matrix is `(256, 32)`: it maps that head's 32 numbers into the full 256-dim space, and the eight results are summed. Concat-then-project and per-head-project-then-sum are the same operation.

52. No — `self.mlp` also mixes across all 256 dims. The precise claim from the code alone: `self.proj` is the only mixing that happens **before the residual add on line 110**. How much removing it would degrade this model is an empirical question the code does not answer; it would need an ablation.

53. `head_dim = 32`, and `32 % 4 == 0`, so it passes.

54. `axis_dims = [16, 16]` (rope.py:46). `_rope_freqs(16, base)` produces `16/2 = 8` frequencies per axis — visible in the `(B, N, axis_dim/2)` comment on line 59, before `torch.cat([c, c])` doubles it back to 16.

55. `head_dim = 256 // 256 = 1`. Each score reduces to a **single product**, `score_h(i,j) = q[i][h] * k[j][h]` — no summation at all. The model does **not** build: `rope.py:45` asserts `head_dim % 4 == 0` for 2-axis RoPE, and `1 % 4 = 1`, so it raises. RoPE needs a factor of 4 because it splits `head_dim` into two axes and then each axis into cos/sin halves.

56. `head_dim` is passed to `RopeCache`, so changing `num_heads` changes the length of the RoPE frequency ladder — and can break the divisibility assertion.

57. A patch of an angle gather, indexed by (angle/trace, time window). From Unit 3: `angle = k // 64`, `time = k % 64`, with 64 = `trace_length / patch_size` = 1024/16.

58. (i) Same angle, nearby time — the wavelet and moveout direction. (ii) Same time, different angle — the AVO direction. A single head produces one softmax per token, hence one ranking, so it must blend the two into a single compromise. Eight heads can hold both notions at once and let `self.proj` combine them.

59. **No.** The architecture permits specialization; it does not demonstrate it. To check you would have to train the model and inspect the attention weights per head — e.g. for a given query token, plot each head's weights back onto the (angle, time) grid and look for structure along one axis. That is an experiment, not something readable from the source.
