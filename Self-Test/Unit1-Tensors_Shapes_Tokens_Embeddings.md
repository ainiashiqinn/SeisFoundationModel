# Unit 1 Self-Test — Tensors, Shapes, Tokens, Embeddings

Cover the answer key. Answer each cold, then scroll down to check.

---

## A. Tensors

1. What is a tensor? Name the rank-0, rank-1, and rank-2 special cases.
2. In one sentence, what distinguishes a "scalar," a "vector," a "matrix," and a general "tensor"?

## B. The shape (B, N, D)

3. What do B, N, and D each stand for?
4. Read the shape `(4, 2048, 256)` out loud as a sentence ("4 ... , each a ... , each ...").
5. Are N and D independent knobs, or does changing one force the other to change? Which hyperparameter controls each?

## C. Batch vs sample vs patch

6. Define "sample" and "batch" — and state the one-line rule that keeps them apart.
7. True/false: a 224×224 image split into 196 patches is "196 samples." Explain.
8. In seismic, what is one sample? What plays the role of the batch? What plays the role of the patch?
9. Which axis of the tensor would disappear if you only had one example?

## D. Tokens and N

10. What is a token, in one sentence?
11. How is N chosen for text vs for images/seismic? Why is one a formula and the other not?
12. Write the formula for N in `multi_1d` and evaluate it for num_traces=32, trace_length=1024, patch_size=16.
13. If you halve the patch size, what happens to N? If you double the number of traces?
14. Does text have patches? Why or why not?

## E. Patching as tokenization

15. Complete: "Patching is one *kind* of tokenization — the kind used for ______ data."
16. For a 224×224 image with 16×16 patches, compute N.
17. For a seismic gather with 4 traces, trace_length 1024, patch_size 32, compute N.

## F. Embedding — the projection

18. What does "projection" mean in one sentence, and what is its formula?
19. Where does the number 768 come from for an RGB image patch?
20. For a `multi_1d` seismic patch, how many raw numbers are in one patch (before embedding)?
21. Give the shape of the weight matrix W that projects a seismic patch to embed_dim=256.
22. Does the seismic embedding *compress* or *expand* the raw patch? What about the image example (768 → 32)? What determines the direction?

## G. Embedding — text is different

23. How is a text token represented before embedding (its data type)?
24. How does text embedding turn a token into a vector — by multiplying or by looking up? Name the object it looks into and its shape.
25. Fill the table from memory:

| domain | raw token | embedding operation | formula |
|---|---|---|---|
| Image | ? | ? | ? |
| Seismic | ? | ? | ? |
| Text | ? | ? | ? |

## H. Where W and b come from

26. Are W and b set by hand or learned? What are they initialized to, and why "small and controlled" rather than arbitrary?
27. Name the four repeating stages of the training loop that turn a random W into a good one.
28. What defines "good" for a weight — is it an abstract notion or a specific number?
29. Classify each as parameter or hyperparameter: W, b, patch_size, embed_dim, learning rate. Which kind does gradient descent move?
30. In this repo, which layer's weights physically hold the `multi_1d` projection matrix W?

## I. Putting it together

31. Trace the shape for `multi_1d`: raw gather `(B, 32, 1024)` → after tokenize+embed → `(B, ?, ?)`.
32. In `(B, 2048, 256)`, which single number changes if you (a) batch twice as many gathers, (b) halve the patch size, (c) set embed_dim=512?

---

# ANSWER KEY

1. A tensor is a single- or multi-dimensional array of values. Rank-0 = scalar, rank-1 = vector, rank-2 = matrix.
2. Scalar = one number (0D); vector = 1D array; matrix = 2D array; tensor = general N-D array.
3. B = batch (number of independent samples); N = sequence length (number of tokens per sample); D = feature/embedding dimension (length of each token's vector).
4. "4 samples, each a sequence of 2048 tokens, each token a 256-long vector."
5. Independent. N is controlled by patch_size (via the patch count); D is controlled by embed_dim.
6. Sample = one complete example. Batch = how many samples are processed together. Rule: **sample = one example; batch = the count of them.**
7. False. The 196 patches are *inside* one sample. The image is one sample; batch counts whole images, not patches.
8. One sample = one seismic gather. Batch = how many gathers stacked together. Patch = a 16-sample time window on one trace (subdivides one gather).
9. The batch axis (B).
10. A token is the unit the model treats as one element of the input sequence.
11. Text: chosen by a tokenizer against a vocabulary → depends on the text, not a formula. Image/seismic: fixed grid of patches → N = a formula. Grid data can be chopped regularly; language can't.
12. N = num_traces × (trace_length / patch_size) = 32 × (1024/16) = 32 × 64 = 2048.
13. Halving patch size doubles N (16→8 gives 4096). Doubling traces doubles N.
14. No. Patching means chopping a continuous grid signal into fixed windows; text is discrete symbols split by a tokenizer, not a grid.
15. "...used for **grid-structured / continuous signal** data."
16. N = (224/16) × (224/16) = 14 × 14 = 196.
17. N = 4 × (1024/32) = 4 × 32 = 128.
18. Projection = multiplying a vector by a learned weight matrix (plus bias) to change its length. Formula: z = W p + b.
19. 16 wide × 16 tall × 3 color channels = 768.
20. 16 (a 16-sample window on a single-channel trace → 16 amplitudes).
21. W ∈ ℝ^(256 × 16).
22. Seismic expands (16 → 256). Image example compresses (768 → 32). Direction depends on whether D is larger or smaller than the raw patch size.
23. A discrete integer id (an index into a vocabulary).
24. By **looking up**, not multiplying. It looks into an embedding table E ∈ ℝ^(V × D), where V = vocab size; embedding = row E[id].
25. Image: raw = vector of pixels (e.g. ℝ^768), op = projection, z = Wp+b. Seismic: raw = vector of 16 amplitudes (ℝ^16), op = projection, z = Wp+b. Text: raw = one integer id, op = lookup, z = E[id].
26. Learned, not set by hand. Initialized to small random values from a controlled scheme (truncated-normal / Xavier, std 0.02 here). Small-and-controlled avoids exploding/vanishing values that would make the first forward pass blow up (nan loss).
27. Forward → loss → backward (backprop computes gradients) → optimizer step (AdamW updates weights). Repeat.
28. A specific number: the loss. "Good W" = a W that makes the loss small. Change the loss, you change what "good" means.
29. Parameters: W, b (learned). Hyperparameters: patch_size, embed_dim, learning rate (set by hand in config.yaml). Gradient descent moves only the parameters.
30. The `nn.Conv2d` weights inside `PatchEmbedMulti1D` (foundation.py / models/patch_embed.py).
31. `(B, 32, 1024)` → `(B, 2048, 256)`.
32. (a) `(2B, 2048, 256)`; (b) `(B, 4096, 256)`; (c) `(B, 2048, 512)`.
