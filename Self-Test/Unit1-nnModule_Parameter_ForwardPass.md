# Unit 1 Self-Test — nn.Module, Parameter, Forward Pass

Cover the answer key. Answer each cold, then scroll down to check.

---

## A. What is nn.Module

1. What is `nn.Module`, in one sentence? What are the two methods every subclass defines at minimum?
2. What kind of code goes in `__init__`? What kind goes in `forward`?

## B. Registration mechanics

3. What method does `nn.Module` override to make `self.proj = nn.Conv1d(...)` behave differently from an ordinary Python attribute assignment?
4. Name the two internal registries an `nn.Module` instance keeps, and what gets filed into each one.
5. Suppose you wrote `self.proj = [nn.Conv1d(...)]` — a plain Python list *containing* a Conv1d, instead of assigning the Conv1d directly. Would `proj`'s weights show up in `model.parameters()`? Why or why not?

## C. Conv1d as patch embedding

6. In `PatchEmbed1D` (`models/patch_embed.py`), `self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)`. Why does setting `stride == kernel_size` turn this into non-overlapping patches?
7. Toy example: `patch_size=2`, one output channel, weights `w0, w1`, bias `b`. Write the formula for the first output value `y0` in terms of `w0, w1, x0, x1, b`.
8. For `Conv1d(in_channels=1, embed_dim=128, kernel_size=20)`, what shape is `self.proj.weight`? Why is it 3D and not 2D?
9. What shape is `self.proj.bias`?
10. Computational: `PatchEmbed1D(trace_length=1000, patch_size=20, embed_dim=128)`, input `x` of shape `(4, 1000)`. Give the shape after `unsqueeze`, after `self.proj(x)`, and after the final `transpose`.
11. Computational: same class, `trace_length=200, patch_size=25, embed_dim=64`, input `(6, 200)`. What's the final output shape of `forward`?

## D. forward vs. `__call__`, and hooks

12. When you write `model(x)`, what method does Python actually invoke?
13. List, in order, the three things `nn.Module.__call__` does around your `forward` code.
14. Why is calling `model.forward(x)` directly considered wrong, even when it produces the exact same numeric output?
15. Does `SeisFoundation` (this repo) register any hooks anywhere? What does that mean for calling `.forward()` directly *in this specific codebase, today*?

## E. nn.Parameter and model.parameters()

16. What two things distinguish an `nn.Parameter` from a plain `torch.Tensor`?
17. Suppose `foundation.py` line 62 read `self.cls_token = torch.zeros(1, 1, embed_dim)` instead of wrapping it in `nn.Parameter`. Would `cls_token` appear in `model.parameters()`? Would it receive a `.grad` after `loss.backward()`? Would `optimizer.step()` change its value?
18. Describe, mechanically, what `model.parameters()` does when it's called — what does it look at, and how does it handle submodules?
19. `SeisFoundation.patch_embed.proj.weight` is three levels of nesting deep. Explain how `model.parameters()` on `SeisFoundation` reaches it.
20. In one sentence, what is `model.parameters()` used for once you have it?

## F. Putting it together

21. Where in the code is a `PatchEmbed1D` / `PatchEmbedMulti1D` / `PatchEmbed2D` object actually constructed, and which function decides which one gets built?
22. Once built, where does `SeisFoundation` store that object as an attribute, and on which line does it get called during the forward pass?
23. `forward(self, x)` only takes `x` as an argument, yet the method body refers to `self.proj`, `self.patch_size`, etc. Given those aren't passed in, where do they come from?

---

# ANSWER KEY

1. `nn.Module` is the base class for building a piece of a neural network — a container that holds learnable weights and defines a computation on tensors. Every subclass defines `__init__` (build/register the pieces) and `forward` (define what happens to the input).
2. `__init__`: construct and register the layers/parameters this module owns (things that persist and get learned). `forward`: the actual dataflow — the sequence of tensor operations that turns input into output, run fresh on every call.
3. `__setattr__`. `nn.Module` overrides it so that every `self.x = value` assignment is inspected: if `value` is itself an `nn.Module`, it's filed into a special registry instead of becoming an ordinary instance attribute.
4. `self._modules` (any `nn.Module` assigned to `self`, e.g. `self.proj`) and `self._parameters` (any `nn.Parameter` assigned to `self`, e.g. weights/biases created directly on this module). Both are populated by the same `__setattr__` override, keyed on the type of the value being assigned.
5. No. `__setattr__` only inspects the value being assigned *directly* to `self`. A Python list is not itself an `nn.Module`, so `self.proj` becomes an ordinary attribute holding a list — nothing inside it gets registered, and `model.parameters()` won't find the Conv1d's weights. (This is a real, well-known PyTorch gotcha — the fix is `nn.ModuleList`, which *is* registration-aware.)
6. With `stride == kernel_size`, each output position starts exactly where the previous window ended — window 0 covers samples `[0:P)`, window 1 covers `[P:2P)`, and so on, with no sample shared between two windows and none skipped.
7. `y0 = w0*x0 + w1*x1 + b`.
8. `(128, 1, 20)` — `(out_channels, in_channels, kernel_size)`. It's 3D because PyTorch always keeps the input-channel axis explicit, even when `in_channels=1`; it only *looks* like a 2D matrix here because that axis happens to have size 1. With `in_channels > 1` it would be a genuinely 3D object.
9. `(128,)` — one bias value per output channel, a 1D tensor.
10. `(4, 1000) → (4, 1, 1000)` after `unsqueeze` → `(4, 128, 50)` after `self.proj(x)` (`1000/20 = 50` patches) → `(4, 50, 128)` after `transpose`.
11. `(6, 200) → (6, 1, 200) → (6, 64, 8)` (`200/25 = 8`) → `(6, 8, 64)`.
12. `model.__call__(x)` — writing `model(x)` is Python's normal syntax for "call this object," and `nn.Module` defines `__call__`.
13. (1) Run any registered forward-pre-hooks (functions that can inspect/modify the input before `forward` runs). (2) Run `forward` itself. (3) Run any registered forward-hooks (functions that can inspect/modify the output after `forward` finishes).
14. Calling `.forward()` directly skips steps 1 and 3 — any hooks registered on the module silently never fire, even though the raw computation is identical. It also breaks the convention every other part of the PyTorch ecosystem (autograd wrappers, `DataParallel`, JIT tracing) assumes you follow.
15. No — confirmed by grepping the repo for `register_forward_hook` / `register_forward_pre_hook` / `register_backward_hook`: no matches anywhere. So today, `self.patch_embed(x)` and `self.patch_embed.forward(x)` would produce identical output in this specific repo — but `self.patch_embed(x)` is still the correct form to write, since it costs nothing and stays correct if hooks are ever added.
16. (1) `requires_grad=True` by default, so autograd tracks gradients through it. (2) `nn.Module`'s `__setattr__` specifically checks for `nn.Parameter` and files it into `self._parameters` — a plain tensor assigned the same way is not registered anywhere special.
17. No to all three. Without the `nn.Parameter` wrapper it's an inert plain tensor: not registered in `_parameters`, so `model.parameters()` won't yield it; `requires_grad=False` by default, so `loss.backward()` won't populate a `.grad` for it; and since it's absent from `model.parameters()`, it was never handed to the optimizer, so `optimizer.step()` has no way to touch it.
18. It looks at `self._parameters` on the current module and yields every entry, then for each child module registered in `self._modules`, it calls that child's `parameters()` again — the same function, recursively, one level down — and yields whatever that call returns. It has no global view; it only ever knows about its own direct `_parameters` and `_modules`, and the recursion is what covers the whole tree.
19. `SeisFoundation.parameters()` finds `patch_embed` in its own `_modules` and calls `patch_embed.parameters()`. That call finds `proj` in `patch_embed`'s `_modules` and calls `proj.parameters()`. That call finds `weight` directly in `proj`'s own `_parameters` and yields it. Three recursive calls, one per level of nesting — nothing "reaches through" three levels at once.
20. It produces the complete list of learnable tensors in the model, which gets handed to the optimizer (`torch.optim.Adam(model.parameters(), ...)`) so it knows exactly what it's allowed to update.
21. `build_patch_embed(cfg)` in `models/patch_embed.py` (lines ~126–150) — it reads `cfg["input_type"]` and constructs `PatchEmbed1D`, `PatchEmbedMulti1D`, or `PatchEmbed2D` accordingly.
22. `SeisFoundation.__init__`, line 53: `self.patch_embed = build_patch_embed(cfg)`. It gets called during the forward pass in `forward_encoder`, line 265: `x = self.patch_embed(x)`.
23. They were set as attributes on `self` during `__init__` at construction time (e.g. `self.patch_size = patch_size`, `self.proj = nn.Conv1d(...)`), and they persist on that object for its whole lifetime. `forward` doesn't need them passed in because `self` — the specific configured object the method is running on — already carries them.
