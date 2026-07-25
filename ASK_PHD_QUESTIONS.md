# Questions for the original author (ASK-PHD)

Running list of questions about SeisFoundationModel that can't be answered from the code and references alone. Grouped by file.

## data/dataset.py

1. **`_gen_multi` / `_gen_2d`, `t0` sampling range** (lines 73, 91): `t0 ∈ [0.2, 0.8)`, versus `_gen_1d`'s `t0 ∈ [0.05, 0.95)` (line 58). Is the narrower range meant to guarantee that moveout-shifted arrivals (`tau = sqrt(t0**2 + (offset/v)**2)`, always `>= t0`) stay inside the `t ∈ [0, 1]` window, or is it just an empirical margin? Worst case — `v = 0.5` (its minimum), `offset = ±1` (its extreme) — still pushes `tau` past `1.0` even at `t0 = 0.2`, so containment isn't actually guaranteed by this range alone.

2. **Apparent velocity range** (lines 74, 92): `v ∈ [0.5, 2.0)`. Both `t` and `offsets` are normalized, dimensionless axes (`t ∈ [0,1]`, `offset ∈ [-1,1]`), not real seconds/meters. Is this range tied to any real velocity distribution or unit convention, or is it purely empirical, chosen to produce a visually reasonable range of moveout curvature (steep to nearly flat) in the normalized domain?

3. **Noise model** (lines 62, 80, 98): additive white Gaussian noise, `sigma = 0.05`, independent across time and (in `_gen_multi`) across traces. Is `0.05` calibrated to any target SNR? Since wavelet amplitude `amp` is drawn from `[0.2, 1.0]`, effective SNR varies per sample rather than being controlled — intentional, or an oversight? Also: was coherent noise (e.g., ground roll, which has offset-dependent moveout like the signal events) intentionally excluded from the synthetic generator, or just not gotten to yet?
</content>
