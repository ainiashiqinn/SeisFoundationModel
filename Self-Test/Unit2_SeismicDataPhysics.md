# Unit 2 Self-Test — Seismic Data Physics (Ricker Wavelet, Moveout, AVO)

Cover the answer key. Answer each cold, then scroll down to check.

---

## A. Ricker wavelet — definition and derivation

1. The Ricker wavelet is defined as the (negative, normalized) *n*-th derivative of a Gaussian. What is *n*?
2. Why does a plain Gaussian pulse make a physically bad seismic wavelet? What does differentiating fix?
3. Why must *n* be even rather than odd, given that real seismic wavelets in this codebase are modeled as zero-phase and symmetric?

## B. Ricker wavelet — code correspondence and behavior

4. Write `_ricker`'s formula in LaTeX, and give the symbol ↔ variable ↔ meaning table (include units/dimensionality caveats).
5. Is `f` in the code literally "Hz"? What assumption would you need to make about the `t` axis for that to be true?

## C. Hyperbolic moveout — formula and meaning

6. Write the moveout formula used in `_gen_multi`, and give the symbol ↔ variable ↔ meaning table.
7. At `offset = 0`, what does `tau` reduce to? Which parameter has zero influence at that point?
8. Between `t0` and `v`, which one controls the *apex height* of an event in a gather plot, and which controls how *steep* the moveout curve is?
9. Name the general geophysical formula/regime this equation matches (what real-world gather geometry does it describe?).

## D. Hand computation — moveout

10. Compute `tau` for `t0 = 0.4`, `v = 1.2`, `offset = 0.6`.
11. Compute `tau` for `t0 = 0.2`, `v = 0.5`, `offset = 1.0`. Is the result inside the `[0, 1]` normalized time axis?
12. Based on Q11, is the sampled `v` range `[0.5, 2.0)` guaranteed to keep every event inside the `[0, 1]` window at every offset? What kind of guarantee (if any) does it actually provide?

## E. Noise model

13. Write the additive noise model used in `_gen_1d`/`_gen_multi`/`_gen_2d` in LaTeX (define $\sigma$).
14. Is the noise variance calibrated to a fixed SNR across samples? Why or why not, given how `amp` is sampled?
15. In `_gen_multi`, is the noise independent or correlated across the `C` traces? What general fact about `torch.randn` explains this, regardless of whether you call it once on `(C, T)` or `C` times on `(T,)`?
16. What kind of real seismic noise does this white-noise model *fail* to capture, and what property would that noise need (hint: think about how the signal events themselves are constructed) that the current model lacks entirely?

## F. Joint normalization and AVO

17. `utils/normalize.py` computes normalization statistics "jointly" rather than "per-trace." What exactly does "jointly" mean here, mechanically (which tensor elements go into one statistic)?
18. What is AVO, in one sentence, and why would per-trace normalization destroy it?
19. For which of the three input types (`1d`, `multi_1d`, `2d`) does the joint-vs-per-trace distinction matter, and for which is it vacuous? Explain why `2d` belongs where it does — look at how `_gen_2d`'s moveout loop is structured.

## G. Angle gather vs. offset gather

20. Define, in standard geophysical terms (not from this codebase), what distinguishes an offset/CMP gather from a true (migrated) angle gather.
21. What does `_gen_multi` actually generate — offset-domain or angle-domain physics? Point to the specific line/formula that proves it.
22. What would a correctly migrated angle gather look like across the trace axis, in contrast to what `_gen_multi` produces?

## H. Putting it together

23. You plot a `_gen_multi` sample and see two hyperbolic events: one with an apex around `t=0.6` and steep legs, another with an apex around `t=0.75` and much flatter legs. Which of the two events has the smaller `v`? Which has the smaller `t0`?
24. `num_traces = 32` (even). Does any trace sit at exactly `offset = 0`? What are the two closest trace indices to zero offset, and roughly what offset do they sit at?

---

# ANSWER KEY

1. The 2nd derivative ($n=2$).
2. A plain Gaussian has nonzero area (nonzero DC/zero-frequency component), but real seismic instruments and sources record transient motion with no sustained static offset. Differentiating removes the DC component (each derivative multiplies the spectrum by $i\omega$, killing $\omega=0$).
3. An odd derivative of a symmetric function (Gaussian) is antisymmetric (odd function) — it would not be a symmetric, zero-phase pulse. An even derivative preserves the even symmetry of the Gaussian, giving a symmetric, zero-phase wavelet.
4. $r(t) = \left(1-2\pi^2f^2(t-t_0)^2\right)e^{-\pi^2f^2(t-t_0)^2}$.

    | Symbol | Variable | Meaning | Note |
    |---|---|---|---|
    | $t$ | `t` | normalized time axis | dimensionless, `linspace(0,1,T)`, not seconds |
    | $t_0$ | `t0` | wavelet center | in $[0,1]$ |
    | $f$ | `f` | controls Gaussian width | units are 1/normalized-time, not Hz, unless the axis is assumed = 1 real second |
    | $r$ | return value | amplitude | peak-normalized to 1 |
5. Not necessarily — `f`'s value (10–30) only corresponds to literal Hz if you additionally assume the normalized `[0,1]` time axis represents exactly 1 real second. That mapping is never stated in the code, so it's a plausible but unverified assumption.
6. $\tau(x) = \sqrt{t_0^2 + (x/v)^2}$.

    | Symbol | Variable | Meaning |
    |---|---|---|
    | $\tau$ | `tau` | arrival time at a given trace/offset |
    | $t_0$ | `t0` | zero-offset two-way time |
    | $x$ | `offsets[ci]` | offset proxy, dimensionless, `linspace(-1,1,C)` |
    | $v$ | `v` | "apparent velocity," dimensionless, $[0.5, 2.0)$ |
7. `tau` reduces to exactly `t0`. `v` has zero influence at `offset = 0`, since the `(x/v)^2` term vanishes regardless of `v`.
8. `t0` controls apex height (the vertex value at `x=0`); `v` controls curvature/steepness (smaller `v` → steeper legs, larger `v` → flatter).
9. Hyperbolic normal-moveout (NMO) geometry of an offset/CMP gather — a single flat reflector under a constant-velocity layer.
10. $\tau=\sqrt{0.4^2+(0.6/1.2)^2}=\sqrt{0.16+0.25}=\sqrt{0.41}\approx 0.640$.
11. $\tau=\sqrt{0.2^2+(1/0.5)^2}=\sqrt{0.04+4}=\sqrt{4.04}\approx 2.01$. Outside `[0,1]` — off the sampled time axis entirely.
12. No — not guaranteed. Q11 is exactly the counterexample: even at the narrower `t0` range's low end, extreme `v`/offset combinations still push `tau` well past `1.0`. The narrower `t0 ∈ [0.2,0.8)` range only reduces how *often* this happens (since `v` is usually not at its extreme), it doesn't bound it.
13. $x_{\text{noisy}}(t) = x_{\text{clean}}(t) + \sigma\,\varepsilon(t)$, $\varepsilon(t)\sim\mathcal{N}(0,1)$ i.i.d., $\sigma = 0.05$.
14. No. `amp` (wavelet amplitude) is drawn from `[0.2, 1.0)` per event, but `sigma=0.05` is fixed regardless, so the effective signal-to-noise ratio varies from sample to sample rather than being held constant.
15. Independent. `torch.randn` draws one i.i.d. sample per output tensor element regardless of the tensor's shape or how many separate calls produce it — calling it once on `(C,T)` or `C` times on `(T,)` gives statistically identical (independent) results.
16. Coherent noise (e.g. ground roll) — a real, spatially/offset-correlated noise source. The current model has no mechanism at all for correlating noise across traces; a coherent-noise addition would need its own offset-dependent moveout term, similar to how the signal events get `tau` from the offset/velocity relationship (though ground roll's true moveout is linear in offset, not hyperbolic like reflections).
17. Every statistic (`mean`, `std`, `min`, `max`, `(x**2).mean()`, `abs().max()`) is computed over *all* elements of the tensor at once — e.g. for a `(C,T)` gather, one scalar is computed from all `C×T` values combined, and that single scalar rescales every trace by the same amount.
18. AVO (amplitude versus offset/angle) is the change in a reflection event's amplitude across traces recorded at different offsets/angles, which is diagnostic of subsurface elastic/fluid properties. Per-trace normalization would rescale each trace to its own independent scale, erasing exactly that relative-amplitude pattern between near and far traces.
19. It's vacuous for `1d` (only one trace exists — joint and per-trace stats are the same computation) and it matters for both `multi_1d` and `2d`. `_gen_2d`'s moveout loop (`for wi in range(W): patch[:, wi] = ...`) shows `W` plays the same offset/trace-axis role as `C` in `_gen_multi`, just reshaped into `(1, H, W)` — so `2d` carries the same AVO-preservation concern as `multi_1d`, even though it's stored as an "image."
20. An offset/CMP gather sorts traces by surface source-receiver offset, with reflection arrivals following hyperbolic NMO moveout. A true angle gather sorts traces by angle of incidence at the reflector — obtained via a velocity model and ray tracing (Snell's law / ray parameter) — and if the velocity model and migration are correct, a flat reflector appears *flat* (no time curvature) across the angle axis, with only amplitude varying by angle.
21. Offset-domain. `_gen_multi`'s `tau = sqrt(t0**2 + (offsets[ci]/v)**2)` (lines 73–79) is exactly the hyperbolic NMO formula parameterized directly by offset and a constant velocity — no ray tracing, no velocity model, no offset→angle conversion anywhere in the function.
22. A correctly migrated angle gather would show the event at the *same* arrival time across all traces (flat, no curvature), with only the amplitude changing systematically across the angle axis — instead of the curved arrival time `_gen_multi` actually produces.
23. The steep-legged event (apex `~0.6`) has the smaller `v` (small `v` → large `offset/v` → steep curvature). Apex height is controlled only by `t0`, so the event with the lower apex time (`~0.6`) has the smaller `t0`; `v` and `t0` are independent choices per event, so "smaller `v`" and "smaller `t0`" both describe the same event here, but for different reasons — steepness comes from `v`, apex height comes from `t0`, not from the same parameter.
24. No — `num_traces=32` is even, so `linspace(-1,1,32)` has no sample exactly at 0 (step $= 2/31 \approx 0.0645$; solving $-1+i\cdot(2/31)=0$ gives $i=15.5$, not an integer). The two closest traces are indices 15 and 16, sitting at offset $\approx \mp0.032$.
