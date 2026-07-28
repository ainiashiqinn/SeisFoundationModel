"""
Quick visualization of one SyntheticSeismicDataset._gen_multi() sample.

_gen_multi builds a synthetic multi-trace gather: shape (C, T) = (num_traces,
trace_length), with 1-3 hyperbolic-moveout Ricker events plus white noise
(see data/dataset.py lines 65-81).

Run: python plot_gen_multi.py
"""
import matplotlib.pyplot as plt

from data.dataset import SyntheticSeismicDataset

# Matches configs/config.yaml's multi_1d settings.
cfg = {
    "input_type": "multi_1d",
    "trace_length": 1024,
    "num_traces": 32,
}

dataset = SyntheticSeismicDataset(cfg, num_samples=1, seed=0, normalize_mode="none")
sample = dataset[0]  # (C, T) -- see _gen_multi's return, line 81
print("sample shape:", tuple(sample.shape))

gather = sample.numpy()  # (num_traces, trace_length)

fig, ax = plt.subplots(figsize=(6, 8))
im = ax.imshow(
    gather.T,              # (T, C): row 0 = t=0 (top), matches seismic-section convention
    aspect="auto",
    cmap="seismic",
    extent=[0, gather.shape[0], 1, 0],  # x = trace index, y = normalized time (0 top -> 1 bottom)
)
ax.set_xlabel("trace index (offset proxy)")
ax.set_ylabel("normalized time")
ax.set_title("_gen_multi sample (seed=0)")
fig.colorbar(im, ax=ax, label="amplitude")
plt.tight_layout()
plt.savefig("gen_multi_sample.png", dpi=150)
plt.show()
print("Saved to gen_multi_sample.png")
