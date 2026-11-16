"""
Fig 5.7 - Distribution of learned fusion weight alpha (adaptive model, test set)
Usage: python make_fig_5_7.py alphas.npy      (or alphas.csv, one value per line)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "alphas.npy"
def to_flat(obj):
    """Flatten arrays / lists of batches / tensors / dicts into one 1-D float array."""
    if isinstance(obj, dict):
        key = next((k for k in obj if "alpha" in str(k).lower()), list(obj)[0])
        print(f"dict found, using key: {key!r}")
        return to_flat(obj[key])
    if isinstance(obj, (list, tuple)) or (isinstance(obj, np.ndarray) and obj.dtype == object):
        items = obj.tolist() if isinstance(obj, np.ndarray) else obj
        if isinstance(items, dict):
            return to_flat(items)
        return np.concatenate([to_flat(i) for i in items])
    if hasattr(obj, "detach"):
        obj = obj.detach().cpu().numpy()
    return np.asarray(obj, dtype=float).ravel()

if path.endswith(".npy"):
    raw = np.load(path, allow_pickle=True)
    if raw.dtype == object and raw.ndim == 0:      # np.save(dict) -> 0-d object array
        raw = raw.item()
else:
    raw = np.loadtxt(path, delimiter=",")
a = to_flat(raw)
print(f"n={len(a)}  mean={a.mean():.4f}  std={a.std():.4f}  min={a.min():.4f}  max={a.max():.4f}")

fig, ax = plt.subplots(figsize=(7, 4.4))
ax.hist(a, bins=np.linspace(0.4, 0.6, 41), color="#4C72B0", edgecolor="white")
ax.axvline(0.5, color="#C62828", ls="--", lw=1.8, label="α = 0.5 (equal weighting)")
ax.axvline(a.mean(), color="black", ls=":", lw=1.5, label=f"mean α = {a.mean():.3f} (std {a.std():.3f})")
ax.set_xlim(0.4, 0.6)
ax.set_xlabel("Learned fusion weight α"); ax.set_ylabel("Number of test samples")
ax.set_title(f"Distribution of learned α on the test set (n = {len(a)})")
ax.legend(frameon=False); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig("fig_5_7_alpha_distribution.png", dpi=300)
print("Saved fig_5_7_alpha_distribution.png")