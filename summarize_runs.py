import numpy as np

runs = {
    "CNN-only":      [(0.8670, 0.8609, 0.8806, 0.8670),
                      (0.8333, 0.8210, 0.8605, 0.8333),
                      (0.8510, 0.8411, 0.8761, 0.8510)],
    "GNN-only":      [(0.8654, 0.8593, 0.8782, 0.8654),
                      (0.8205, 0.8040, 0.8582, 0.8205),
                      (0.8269, 0.8119, 0.8622, 0.8269)],
    "Fixed 0.5/0.5": [(0.8606, 0.8531, 0.8783, 0.8606),
                      (0.8478, 0.8380, 0.8706, 0.8478),
                      (0.8718, 0.8656, 0.8879, 0.8718)],
    "Adaptive":      [(0.8878, 0.8841, 0.8961, 0.8878),
                      (0.8638, 0.8570, 0.8794, 0.8638),
                      (0.8734, 0.8676, 0.8879, 0.8734)],
}

metrics = ["Accuracy", "F1", "Precision", "Recall"]

print(f"{'Model':<16}", end="")
for m in metrics:
    print(f"  {m:>18}", end="")
print()
print("-" * 90)

for model, rows in runs.items():
    arr = np.array(rows)
    print(f"{model:<16}", end="")
    for i in range(4):
        mean = arr[:, i].mean()
        std  = arr[:, i].std(ddof=1)
        print(f"  {mean:.4f} ± {std:.4f}", end="")
    print()