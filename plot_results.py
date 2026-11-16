import matplotlib.pyplot as plt
import numpy as np

# Models
models = ["CNN-only", "GCN-only", "Fixed 0.5/0.5", "Adaptive"]
x = np.arange(len(models))

# Accuracy (mean ± std)
acc_mean = np.array([0.8595, 0.8723, 0.8611, 0.8510])
acc_std  = np.array([0.0299, 0.0026, 0.0165, 0.0081])

# F1 (mean ± std)
f1_mean = np.array([0.8513, 0.8669, 0.8536, 0.8417])
f1_std  = np.array([0.0348, 0.0026, 0.0179, 0.0095])

plt.style.use("seaborn-v0_8-whitegrid")

# ---------- Fig. 1: Accuracy bar chart ----------
fig, ax = plt.subplots(figsize=(6, 4))

bars_acc = ax.bar(x, acc_mean, yerr=acc_std, capsize=5, width=0.6, color="#4C72B0", label="Accuracy")

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.set_ylabel("Accuracy")
ax.set_ylim(0.80, 0.90)

for bar in bars_acc:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
plt.savefig("fig1_accuracy.png", dpi=300)
plt.close()


# ---------- Fig. 2: Grouped bar chart (Accuracy + F1) ----------
width = 0.35
x_acc = x - width/2
x_f1  = x + width/2

fig, ax = plt.subplots(figsize=(7, 4))

bars_acc = ax.bar(x_acc, acc_mean, width, yerr=acc_std, capsize=5, color="#4C72B0", label="Accuracy")
bars_f1  = ax.bar(x_f1, f1_mean, width, yerr=f1_std, capsize=5, color="#55A630", label="Weighted F1")

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.set_ylabel("Score")
ax.set_ylim(0.80, 0.90)
ax.legend()

# Add value labels
for bar in bars_acc:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

for bar in bars_f1:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

plt.tight_layout()
plt.savefig("fig2_grouped_acc_f1.png", dpi=300)
plt.close()

print("Saved fig1_accuracy.png and fig2_grouped_acc_f1.png")