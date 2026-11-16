import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from medmnist import PneumoniaMNIST
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix, accuracy_score

from model_classical import CNNGNNFusionModel, feature_map_to_graph

DEVICE = torch.device("cpu")
model = CNNGNNFusionModel(num_classes=2, cnn_in_channels=1).to(DEVICE)
model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
model.eval()

test_ds = PneumoniaMNIST(split="test", download=True, transform=T.ToTensor())

all_preds, all_labels = [], []
for i in range(len(test_ds)):
    img, label = test_ds[i]
    label = int(np.array(label).flatten()[0])
    img = img.unsqueeze(0)
    with torch.no_grad():
        cnn_feat, feat_maps = model.cnn(img)
        nf, ei = feature_map_to_graph(feat_maps[0], k=5)
        batch_index = torch.zeros(nf.size(0), dtype=torch.long)
        gnn_feat = model.gnn(nf, ei, batch_index)
        cnn_logits = model.cnn_head(cnn_feat)
        gnn_logits = model.gnn_head(gnn_feat)
        cnn_conf = F.softmax(cnn_logits, dim=-1)
        gnn_conf = F.softmax(gnn_logits, dim=-1)
        stats = torch.stack([cnn_conf.mean(-1), cnn_conf.std(-1),
                             gnn_conf.mean(-1), gnn_conf.std(-1)], dim=-1)
        alpha = model.fusion_gate(stats)
        final = alpha * cnn_logits + (1 - alpha) * gnn_logits
        pred = final.argmax(dim=1).item()
    all_preds.append(pred)
    all_labels.append(label)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
cm = confusion_matrix(all_labels, all_preds)
acc = accuracy_score(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(cm, cmap="Blues")

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Normal", "Pneumonia"])
ax.set_yticklabels(["Normal", "Pneumonia"])
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title(f"Confusion matrix (accuracy = {acc:.4f})")

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("fig_confusion.png", dpi=300)
plt.savefig("fig_confusion.pdf")
print(f"Accuracy = {acc:.4f}")
print(f"Confusion matrix:\n{cm}")
print("Saved fig_confusion.png and .pdf")