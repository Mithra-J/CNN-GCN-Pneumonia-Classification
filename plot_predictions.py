import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from medmnist import PneumoniaMNIST
import torchvision.transforms as T

from model_classical import CNNGNNFusionModel, feature_map_to_graph

DEVICE = torch.device("cpu")
model = CNNGNNFusionModel(num_classes=2, cnn_in_channels=1).to(DEVICE)
model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
model.eval()

test_ds = PneumoniaMNIST(split="test", download=True, transform=T.ToTensor())


def predict(img):
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
    probs = F.softmax(final, dim=-1)
    pred = probs.argmax(dim=1).item()
    conf = probs[0, pred].item()
    return pred, conf, alpha.item()


# Collect up to 4 correct and up to 4 incorrect, whichever exists
correct, wrong = [], []
for i in range(len(test_ds)):
    img, label = test_ds[i]
    label = int(np.array(label).flatten()[0])
    img = img.unsqueeze(0)
    pred, conf, alpha = predict(img)
    entry = (img, label, pred, conf, alpha)
    if pred == label and len(correct) < 4:
        correct.append(entry)
    elif pred != label and len(wrong) < 4:
        wrong.append(entry)
    if len(correct) == 4 and len(wrong) == 4:
        break

samples = correct + wrong
print(f"Collected {len(correct)} correct + {len(wrong)} incorrect = {len(samples)} samples")

# Pad to a multiple of 4 if needed
n_cols = 4
n_rows = 2
n_panels = n_rows * n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 7))

for idx in range(n_panels):
    r, c = divmod(idx, n_cols)
    ax = axes[r, c]
    if idx < len(samples):
        img, label, pred, conf, alpha = samples[idx]
        img_np = img[0, 0].detach().numpy()
        ax.imshow(img_np, cmap="gray")
        correct_flag = pred == label
        ax.set_title(
            f"True: {label} | Pred: {pred}\nconf={conf:.2f} | α={alpha:.3f}",
            fontsize=9, color=("green" if correct_flag else "red")
        )
    ax.axis("off")

plt.tight_layout()
plt.savefig("fig_predictions.png", dpi=300)
plt.savefig("fig_predictions.pdf")
print("Saved fig_predictions.png and .pdf")