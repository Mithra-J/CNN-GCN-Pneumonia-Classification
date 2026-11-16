import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from medmnist import PneumoniaMNIST
import torchvision.transforms as T

from model_classical import CNNGNNFusionModel, grad_cam, feature_map_to_graph

DEVICE = torch.device("cpu")
model = CNNGNNFusionModel(num_classes=2, cnn_in_channels=1).to(DEVICE)
model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
model.eval()

test_ds = PneumoniaMNIST(split="test", download=True, transform=T.ToTensor())


def forward_and_cam(model, img, target_class=None):
    """Full forward pass with a single backward, returning cam and prediction."""
    cnn_feat, feat_maps = model.cnn(img)
    feat_maps.retain_grad()

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

    # backward on the predicted class score through the CNN branch only,
    # so feat_maps.grad is populated for Grad-CAM
    if target_class is None:
        target_class = pred
    score = cnn_logits[0, target_class]
    model.zero_grad()
    score.backward()

    cam = grad_cam(model, img, target_class, feat_maps)[0].detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, pred


# collect 2 correct + 2 incorrect samples
correct_samples = []
wrong_samples = []
for i in range(len(test_ds)):
    img, label = test_ds[i]
    label = int(np.array(label).flatten()[0])
    img = img.unsqueeze(0)
    cam, pred = forward_and_cam(model, img)
    entry = (img, label, pred, cam)
    if pred == label and len(correct_samples) < 2:
        correct_samples.append(entry)
    elif pred != label and len(wrong_samples) < 2:
        wrong_samples.append(entry)
    if len(correct_samples) >= 2 and len(wrong_samples) >= 2:
        break

samples = correct_samples + wrong_samples

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for idx, (img, label, pred, cam) in enumerate(samples):
    img_np = img[0, 0].detach().numpy()
    cam_t = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
    cam_up = F.interpolate(cam_t, size=(28, 28), mode="bilinear",
                           align_corners=False)[0, 0].numpy()

    axes[0, idx].imshow(img_np, cmap="gray")
    axes[0, idx].set_title(f"True: {label}, Pred: {pred}", fontsize=9)
    axes[0, idx].axis("off")

    axes[1, idx].imshow(img_np, cmap="gray")
    axes[1, idx].imshow(cam_up, cmap="jet", alpha=0.5)
    axes[1, idx].set_title("Grad-CAM" + (" ✓" if pred == label else " ✗"), fontsize=9)
    axes[1, idx].axis("off")

plt.tight_layout()
plt.savefig("fig_gradcam.png", dpi=300)
plt.savefig("fig_gradcam.pdf")
print("Saved fig_gradcam.png and .pdf")