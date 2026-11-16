"""
CNN-GNN Fusion for Lung Disease Classification
================================================
Approved novelty (per faculty guide sign-off):
  1. Adaptive graph construction (kNN in feature space, not fixed grid)
  2. Dynamic confidence-weighted CNN-GNN fusion (classical MLP gate,
     replacing a fixed ensemble weight)
  3. Grad-CAM explainability

No quantum, no unverified claims. This is a real, runnable pipeline —
verified with a smoke test below — but the RESULTS section of your
paper needs REAL numbers from training on an actual dataset. Do not
fabricate accuracy/F1 figures; train this for real, then report what
you get, including if it underperforms your baseline. A modest honest
result is publishable; a fabricated strong one is misconduct if caught.

Install:
    pip install torch torchvision torch_geometric medmnist --break-system-packages
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


# ---------------------------------------------------------------------------
# 1. CNN branch (swap in ResNet18/EfficientNet once you have real data —
#    torchvision.models.resnet18(weights="IMAGENET1K_V1") is the easy path)
# ---------------------------------------------------------------------------
class CNNBranch(nn.Module):
    def __init__(self, in_channels=1, feat_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        )
        # Keep the last conv output around (unpooled) for Grad-CAM hooks
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, feat_dim)
        self._last_conv_output = None  # populated in forward, used by Grad-CAM

    def forward(self, x):
        feat_map = self.conv(x)              # [B, 128, H', W'] — Grad-CAM target
        self._last_conv_output = feat_map
        if feat_map.requires_grad:
            feat_map.retain_grad()            # only during training / Grad-CAM runs
        pooled = self.pool(feat_map).flatten(1)
        return self.fc(pooled), feat_map


# ---------------------------------------------------------------------------
# 2. Adaptive graph construction (dependency-free, dense kNN)
# ---------------------------------------------------------------------------
def build_adaptive_graph(node_features, k=5):
    """kNN graph in feature space — adaptive vs. a fixed grid graph."""
    n = node_features.size(0)
    k = min(k, n - 1)
    dist = torch.cdist(node_features, node_features)
    dist.fill_diagonal_(float("inf"))
    knn_idx = dist.topk(k, largest=False).indices
    src = torch.arange(n).unsqueeze(1).repeat(1, k).flatten()
    dst = knn_idx.flatten()
    return torch.stack([src, dst], dim=0)


def feature_map_to_graph(feat_map, k=5):
    """
    Converts a CNN feature map [C, H, W] into graph nodes: each spatial
    location becomes a node with a C-dim feature vector. This is what
    'adaptive graph construction over CNN feature patches' means concretely.
    """
    c, h, w = feat_map.shape
    node_features = feat_map.permute(1, 2, 0).reshape(h * w, c)  # [H*W, C]
    edge_index = build_adaptive_graph(node_features, k=k)
    return node_features, edge_index


# ---------------------------------------------------------------------------
# 3. GNN branch
# ---------------------------------------------------------------------------
class GNNBranch(nn.Module):
    def __init__(self, node_feat_dim=128, hidden_dim=64, out_dim=128):
        super().__init__()
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)


# ---------------------------------------------------------------------------
# 4. Classical confidence-weighted fusion gate (MLP — this is your
#    guide-approved novelty; no quantum)
# ---------------------------------------------------------------------------
class ConfidenceFusionGate(nn.Module):
    """Learns a per-sample fusion weight alpha from both branches'
    prediction confidence, replacing a fixed 50/50 (or any fixed) weight."""

    def __init__(self, hidden_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, stats):
        return torch.sigmoid(self.mlp(stats))  # [B, 1] in (0, 1)


# ---------------------------------------------------------------------------
# 5. Full model
# ---------------------------------------------------------------------------
class CNNGNNFusionModel(nn.Module):
    def __init__(self, num_classes, cnn_in_channels=1, graph_k=5):
        super().__init__()
        self.cnn = CNNBranch(in_channels=cnn_in_channels, feat_dim=128)
        self.gnn = GNNBranch(node_feat_dim=128, out_dim=128)
        self.cnn_head = nn.Linear(128, num_classes)
        self.gnn_head = nn.Linear(128, num_classes)
        self.fusion_gate = ConfidenceFusionGate()
        self.graph_k = graph_k

    def forward(self, images):
        B = images.size(0)
        cnn_feat, feat_maps = self.cnn(images)      # [B,128], [B,128,H,W]
        cnn_logits = self.cnn_head(cnn_feat)

        # Build one graph per sample from its own feature map, adaptively
        node_feats_list, edge_index_list, batch_list = [], [], []
        offset = 0
        for i in range(B):
            nf, ei = feature_map_to_graph(feat_maps[i], k=self.graph_k)
            node_feats_list.append(nf)
            edge_index_list.append(ei + offset)
            batch_list.append(torch.full((nf.size(0),), i, dtype=torch.long))
            offset += nf.size(0)

        node_features = torch.cat(node_feats_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1)
        batch_index = torch.cat(batch_list, dim=0)

        gnn_feat = self.gnn(node_features, edge_index, batch_index)
        gnn_logits = self.gnn_head(gnn_feat)

        cnn_conf = F.softmax(cnn_logits, dim=-1)
        gnn_conf = F.softmax(gnn_logits, dim=-1)
        stats = torch.stack([
            cnn_conf.mean(-1), cnn_conf.std(-1),
            gnn_conf.mean(-1), gnn_conf.std(-1),
        ], dim=-1)

        alpha = self.fusion_gate(stats)  # [B, 1]
        final_logits = alpha * cnn_logits + (1 - alpha) * gnn_logits
        return final_logits, alpha, feat_maps  # feat_maps needed for Grad-CAM


# ---------------------------------------------------------------------------
# 6. Grad-CAM (for the CNN branch)
# ---------------------------------------------------------------------------
def grad_cam(model, images, target_class, feat_maps):
    """
    Standard Grad-CAM: weight each feature-map channel by the gradient of
    the target class score w.r.t. that channel, then combine.
    Call AFTER a forward + backward pass so gradients exist.
    """
    grads = feat_maps.grad  # [B, C, H, W] — populated by retain_grad() + backward()
    weights = grads.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
    cam = F.relu((weights * feat_maps).sum(dim=1))  # [B, H, W]
    cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return cam  # normalized 0-1 heatmap, upsample to image size for overlay


if __name__ == "__main__":
    print("=== SMOKE TEST on synthetic random data (NOT real results) ===")
    B, C, H, W, NUM_CLASSES = 4, 1, 64, 64, 4
    model = CNNGNNFusionModel(num_classes=NUM_CLASSES, cnn_in_channels=C)

    images = torch.randn(B, C, H, W, requires_grad=True)
    logits, alpha, feat_maps = model(images)
    print("Logits shape:", logits.shape)
    print("Fusion weights (alpha):", alpha.squeeze().tolist())

    # Backward pass for Grad-CAM
    target_class = logits.argmax(dim=1)
    loss = logits[range(B), target_class].sum()
    loss.backward()
    cams = grad_cam(model, images, target_class, feat_maps)
    print("Grad-CAM heatmap shape:", cams.shape)
    print("\nSmoke test passed — pipeline (CNN branch, adaptive graph, GNN branch,")
    print("confidence fusion, Grad-CAM) all run end to end.")
    print("\n>>> NEXT: train_lung_disease.py loads a REAL dataset. Run that for")
    print(">>> actual results — do not report the numbers above, they are random.")