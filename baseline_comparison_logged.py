"""
baseline_comparison_logged.py
Same experiment as baseline_comparison.py (models, training, evaluation, RNG order all unchanged),
but it now ALSO saves everything needed for the report figures:

  training_losses.csv      per-epoch train loss, every model, every seed   -> Fig 5.5
  alphas_seed<S>.npy       learned alpha per test image (adaptive model)   -> Fig 5.7
  alphas.npy               copy of the first seed's alphas (624 values)    -> Fig 5.7
  results_per_seed.csv     test metrics per model per seed                 -> check Table I / Fig 5.6
"""

import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from medmnist import PneumoniaMNIST, INFO
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import copy
import random

from model_classical import CNNGNNFusionModel, CNNBranch, GNNBranch, feature_map_to_graph
import torch.nn.functional as F


# --------------------------- Global settings ---------------------------
# >>> PUT THE 3 SEEDS YOU USED FOR TABLE I HERE (456 is the one visible in your script).
# With the same seeds on CPU, results should reproduce your earlier runs.
SEEDS = [456, 123, 789]

DATASET_CLASS = PneumoniaMNIST
INFO_KEY = "pneumoniamnist"
NUM_CLASSES = len(INFO[INFO_KEY]["label"])
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# names used in the CSV (match make_fig_5_5.py)
LOG_NAMES = {
    "CNN-only": "CNN-only",
    "GNN-only": "GCN-only",
    "Fixed 0.5/0.5": "Fixed 0.5/0.5",
    "Global scalar fusion": "Global scalar fusion",
    "Adaptive (MLP gate)": "Adaptive (MLP gate)",
}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# --------------------------- Data loaders ---------------------------
def get_dataloaders():
    transform = T.Compose([T.ToTensor()])
    train_ds = DATASET_CLASS(split="train", download=True, transform=transform)
    val_ds = DATASET_CLASS(split="val", download=True, transform=transform)
    test_ds = DATASET_CLASS(split="test", download=True, transform=transform)
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False),
    )


# --------------------------- Models (unchanged) ---------------------------
class CNNOnly(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = CNNBranch(in_channels=1, feat_dim=128)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        feat, _ = self.cnn(x)
        return self.head(feat), None, None


class GNNOnly(nn.Module):
    def __init__(self, num_classes, graph_k=5):
        super().__init__()
        self.cnn = CNNBranch(in_channels=1, feat_dim=128)  # feature extractor only
        self.gnn = GNNBranch(node_feat_dim=128, out_dim=128)
        self.head = nn.Linear(128, num_classes)
        self.graph_k = graph_k

    def forward(self, x):
        B = x.size(0)
        _, feat_maps = self.cnn(x)
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
        return self.head(gnn_feat), None, None


class FixedFusion(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = CNNBranch(in_channels=1, feat_dim=128)
        self.gnn = GNNBranch(node_feat_dim=128, out_dim=128)
        self.cnn_head = nn.Linear(128, num_classes)
        self.gnn_head = nn.Linear(128, num_classes)

    def forward(self, x):
        B = x.size(0)
        cnn_feat, feat_maps = self.cnn(x)
        cnn_logits = self.cnn_head(cnn_feat)
        node_feats_list, edge_index_list, batch_list = [], [], []
        offset = 0
        for i in range(B):
            nf, ei = feature_map_to_graph(feat_maps[i], k=5)
            node_feats_list.append(nf)
            edge_index_list.append(ei + offset)
            batch_list.append(torch.full((nf.size(0),), i, dtype=torch.long))
            offset += nf.size(0)
        node_features = torch.cat(node_feats_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1)
        batch_index = torch.cat(batch_list, dim=0)
        gnn_feat = self.gnn(node_features, edge_index, batch_index)
        gnn_logits = self.gnn_head(gnn_feat)
        return 0.5 * cnn_logits + 0.5 * gnn_logits, None, None


class GlobalScalarFusion(nn.Module):
    """Learned global alpha (single scalar for all samples)."""
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = CNNBranch(in_channels=1, feat_dim=128)
        self.gnn = GNNBranch(node_feat_dim=128, out_dim=128)
        self.cnn_head = nn.Linear(128, num_classes)
        self.gnn_head = nn.Linear(128, num_classes)
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        B = x.size(0)
        cnn_feat, feat_maps = self.cnn(x)
        cnn_logits = self.cnn_head(cnn_feat)

        node_feats_list, edge_index_list, batch_list = [], [], []
        offset = 0
        for i in range(B):
            nf, ei = feature_map_to_graph(feat_maps[i], k=5)
            node_feats_list.append(nf)
            edge_index_list.append(ei + offset)
            batch_list.append(torch.full((nf.size(0),), i, dtype=torch.long))
            offset += nf.size(0)
        node_features = torch.cat(node_feats_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1)
        batch_index = torch.cat(batch_list, dim=0)

        gnn_feat = self.gnn(node_features, edge_index, batch_index)
        gnn_logits = self.gnn_head(gnn_feat)

        alpha = torch.sigmoid(self.alpha_raw)
        final_logits = alpha * cnn_logits + (1 - alpha) * gnn_logits
        return final_logits, alpha, feat_maps


# --------------------------- Train / eval ---------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).squeeze().long()
        optimizer.zero_grad()
        logits, _, _ = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.squeeze().long()
        logits, _, _ = model(images)
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted"),
    }


@torch.no_grad()
def collect_alphas(model, loader):
    """Per-image alpha from the adaptive model's 2nd return value."""
    model.eval()
    out = []
    for images, _ in loader:
        _, alpha, _ = model(images.to(DEVICE))
        if alpha is None or not torch.is_tensor(alpha):
            raise RuntimeError("Adaptive model's 2nd return value is not a tensor of alpha values. "
                               "Send me model_classical.py (the forward of CNNGNNFusionModel).")
        out.append(alpha.detach().reshape(-1).cpu().numpy())
    a = np.concatenate(out)
    n_test = len(loader.dataset)
    if len(a) != n_test or a.min() < 0 or a.max() > 1:
        raise RuntimeError(f"Got {len(a)} values (expected {n_test}) with range [{a.min():.3f}, {a.max():.3f}]. "
                           "The 2nd return value is probably not per-image alpha. "
                           "Send me model_classical.py (the forward of CNNGNNFusionModel).")
    return a


def run_experiment(name, model, train_loader, val_loader, test_loader, seed, loss_rows, want_alpha=False):
    print(f"\n=== {name} (seed {seed}) ===")
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    best_val_f1 = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        loss_rows.append((LOG_NAMES.get(name, name), seed, epoch, float(train_loss)))   # <-- logged every epoch
        val_metrics = evaluate(model, val_loader)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
        if epoch % 5 == 0 or epoch == EPOCHS:
            print(f"  Epoch {epoch:2d} | train_loss={train_loss:.4f} | "
                  f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['f1']:.4f}")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader)
    print(f"  TEST: acc={test_metrics['accuracy']:.4f} | f1={test_metrics['f1']:.4f} | "
          f"prec={test_metrics['precision']:.4f} | rec={test_metrics['recall']:.4f}")

    alphas = collect_alphas(model, test_loader) if want_alpha else None
    return test_metrics, alphas


def main():
    print(f"Device: {DEVICE}")
    loss_rows = []          # (model, seed, epoch, loss)
    result_rows = []        # (model, seed, acc, f1, prec, rec)

    for si, seed in enumerate(SEEDS):
        print(f"\n################  SEED {seed}  ################")
        set_seed(seed)                                   # same RNG order as the original script
        train_loader, val_loader, test_loader = get_dataloaders()
        results = {}

        configs = [
            ("CNN-only",             "CNN-only",             lambda: CNNOnly(NUM_CLASSES)),
            ("GNN-only",             "GNN-only",             lambda: GNNOnly(NUM_CLASSES)),
            ("Fixed 0.5/0.5",        "Fixed 0.5/0.5 Fusion", lambda: FixedFusion(NUM_CLASSES)),
            ("Global scalar fusion", "Global Scalar Fusion", lambda: GlobalScalarFusion(NUM_CLASSES)),
            ("Adaptive (MLP gate)",  "Adaptive (MLP gate)",  lambda: CNNGNNFusionModel(num_classes=NUM_CLASSES, cnn_in_channels=1)),
        ]
        for key, title, make in configs:
            is_adaptive = key == "Adaptive (MLP gate)"
            m, alphas = run_experiment(key, make(), train_loader, val_loader, test_loader,
                                       seed, loss_rows, want_alpha=is_adaptive)
            results[key] = m
            result_rows.append((key, seed, m["accuracy"], m["f1"], m["precision"], m["recall"]))
            if alphas is not None:
                np.save(f"alphas_seed{seed}.npy", alphas)
                if si == 0:
                    np.save("alphas.npy", alphas)
                print(f"  saved alphas_seed{seed}.npy  (n={len(alphas)}, mean={alphas.mean():.4f}, "
                      f"std={alphas.std():.4f}, min={alphas.min():.4f}, max={alphas.max():.4f})")

        # save after every seed so nothing is lost if you stop early
        with open("training_losses.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["model", "seed", "epoch", "loss"]); w.writerows(loss_rows)
        with open("results_per_seed.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["model", "seed", "accuracy", "f1", "precision", "recall"]); w.writerows(result_rows)

    print("\n=== SUMMARY (mean ± std over seeds, sample std) ===")
    print(f"{'Model':<25} {'Acc':>16} {'F1':>16} {'Prec':>16} {'Rec':>16}")
    for key, _, _ in configs:
        rows = np.array([r[2:] for r in result_rows if r[0] == key])
        mu, sd = rows.mean(0), rows.std(0, ddof=1) if len(rows) > 1 else np.zeros(4)
        print(f"{key:<25} " + " ".join(f"{mu[i]:.4f}±{sd[i]:.4f}" for i in range(4)))
    print("\nSaved: training_losses.csv, results_per_seed.csv, alphas.npy, alphas_seed<S>.npy")


if __name__ == "__main__":
    main()