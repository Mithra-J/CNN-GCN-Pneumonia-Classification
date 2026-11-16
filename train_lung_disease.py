"""
Train CNN-GNN Fusion Model on Real Lung/Chest Data
====================================================
Run this on YOUR machine (needs internet access to download the dataset —
this sandbox's network couldn't reach the dataset host, so I couldn't
generate real numbers here).

Uses PneumoniaMNIST by default (binary: pneumonia vs. normal, chest X-ray,
28x28, auto-downloads via the medmnist package). Swap DATASET_CLASS for
ChestMNIST (14-class, multi-label) once this pipeline is validated, or for
your final clinical dataset once you and Paripurani pick one.

    pip install torch torchvision torch_geometric medmnist --break-system-packages
    python train_lung_disease.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from medmnist import PneumoniaMNIST, INFO
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

import random

SEED = 456
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

from model_classical import CNNGNNFusionModel

DATASET_CLASS = PneumoniaMNIST
INFO_KEY = "pneumoniamnist"
NUM_CLASSES = len(INFO[INFO_KEY]["label"])
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders():
    transform = T.Compose([T.ToTensor()])  # medmnist images are already 28x28 grayscale
    train_ds = DATASET_CLASS(split="train", download=True, transform=transform)
    val_ds = DATASET_CLASS(split="val", download=True, transform=transform)
    test_ds = DATASET_CLASS(split="test", download=True, transform=transform)
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False),
    )


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).squeeze().long()

        optimizer.zero_grad()
        logits, alpha, _ = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_alphas = [], [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.squeeze().long()
        logits, alpha, _ = model(images)
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
        all_alphas.append(alpha.squeeze().cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    alphas = torch.cat(all_alphas).numpy()

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted"),
        "alpha_mean": float(np.mean(alphas)),
        "alpha_std": float(np.std(alphas)),  # <-- THE KEY ABLATION NUMBER.
        # If this is near 0, the fusion gate collapsed to a constant weight
        # and isn't doing adaptive fusion — report this honestly either way.
    }
    return metrics


def main():
    print(f"Device: {DEVICE}")
    train_loader, val_loader, test_loader = get_dataloaders()

    model = CNNGNNFusionModel(num_classes=NUM_CLASSES, cnn_in_channels=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_metrics = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d} | train_loss={train_loss:.4f} | "
              f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['f1']:.4f} | "
              f"alpha_std={val_metrics['alpha_std']:.4f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), "best_model.pt")

    print("\n=== FINAL TEST SET RESULTS (report these in your paper, as-is) ===")
    model.load_state_dict(torch.load("best_model.pt"))
    test_metrics = evaluate(model, test_loader)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    if test_metrics["alpha_std"] < 0.02:
        print("\n*** WARNING: alpha_std is very low. The fusion gate may have ***")
        print("*** collapsed to a near-constant weight. This is a real finding")
        print("*** — report it honestly. It means the classical gate isn't ")
        print("*** learning meaningfully different weights per sample. ***")


if __name__ == "__main__":
    main()