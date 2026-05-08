"""Train the MLP classifier on data/landmarks/.

End-to-end training loop with class-weighted cross-entropy, Adam,
ReduceLROnPlateau, and early stopping. Hyperparameters default to
feature-3 §3.5 values; CLI flags exist for experimentation.

After training, evaluates the best checkpoint on the held-out test
set, prints macro-F1 + per-class F1 + confusion matrix, and runs the
phase-2 -> phase-3 acceptance gate (feature-3 §3.6). Exits non-zero
if the gate fails — the run does not proceed to ONNX export
(commit 5) until both bars are met.

Run as a module so the package import path resolves cleanly::

    python -m asl_live.train.train_mlp
"""
from __future__ import annotations

import argparse
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from asl_live.train.augment import RandomAugment
from asl_live.train.dataset import (
    LandmarkDataset,
    compute_class_weights,
    make_splits,
)
from asl_live.train.export import export_run
from asl_live.train.metrics import (
    MACRO_F1_THRESHOLD,
    OFF_DIAGONAL_THRESHOLD,
    check_acceptance_bar,
    confusion_matrix,
    format_confusion_matrix,
    macro_f1,
    per_class_f1,
)
from asl_live.train.model import MLP


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ASL-live MLP classifier")
    # Hyperparameters (feature-3 §3.5 defaults)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto picks cuda if available, else cpu",
    )
    # Training schedule
    p.add_argument("--patience-stop", type=int, default=10, help="epochs without val improvement before halting")
    p.add_argument("--patience-lr", type=int, default=5, help="epochs without improvement before halving LR")
    p.add_argument("--lr-factor", type=float, default=0.5)
    # Augmentation knobs (feature-3 §3.3)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--scale-lo", type=float, default=0.95)
    p.add_argument("--scale-hi", type=float, default=1.05)
    p.add_argument("--trans-range", type=float, default=0.02)
    p.add_argument("--aug-prob", type=float, default=0.5)
    # I/O
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (default 0; >0 needs multiprocessing-safe Dataset on Windows)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested via --device cuda but is not available")
    return torch.device(spec)


def make_train_collate(augment: RandomAugment):
    """Default-collate a list of (vec, label) and apply augmentation to the batch."""

    def collate(batch):
        xs = torch.stack([item[0] for item in batch])
        ys = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return augment(xs), ys

    return collate


def make_eval_collate():
    """Plain collate: no augmentation, val/test see clean data."""

    def collate(batch):
        xs = torch.stack([item[0] for item in batch])
        ys = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return xs, ys

    return collate


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """One full pass over the training loader. Returns sample-weighted mean loss."""
    model.train()
    total_loss = 0.0
    total_n = 0
    for xs, ys in loader:
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(xs)
        loss = criterion(logits, ys)
        loss.backward()
        optimizer.step()

        bs = xs.size(0)
        total_loss += loss.item() * bs
        total_n += bs
    return total_loss / total_n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Eval loop used per-epoch on the val split. Returns (mean_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    total_n = 0
    correct = 0
    for xs, ys in loader:
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        logits = model(xs)
        loss = criterion(logits, ys)

        bs = xs.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        correct += (logits.argmax(dim=1) == ys).sum().item()
    return total_loss / total_n, correct / total_n


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model and return (preds, targets) as 1-D int arrays.

    Used post-training on the test set to feed the metrics module.
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for xs, ys in loader:
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        logits = model(xs)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_targets.append(ys.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_dataloaders(
    dataset: LandmarkDataset,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    augment: RandomAugment,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_train_collate(augment),
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_eval_collate(),
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_eval_collate(),
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    device: torch.device,
    *,
    epochs: int,
    patience_stop: int,
) -> dict:
    """Run the full training schedule with early stopping. Returns the best state-dict."""
    best_val_loss = float("inf")
    best_state: dict | None = None
    epochs_since_improvement = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(
            f"epoch {epoch:3d}/{epochs}  "
            f"train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
            f"val_acc {val_acc:.4f}  lr {lr:.2e}  ({elapsed:.1f}s)",
            flush=True,
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience_stop:
                print(f"Early stopping at epoch {epoch}: val_loss did not improve for {patience_stop} epochs")
                break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint (loss never improved)")
    print(f"Best val_loss: {best_val_loss:.4f}")
    return best_state


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    print("Loading dataset...")
    dataset = LandmarkDataset()
    counts = dataset.class_counts()
    print(
        f"  {len(dataset)} samples across {dataset.num_classes} classes  "
        f"(per-class min {int(counts.min())}, max {int(counts.max())}, "
        f"median {int(np.median(counts))})"
    )

    train_idx, val_idx, test_idx = make_splits(dataset, seed=args.seed)
    print(f"  splits: train {len(train_idx)}  val {len(val_idx)}  test {len(test_idx)}")

    augment = RandomAugment(
        noise_std=args.noise_std,
        scale_range=(args.scale_lo, args.scale_hi),
        trans_range=args.trans_range,
        p=args.aug_prob,
    )
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset,
        train_idx,
        val_idx,
        test_idx,
        augment,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = MLP().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: MLP, {n_params:,} parameters")

    # Class weights are calibrated on the train split (not the full dataset)
    # so the loss reflects what the model actually sees during training.
    train_counts = np.bincount(
        dataset.labels[np.array(train_idx)],
        minlength=dataset.num_classes,
    )
    class_weights = compute_class_weights(train_counts).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.patience_lr,
    )

    print(f"\nTraining: epochs={args.epochs}  batch_size={args.batch_size}  "
          f"lr={args.lr}  weight_decay={args.weight_decay}  seed={args.seed}\n")

    best_state = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler,
        device,
        epochs=args.epochs,
        patience_stop=args.patience_stop,
    )

    # Restore best checkpoint, then full test-set evaluation + acceptance gate.
    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest loss: {test_loss:.4f}  Test acc: {test_acc:.4f}")

    print("\n=== Test-set classification report ===")
    preds, targets = collect_predictions(model, test_loader, device)
    class_names = tuple(dataset.label_map[i] for i in range(dataset.num_classes))
    cm = confusion_matrix(targets, preds, n_classes=dataset.num_classes)

    f1_by_class = per_class_f1(cm, class_names)
    print("\nPer-class F1:")
    for name in class_names:
        print(f"  {name:>8}  {f1_by_class[name]:.4f}")

    score = macro_f1(cm)
    print(f"\nMacro-F1: {score:.4f}")

    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print(format_confusion_matrix(cm, class_names))

    passed, failures = check_acceptance_bar(cm, class_names)
    print()
    if not passed:
        print("ACCEPTANCE BAR FAILED:")
        for f in failures:
            print(f"  - {f}")
        print("\nDid not export. Iterate per feature-3 §3.6 (collect more data, swap a confusing gesture, etc.)")
        sys.exit(1)

    print(f"[PASS] Macro-F1 {score:.4f} >= {MACRO_F1_THRESHOLD:.2f}")
    print(f"[PASS] All off-diagonal cells <= {OFF_DIAGONAL_THRESHOLD * 100:.0f}% of row total")

    # Build the per-run report payload and export ONNX + sidecars.
    hyperparams_dict = {
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "patience_stop": args.patience_stop,
        "patience_lr": args.patience_lr,
        "lr_factor": args.lr_factor,
        "noise_std": args.noise_std,
        "scale_range": [args.scale_lo, args.scale_hi],
        "trans_range": args.trans_range,
        "aug_prob": args.aug_prob,
    }
    dataset_stats = {
        "total": int(len(dataset)),
        "num_classes": int(dataset.num_classes),
        "per_class": {name: int(counts[i]) for i, name in enumerate(class_names)},
        "split_sizes": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },
    }
    metrics_dict = {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "macro_f1": float(score),
        "per_class_f1": f1_by_class,
        "confusion_matrix": cm.tolist(),
    }
    run_dir = export_run(
        model,
        label_map=dataset.label_map,
        hyperparams=hyperparams_dict,
        dataset_stats=dataset_stats,
        metrics=metrics_dict,
    )
    print(f"\nWrote {run_dir}/{{mlp.onnx, label_map.json, training_report.json}}")
    print("Updated models/mlp.onnx + models/label_map.json -> latest")


if __name__ == "__main__":
    main()
