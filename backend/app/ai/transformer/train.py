"""
Training script for TemporalKeypointTransformer.

Usage:
    python -m app.ai.transformer.train \
        --n_samples 20000 \
        --epochs 100 \
        --batch_size 256 \
        --output_dir ./models/transformer
"""

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data_pipeline import (
    DEFAULT_SEQ_LEN,
    N_FEATURES,
    SyntheticDataset,
    load_synthetic_dataset,
)
from .model import TemporalKeypointTransformer


class TemporalAugmentation:
    """Data augmentation for temporal keypoint sequences."""

    def __init__(
        self,
        temporal_crop_range: Tuple[float, float] = (0.8, 1.0),
        noise_std: float = 0.02,
        side_dropout_prob: float = 0.1,
        feature_dropout_prob: float = 0.05,
    ):
        self.temporal_crop_range = temporal_crop_range
        self.noise_std = noise_std
        self.side_dropout_prob = side_dropout_prob
        self.feature_dropout_prob = feature_dropout_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, F] single sequence
        Returns:
            augmented [T, F]
        """
        # Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            noise[:, -1] = 0  # don't noise validity flag
            x = x + noise

        # Side dropout (zero out one side's wrist features)
        if torch.rand(1).item() < self.side_dropout_prob:
            if torch.rand(1).item() < 0.5:
                x[:, 0:2] = 0  # left wrist
            else:
                x[:, 2:4] = 0  # right wrist

        # Random feature dropout
        mask = torch.rand(x.shape[1], device=x.device) > self.feature_dropout_prob
        mask[-1] = True  # never drop validity
        x = x * mask.float()

        # Temporal crop + resize (simulates variable-length sequences)
        if self.temporal_crop_range[0] < 1.0:
            T = x.shape[0]
            crop_ratio = torch.empty(1).uniform_(*self.temporal_crop_range).item()
            crop_len = max(int(T * crop_ratio), T // 2)
            start = torch.randint(0, T - crop_len + 1, (1,)).item()
            indices = torch.linspace(start, start + crop_len - 1, T).long().clamp(0, T - 1)
            x = x[indices]

        return x


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute classification metrics."""
    y_pred = np.asarray(y_pred).reshape(-1)  # ensure [N] shape
    y_true = np.asarray(y_true).reshape(-1)
    y_pred_binary = (y_pred >= threshold).astype(np.int64)

    tp = int((y_true & y_pred_binary).sum())
    fp = int(((1 - y_true) & y_pred_binary).sum())
    fn = int((y_true & (1 - y_pred_binary)).sum())
    tn = int(((1 - y_true) & (1 - y_pred_binary)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


class Trainer:
    def __init__(
        self,
        model: TemporalKeypointTransformer,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        label_smoothing: float = 0.0,
        pos_weight: float = 2.0,
        warmup_epochs: int = 5,
    ):
        self.model = model.to(device)
        self.device = device
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight
        self.warmup_epochs = warmup_epochs

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=10,
        )
        self.augmentation = TemporalAugmentation()
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.history: Dict[str, list] = defaultdict(list)

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """BCE loss with label smoothing and class weighting."""
        # Label smoothing
        smooth_target = target.float() * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Per-sample weight: positive samples weighted higher
        weight = torch.where(target > 0.5, self.pos_weight, 1.0)

        pred_flat = pred.view(-1)
        target_flat = smooth_target.view(-1)
        weight_flat = weight.view(-1)

        loss = F.binary_cross_entropy(pred_flat, target_flat, weight=weight_flat, reduction="mean")
        return loss

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds, all_targets = [], []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Apply augmentation
            x_aug = torch.stack([self.augmentation(x) for x in x_batch])

            self.optimizer.zero_grad()
            pred = self.model(x_aug)
            loss = self._loss(pred, y_batch.unsqueeze(1))
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            all_preds.append(pred.detach().cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        metrics = compute_metrics(all_targets, all_preds)
        metrics["loss"] = total_loss / len(loader.dataset)
        return metrics

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            pred = self.model(x_batch)
            loss = self._loss(pred, y_batch.unsqueeze(1))

            total_loss += loss.item() * x_batch.size(0)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Find best threshold via simple grid search
        best_f1 = 0.0
        best_metrics = None
        for thresh in np.arange(0.3, 0.8, 0.05):
            m = compute_metrics(all_targets, all_preds, threshold=thresh)
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_metrics = m

        best_metrics["loss"] = total_loss / len(loader.dataset)
        best_metrics["best_threshold"] = float(
            np.arange(0.3, 0.8, 0.05)[
                max(
                    range(len(np.arange(0.3, 0.8, 0.05))),
                    key=lambda i: compute_metrics(all_targets, all_preds,
                                                  threshold=np.arange(0.3, 0.8, 0.05)[i])["f1"],
                )
            ]
        )
        return best_metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_path: str,
        patience: int = 20,
    ):
        patience_counter = 0

        for epoch in range(epochs):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            self.scheduler.step(val_metrics["f1"])
            elapsed = time.time() - t0

            lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_f1"].append(train_metrics["f1"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["lr"].append(lr)

            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"LR {lr:.2e} | "
                f"train_loss {train_metrics['loss']:.4f} | "
                f"train_f1 {train_metrics['f1']:.3f} | "
                f"val_loss {val_metrics['loss']:.4f} | "
                f"val_f1 {val_metrics['f1']:.3f} | "
                f"val_prec {val_metrics['precision']:.3f} | "
                f"val_rec {val_metrics['recall']:.3f} | "
                f"{elapsed:.1f}s"
            )

            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                self.best_epoch = epoch + 1
                patience_counter = 0

                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                        "config": {
                            "d_model": self.model.d_model,
                            "n_features": self.model.n_features,
                            "n_head": self.model.encoder.layers[0].self_attn.num_heads,
                            "n_layers": len(self.model.encoder.layers),
                            "seq_len": self.model.seq_len,
                        },
                    },
                    save_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        print(f"\nBest model: epoch {self.best_epoch}, val_f1={self.best_val_f1:.4f}")

    def load_best(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from {path} (epoch {ckpt['epoch']}, "
              f"val_f1={ckpt['val_metrics']['f1']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Train TemporalKeypointTransformer")
    parser.add_argument("--n_samples", type=int, default=20000,
                        help="Number of synthetic samples")
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN,
                        help="Sequence length (frames)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=128, help="Transformer hidden dim")
    parser.add_argument("--n_layers", type=int, default=4, help="Transformer layers")
    parser.add_argument("--n_head", type=int, default=4, help="Attention heads")
    parser.add_argument("--pos_weight", type=float, default=2.0,
                        help="Positive class weight for BCE")
    parser.add_argument("--output_dir", type=str, default="./models/transformer",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load/generate data
    print(f"Generating {args.n_samples} synthetic samples...")
    (X_train, y_train), (X_val, y_val) = load_synthetic_dataset(
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        save_dir=args.output_dir,
    )
    print(f"Train: {X_train.shape}, Pos={y_train.sum()}, Neg={(1-y_train).sum()}")
    print(f"Val:   {X_val.shape}, Pos={y_val.sum()}, Neg={(1-y_val).sum()}")

    # Compute normalization stats from training set
    mean, std = SyntheticDataset.compute_normalization_stats(X_train)

    # Normalize
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    # Save normalization stats
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(os.path.join(args.output_dir, "norm_stats.npz"), mean=mean, std=std)

    # DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train_norm), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val_norm), torch.tensor(y_val, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Model
    model = TemporalKeypointTransformer(
        n_features=N_FEATURES,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layers=args.n_layers,
        seq_len=args.seq_len,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # Train
    trainer = Trainer(
        model, device,
        lr=args.lr,
        pos_weight=args.pos_weight,
    )

    save_path = os.path.join(args.output_dir, "best_model.pt")
    trainer.fit(train_loader, val_loader, args.epochs, save_path)

    # Save history
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in trainer.history.items()}, f, indent=2)

    print(f"Training complete. Best val F1: {trainer.best_val_f1:.4f}")


if __name__ == "__main__":
    main()
