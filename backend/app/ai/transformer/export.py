"""
Export trained TemporalKeypointTransformer to TorchScript for deployment.

Usage:
    python -m app.ai.transformer.export \
        --checkpoint ./models/transformer/best_model.pt \
        --norm_stats ./models/transformer/norm_stats.npz \
        --output ./models/transformer/waving_transformer.pt
"""

import argparse
import os

import numpy as np
import torch

from .model import TemporalKeypointTransformer


def export_to_torchscript(
    checkpoint_path: str,
    norm_stats_path: str,
    output_path: str,
    seq_len: int = 45,
):
    # Load normalization stats
    stats = np.load(norm_stats_path)
    mean = torch.tensor(stats["mean"])
    std = torch.tensor(stats["std"])

    # Load model
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt.get("config", {})
    model = TemporalKeypointTransformer(
        n_features=config.get("n_features", 12),
        d_model=config.get("d_model", 64),
        n_head=config.get("n_head", 2),
        n_layers=config.get("n_layers", 2),
        seq_len=seq_len,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    class NormalizedModel(torch.nn.Module):
        """Wrapper that applies normalization before the transformer."""

        def __init__(self, transformer, mean, std):
            super().__init__()
            self.transformer = transformer
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

        def forward(self, x):
            # x: [B, T, F]
            x = (x - self.mean) / self.std
            return self.transformer(x)

        def get_attention_weights(self, x):
            x = (x - self.mean) / self.std
            return self.transformer.get_attention_weights(x)

    wrapped = NormalizedModel(model, mean, std)

    # Script the model (handles dynamic shapes better than trace for transformers)
    scripted = torch.jit.script(wrapped)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scripted.save(output_path)

    # Verify
    loaded = torch.jit.load(output_path)
    example = torch.randn(1, seq_len, config.get("n_features", 12))
    with torch.no_grad():
        out1 = wrapped(example)
        out2 = loaded(example)
        max_diff = (out1 - out2).abs().max().item()

    print(f"Exported to: {output_path}")
    print(f"Model size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"Trace verification max diff: {max_diff:.2e}")
    print(f"Val metrics from checkpoint: f1={ckpt['val_metrics']['f1']:.4f}, "
          f"precision={ckpt['val_metrics']['precision']:.4f}, "
          f"recall={ckpt['val_metrics']['recall']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Export model to TorchScript")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--norm_stats", type=str, required=True)
    parser.add_argument("--output", type=str, default="./models/transformer/waving_transformer.pt")
    parser.add_argument("--seq_len", type=int, default=45)
    args = parser.parse_args()

    export_to_torchscript(args.checkpoint, args.norm_stats, args.output, args.seq_len)


if __name__ == "__main__":
    main()
