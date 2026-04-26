#!/usr/bin/env python3
"""
STr-GCN 手势分类器训练脚本

训练流程:
    1. 加载预标注或规则引擎伪标注的骨架序列数据
    2. 划分训练/验证集
    3. 训练 STr-GCN 模型
    4. 保存最佳模型权重

数据集格式 (JSON Lines):
    每行一个样本:
    {
        "track_id": "front_p1",
        "frames": [
            {"body": [[x,y,c], ...], "left_hand": [[x,y,z], ...], "right_hand": [...]},
            ...
        ],
        "label": "hailing"   // none | hand_up | greeting | hailing
    }

用法:
    python train/train_stgcn_gesture.py \
        --data ./data/gesture_dataset.jsonl \
        --epochs 100 \
        --batch-size 32 \
        --lr 0.001 \
        --output ./models/stgcn_gesture.pt
"""

import argparse
import json
import os
import random
import sys
from typing import List, Dict, Tuple

import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ai.gesture_stgcn import (
    SkeletonGraph,
    PoseNormalizer,
    _STrGCNModel,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_dataset(path: str) -> List[Dict]:
    """加载 JSON Lines 格式的数据集。"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


class GestureDataset:
    """手势识别数据集。"""

    LABEL_MAP = {"none": 0, "hand_up": 1, "greeting": 2, "hailing": 3}
    INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

    def __init__(
        self,
        samples: List[Dict],
        seq_len: int = 32,
        augment: bool = False,
    ) -> None:
        self.samples = samples
        self.seq_len = seq_len
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        sample = self.samples[idx]
        frames = sample["frames"]
        label_str = sample.get("label", "none")
        label = self.LABEL_MAP.get(label_str, 0)

        # 提取关键点序列
        seq = []
        for frame in frames:
            body = np.array(frame.get("body", []), dtype=np.float32)
            left = frame.get("left_hand")
            right = frame.get("right_hand")
            feat = PoseNormalizer.normalize_frame(body, left, right)
            seq.append(feat)

        seq = np.array(seq, dtype=np.float32)  # (T, V, 2)

        # 数据增强
        if self.augment:
            seq = self._augment(seq)

        # 采样到固定长度
        seq = self._resample(seq, self.seq_len)

        return seq, label

    def _resample(self, seq: np.ndarray, target_len: int) -> np.ndarray:
        T = len(seq)
        if T == target_len:
            return seq
        if T < target_len:
            # 重复最后一帧填充
            pad = np.repeat(seq[-1:], target_len - T, axis=0)
            return np.concatenate([seq, pad], axis=0)
        # 均匀采样
        indices = np.linspace(0, T - 1, target_len, dtype=int)
        return seq[indices]

    def _augment(self, seq: np.ndarray) -> np.ndarray:
        """随机数据增强: 缩放、旋转、噪声。"""
        # 随机缩放 (0.9 ~ 1.1)
        scale = np.random.uniform(0.9, 1.1)
        seq = seq * scale

        # 随机水平翻转 (50%)
        if np.random.rand() > 0.5:
            seq[:, :, 0] = -seq[:, :, 0]

        # 随机平移
        tx = np.random.uniform(-0.1, 0.1)
        ty = np.random.uniform(-0.1, 0.1)
        seq = seq + np.array([tx, ty])

        # 高斯噪声
        noise = np.random.normal(0, 0.01, seq.shape)
        seq = seq + noise

        return seq


def collate_fn(batch: List[Tuple[np.ndarray, int]]):
    """自定义 collate 函数。"""
    import torch

    seqs = []
    labels = []
    for seq, label in batch:
        seqs.append(seq)
        labels.append(label)
    seqs = torch.from_numpy(np.array(seqs, dtype=np.float32))
    labels = torch.from_numpy(np.array(labels, dtype=np.int64))
    return seqs, labels


def train_epoch(
    model: _STrGCNModel,
    loader,
    optimizer,
    criterion,
    device,
) -> Tuple[float, float]:
    import torch

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = logits.max(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total


def eval_epoch(
    model: _STrGCNModel,
    loader,
    criterion,
    device,
) -> Tuple[float, float]:
    import torch

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            _, pred = logits.max(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return total_loss / len(loader), correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 STr-GCN 手势分类器")
    parser.add_argument("--data", type=str, required=True, help="训练数据路径 (JSON Lines)")
    parser.add_argument("--val-split", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seq-len", type=int, default=32, help="时序长度")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="./models/stgcn_gesture.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("错误: 需要安装 PyTorch 才能训练。请执行 pip install torch torchvision")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    print(f"加载数据集: {args.data}")
    all_samples = load_dataset(args.data)
    print(f"总样本数: {len(all_samples)}")

    # 统计类别分布
    label_counts = {}
    for s in all_samples:
        lbl = s.get("label", "none")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    print("类别分布:", label_counts)

    # 划分训练/验证集
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * (1 - args.val_split))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    print(f"训练集: {len(train_samples)}, 验证集: {len(val_samples)}")

    train_dataset = GestureDataset(train_samples, seq_len=args.seq_len, augment=True)
    val_dataset = GestureDataset(val_samples, seq_len=args.seq_len, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # 初始化模型
    model = _STrGCNModel(
        num_classes=4,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.input_embed.parameters(), "lr": args.lr * 0.5},
            {"params": model.gcn_layers.parameters(), "lr": args.lr},
            {"params": model.transformer.parameters(), "lr": args.lr},
            {"params": model.classifier.parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                args.output,
            )
            print(f"  -> 保存最佳模型 (val_acc={val_acc:.4f}) 到 {args.output}")

    print(f"训练完成。最佳验证准确率: {best_val_acc:.4f}")
    print(f"模型已保存: {args.output}")


if __name__ == "__main__":
    main()
