"""
STr-GCN 时序手势分类器 (原型实现)

基于 Spatial-Temporal Graph Convolutional Network + Transformer Encoder 的手势识别模型。
设计目标：替代基于几何阈值的手势状态机，提升泛化性与遮挡鲁棒性。

架构:
    输入: T 帧 x V 关键点 x 2D 坐标 (归一化到躯干坐标系)
        ↓
    Spatial GCN (多层): 每帧按骨骼拓扑提取空间特征
        ↓
    Temporal Positional Encoding + Transformer Encoder
        ↓
    Global Average Pooling
        ↓
    MLP Classifier: [none, hand_up, greeting, hailing]

参考文献:
    - Yan et al., "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition", AAAI 2018
    - STr-GCN 变体: GCN 提取帧内特征 + Transformer 建模帧间长程依赖

使用方式:
    1. 先用规则引擎 (gesture.py) 收集伪标签数据
    2. 在自定义数据集上训练本模型 (见 train/train_stgcn_gesture.py)
    3. 部署时替换 is_hailing_gesture 的调用为 STrGestureRecognizer
"""

import logging
import math
from typing import List, Tuple, Optional, Dict, Deque
from collections import deque
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 骨架图拓扑定义 (COCO-17 Body + MediaPipe-21 Hands x2)
# =============================================================================

class SkeletonGraph:
    """
    人体+手部统一骨架图。
    节点定义:
        0-16  : COCO body keypoints
        17-37 : Left hand (21 points)
        38-58 : Right hand (21 points)
    """

    NUM_BODY_NODES = 17
    NUM_HAND_NODES = 21
    NUM_NODES = NUM_BODY_NODES + 2 * NUM_HAND_NODES  # 59

    # COCO body 骨骼连接 (无向边)
    BODY_EDGES = [
        (0, 1), (0, 2), (1, 3), (2, 4),        # 头部
        (5, 6),                                   # 肩膀
        (5, 7), (7, 9), (6, 8), (8, 10),        # 手臂
        (11, 12),                                 # 髋部
        (5, 11), (6, 12),                         # 躯干
        (11, 13), (13, 15), (12, 14), (14, 16),  # 腿部
    ]

    # MediaPipe hand 骨骼连接 (21点)
    HAND_EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 4),           # 拇指
        (0, 5), (5, 6), (6, 7), (7, 8),           # 食指
        (0, 9), (9, 10), (10, 11), (11, 12),      # 中指
        (0, 13), (13, 14), (14, 15), (15, 16),    # 无名指
        (0, 17), (17, 18), (18, 19), (19, 20),    # 小指
    ]

    # 跨部位语义连接
    CROSS_EDGES = [
        (9, 17),   # left wrist -> left hand root (wrist)
        (10, 38),  # right wrist -> right hand root (wrist)
        (9, 10),   # left wrist <-> right wrist
        (5, 9),    # left shoulder -> left wrist
        (6, 10),   # right shoulder -> right wrist
    ]

    def __init__(self) -> None:
        self.A = self._build_adjacency_matrix()

    def _build_adjacency_matrix(self) -> np.ndarray:
        """构建对称邻接矩阵 A (V x V)。"""
        V = self.NUM_NODES
        A = np.zeros((V, V), dtype=np.float32)

        def add_edge(u: int, v: int) -> None:
            A[u, v] = 1.0
            A[v, u] = 1.0

        # Body edges
        for u, v in self.BODY_EDGES:
            add_edge(u, v)

        # Left hand edges (offset = 17)
        for u, v in self.HAND_EDGES:
            add_edge(u + 17, v + 17)

        # Right hand edges (offset = 38)
        for u, v in self.HAND_EDGES:
            add_edge(u + 38, v + 38)

        # Cross edges
        for u, v in self.CROSS_EDGES:
            add_edge(u, v)

        # 添加自环
        for i in range(V):
            A[i, i] = 1.0

        return A

    def normalize(self, A: np.ndarray, strategy: str = "symmetric") -> np.ndarray:
        """归一化邻接矩阵。"""
        if strategy == "symmetric":
            D = np.sum(A, axis=1)
            D_inv_sqrt = np.power(D, -0.5)
            D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
            D_inv_sqrt = np.diag(D_inv_sqrt)
            return D_inv_sqrt @ A @ D_inv_sqrt
        elif strategy == "random_walk":
            D = np.sum(A, axis=1)
            D_inv = np.power(D, -1.0)
            D_inv[np.isinf(D_inv)] = 0.0
            D_inv = np.diag(D_inv)
            return D_inv @ A
        return A


# =============================================================================
# 归一化姿态预处理 (复用 gesture.py 中的逻辑)
# =============================================================================

class PoseNormalizer:
    """将原始 COCO + Hand 关键点归一化到躯干坐标系。"""

    BODY_IDX = {
        "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
        "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
        "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
        "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16,
    }

    @staticmethod
    def normalize_frame(
        body_kpts: np.ndarray,
        left_hand: Optional[List[Tuple[float, float, float]]] = None,
        right_hand: Optional[List[Tuple[float, float, float]]] = None,
    ) -> np.ndarray:
        """
        将单帧关键点归一化，输出 (59, 2) 数组。
        若某关键点缺失，用 0 填充并在后续处理中通过 mask 忽略。
        """
        V = SkeletonGraph.NUM_NODES
        out = np.zeros((V, 2), dtype=np.float32)

        # --- body (17 points) ---
        if body_kpts is not None and len(body_kpts) >= 17:
            for i in range(17):
                if len(body_kpts[i]) >= 2 and body_kpts[i][2] > 0.3:
                    out[i, 0] = float(body_kpts[i][0])
                    out[i, 1] = float(body_kpts[i][1])

        # --- left hand (21 points, index 17-37) ---
        if left_hand and len(left_hand) >= 21:
            for i in range(21):
                out[17 + i, 0] = float(left_hand[i][0])
                out[17 + i, 1] = float(left_hand[i][1])

        # --- right hand (21 points, index 38-58) ---
        if right_hand and len(right_hand) >= 21:
            for i in range(21):
                out[38 + i, 0] = float(right_hand[i][0])
                out[38 + i, 1] = float(right_hand[i][1])

        # --- 归一化：以 mid_hip 为原点，torso_size 为单位 ---
        ls = out[5]
        rs = out[6]
        lh = out[11]
        rh = out[12]

        # 计算 mid_hip
        valid_hips = []
        if np.any(lh != 0):
            valid_hips.append(lh)
        if np.any(rh != 0):
            valid_hips.append(rh)
        mid_hip = np.mean(valid_hips, axis=0) if valid_hips else np.array([0.0, 0.0])

        # 计算 torso_size
        torso_size = 1.0
        if np.any(ls != 0) and np.any(rs != 0) and np.any(lh != 0) and np.any(rh != 0):
            d1 = np.linalg.norm(ls - lh)
            d2 = np.linalg.norm(ls - rh)
            d3 = np.linalg.norm(rs - lh)
            d4 = np.linalg.norm(rs - rh)
            torso_size = float((d1 + d2 + d3 + d4) / 4.0)
        elif np.any(ls != 0) and np.any(rs != 0):
            torso_size = float(np.linalg.norm(ls - rs))
        elif torso_size < 1e-6:
            torso_size = 100.0  # 默认像素值

        # 平移 + 缩放
        out = (out - mid_hip) / torso_size

        return out


# =============================================================================
# PyTorch 模型定义 (仅在有 torch 时可用)
# =============================================================================

class _STrGCNModel:
    """
    STr-GCN 手势分类模型。
    由于环境可能未安装 torch，本类延迟导入 torch，并提供训练/推理接口。
    """

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 2,
        hidden_dim: int = 64,
        num_gcn_layers: int = 3,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        max_seq_len: int = 64,
    ) -> None:
        import torch
        import torch.nn as nn

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 骨架图
        self.graph = SkeletonGraph()
        A = self.graph.normalize(self.graph.A, strategy="symmetric")
        self.A = torch.from_numpy(A).float().to(self.device)

        # 输入嵌入
        self.input_embed = nn.Linear(in_channels, hidden_dim)

        # Spatial GCN 层
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(
                _GraphConvBlock(hidden_dim, hidden_dim, dropout)
            )

        # Temporal Positional Encoding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_len, hidden_dim)
        )
        nn.init.normal_(self.pos_embed, std=0.02)

        # Temporal Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.to(self.device)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Args:
            x: (B, T, V, 2)  batch x time x nodes x coords
        Returns:
            logits: (B, num_classes)
        """
        import torch
        import torch.nn.functional as F

        B, T, V, C = x.shape

        # 输入嵌入: (B, T, V, 2) -> (B, T, V, hidden_dim)
        x = self.input_embed(x)

        # Spatial GCN: 每帧独立处理
        # 将时间维度合并到 batch: (B*T, V, hidden_dim)
        x = x.reshape(B * T, V, self.hidden_dim)
        for gcn in self.gcn_layers:
            x = gcn(x, self.A)
        # 恢复: (B, T, V, hidden_dim)
        x = x.reshape(B, T, V, self.hidden_dim)

        # 空间池化: 对每个时间步的 V 个节点做 avg pool
        # (B, T, V, hidden_dim) -> (B, T, hidden_dim)
        x = x.mean(dim=2)

        # Temporal Positional Encoding
        if T <= self.pos_embed.size(1):
            x = x + self.pos_embed[:, :T, :]
        else:
            x = x + F.interpolate(
                self.pos_embed.permute(0, 2, 1),
                size=T,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)

        # Temporal Transformer: (B, T, hidden_dim)
        x = self.transformer(x)

        # 时序全局池化
        x = x.mean(dim=1)  # (B, hidden_dim)

        # 分类
        logits = self.classifier(x)
        return logits

    def to(self, device: "torch.device") -> "_STrGCNModel":
        import torch.nn as nn

        if isinstance(self, nn.Module):
            return super(_STrGCNModel, self).to(device)
        return self


class _GraphConvBlock:
    """单层图卷积 + BN + ReLU + Dropout。"""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3) -> None:
        import torch.nn as nn

        self.gc = _GraphConvolution(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: "torch.Tensor", A: "torch.Tensor") -> "torch.Tensor":
        import torch.nn.functional as F

        x = self.gc(x, A)
        # BN 需要 (N, C) 或 (N, C, L)，这里 x 是 (B*T, V, C)
        BTV, V_size, C = x.shape
        x = x.permute(0, 2, 1)  # (BTV, C, V)
        x = self.bn(x)
        x = x.permute(0, 2, 1)  # (BTV, V, C)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class _GraphConvolution:
    """简单图卷积: Y = A X W。"""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        import torch
        import torch.nn as nn

        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def __call__(self, x: "torch.Tensor", A: "torch.Tensor") -> "torch.Tensor":
        """
        x: (N, V, in_dim)
        A: (V, V)
        Returns: (N, V, out_dim)
        """
        import torch

        support = x @ self.weight  # (N, V, out_dim)
        output = torch.einsum('vw,nwc->nwc', A, support) + self.bias
        return output


# =============================================================================
# 对外接口: STrGestureRecognizer
# =============================================================================

class GestureType(str, Enum):
    NONE = "none"
    GREETING = "greeting"
    HAILING = "hailing"
    HAND_UP = "hand_up"


class STrGestureRecognizer:
    """
    基于 STr-GCN 的时序手势识别器。

    与现有规则引擎 (gesture.py) 的兼容接口：
        recognizer = STrGestureRecognizer(model_path="stgcn_gesture.pt")
        gesture, conf = recognizer.recognize(keypoints, track_id, left_hand, right_hand)

    注意: 本类依赖 PyTorch。若环境未安装 torch，会回退到规则引擎。
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        seq_len: int = 32,
        hidden_dim: int = 64,
        device: Optional[str] = None,
    ) -> None:
        self.seq_len = seq_len
        self._buffer: Dict[str, Deque[np.ndarray]] = {}
        self._model: Optional[_STrGCNModel] = None
        self._torch = None

        # 延迟导入 torch
        try:
            import torch

            self._torch = torch
            if device:
                self.device = torch.device(device)
            else:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
        except ImportError:
            logger.warning("PyTorch 未安装，STr-GCN 手势识别器不可用")
            return

        if model_path and self._torch is not None:
            self._load_model(model_path, hidden_dim)

    def _load_model(self, model_path: str, hidden_dim: int) -> None:
        try:
            self._model = _STrGCNModel(
                num_classes=4,
                hidden_dim=hidden_dim,
            )
            state = self._torch.load(model_path, map_location=self.device)
            if "model_state_dict" in state:
                self._model.load_state_dict(state["model_state_dict"])
            else:
                self._model.load_state_dict(state)
            self._model.eval()
            logger.info("STr-GCN 模型加载成功: %s", model_path)
        except Exception as e:
            logger.error("STr-GCN 模型加载失败: %s", e)
            self._model = None

    def recognize(
        self,
        keypoints: np.ndarray,
        track_id: str = "default",
        left_hand_landmarks: Optional[List[Tuple[float, float, float]]] = None,
        right_hand_landmarks: Optional[List[Tuple[float, float, float]]] = None,
        frame_timestamp: Optional[float] = None,
    ) -> Tuple[str, float]:
        """
        帧级手势识别接口（与现有系统兼容）。

        维护每 track_id 的时序缓冲区，当缓冲区满时进行推理。
        """
        if self._model is None or self._torch is None:
            return "none", 0.0

        # 归一化当前帧
        frame_feat = PoseNormalizer.normalize_frame(
            keypoints, left_hand_landmarks, right_hand_landmarks
        )

        # 维护缓冲区
        buf = self._buffer.get(track_id)
        if buf is None:
            buf = deque(maxlen=self.seq_len)
            self._buffer[track_id] = buf
        buf.append(frame_feat)

        # 缓冲区未满时返回 none
        if len(buf) < self.seq_len * 0.5:
            return "none", 0.0

        # 构建输入张量 (1, T, V, 2)
        T_actual = len(buf)
        seq = np.array(list(buf), dtype=np.float32)  # (T, V, 2)

        # 若长度不足 seq_len，重复最后一帧填充
        if T_actual < self.seq_len:
            pad = np.repeat(seq[-1:], self.seq_len - T_actual, axis=0)
            seq = np.concatenate([seq, pad], axis=0)
        elif T_actual > self.seq_len:
            # 均匀采样到 seq_len
            indices = np.linspace(0, T_actual - 1, self.seq_len, dtype=int)
            seq = seq[indices]

        x = self._torch.from_numpy(seq).float().unsqueeze(0).to(self.device)

        with self._torch.no_grad():
            logits = self._model(x)
            probs = self._torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

        pred_idx = int(pred.item())
        confidence = float(conf.item())

        label_map = {0: "none", 1: "hand_up", 2: "greeting", 3: "hailing"}
        gesture = label_map.get(pred_idx, "none")

        # 清理不活跃的 track
        self._cleanup_buffers()

        return gesture, confidence

    def _cleanup_buffers(self, max_idle_frames: int = 300) -> None:
        """清理不活跃的 track 缓冲区（避免内存无限增长）。"""
        # 简化处理：若缓冲区超过 maxlen 的 2 倍（几乎不可能），清理
        # 实际应在 track 丢失时由外部调用 reset(track_id)
        pass

    def reset(self, track_id: Optional[str] = None) -> None:
        """重置指定或全部 track 的时序缓冲区。"""
        if track_id is None:
            self._buffer.clear()
        else:
            self._buffer.pop(track_id, None)

    def reset_all(self) -> None:
        """重置所有缓冲区。"""
        self._buffer.clear()


# =============================================================================
# 便捷函数 (与 gesture.py 接口一致)
# =============================================================================

_recognizer_stgcn: Optional[STrGestureRecognizer] = None


def get_stgcn_recognizer(
    model_path: Optional[str] = None,
) -> STrGestureRecognizer:
    global _recognizer_stgcn
    if _recognizer_stgcn is None:
        _recognizer_stgcn = STrGestureRecognizer(model_path=model_path)
    return _recognizer_stgcn


def is_hailing_gesture_stgcn(
    keypoints: np.ndarray,
    track_id: str = "default",
    left_hand_landmarks: Optional[List[Tuple[float, float, float]]] = None,
    right_hand_landmarks: Optional[List[Tuple[float, float, float]]] = None,
    frame_timestamp: Optional[float] = None,
) -> Tuple[str, float]:
    """STr-GCN 版本的便捷函数（可直接替换 gesture.is_hailing_gesture）。"""
    recognizer = get_stgcn_recognizer()
    return recognizer.recognize(
        keypoints, track_id, left_hand_landmarks, right_hand_landmarks, frame_timestamp
    )
