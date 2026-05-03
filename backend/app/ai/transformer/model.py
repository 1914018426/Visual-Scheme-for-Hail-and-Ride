"""
Temporal Keypoint Transformer for pedestrian waving-intent recognition.

Input:  [B, T, F]  where T=45 frames, F=12 TNLF-normalized features
Output: [B, 1]     sigmoid confidence (waving probability)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[: x.size(1)])


class TemporalKeypointTransformer(nn.Module):
    """
    Lightweight temporal transformer for binary waving classification.

    Args:
        n_features:   Number of input features per frame (default 12)
        d_model:      Transformer hidden dimension
        n_head:       Number of attention heads
        n_layers:     Number of transformer encoder layers
        dim_feedforward: FFN inner dimension
        seq_len:      Expected sequence length (for positional encoding init)
        dropout:      Dropout rate
    """

    def __init__(
        self,
        n_features: int = 12,
        d_model: int = 128,
        n_head: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 512,
        seq_len: int = 45,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.seq_len = seq_len

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len + 16, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            enable_nested_tensor=False,
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.constant_(p, 0.0)
        nn.init.normal_(self.cls_token, std=0.02)
        # Zero-init final linear layer bias so model starts at p=0.5
        last_linear = self.head[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.constant_(last_linear.bias, 0.0)
            nn.init.xavier_uniform_(last_linear.weight, gain=0.01)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    [B, T, F] feature sequence
            mask: [B, T]   True where padded (optional)
        Returns:
            [B, 1] sigmoid confidence
        """
        B = x.shape[0]

        x = self.input_proj(x) * math.sqrt(self.d_model)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        if mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=mask.dtype, device=mask.device)
            src_key_padding_mask = torch.cat([cls_mask, mask], dim=1)
        else:
            src_key_padding_mask = None

        x = self.pos_encoding(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        cls_out = x[:, 0, :]
        return torch.sigmoid(self.head(cls_out))

    def get_attention_weights(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return attention weights from the final encoder layer (for interpretability)."""
        B = x.shape[0]
        x = self.input_proj(x) * math.sqrt(self.d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        if mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=mask.dtype, device=mask.device)
            src_key_padding_mask = torch.cat([cls_mask, mask], dim=1)
        else:
            src_key_padding_mask = None

        x = self.pos_encoding(x)

        # Manually forward through encoder to capture last layer attention
        for i, layer in enumerate(self.encoder.layers):
            if i == len(self.encoder.layers) - 1:
                # Capture attention from final layer's self-attention
                x2 = layer.self_attn(x, x, x, key_padding_mask=src_key_padding_mask, need_weights=True)
                attn_output, attn_weights = x2
                x = layer.norm1(x + layer.dropout1(attn_output))
                x = layer.norm2(x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))))
                return attn_weights
            x = layer(x, src_key_padding_mask=src_key_padding_mask)

        return torch.empty(0)


def compute_arm_angles(kpts: torch.Tensor, side: str) -> Tuple[float, float, float]:
    """
    Compute arm pose angles from COCO-17 keypoints.

    Args:
        kpts: [17, 3] keypoints (x, y, conf)
        side: "left" or "right"
    Returns:
        (theta1, theta2, ext_ratio)
        theta1:   hip-shoulder-elbow angle (arm lift, degrees)
        theta2:   shoulder-elbow-wrist angle (arm straightness, degrees)
        ext_ratio: arm extension ratio
    """
    if side == "left":
        hip, shoulder, elbow, wrist = 11, 5, 7, 9
    else:
        hip, shoulder, elbow, wrist = 12, 6, 8, 10

    h = kpts[hip, :2]
    s = kpts[shoulder, :2]
    e = kpts[elbow, :2]
    w = kpts[wrist, :2]

    v_se = e - s
    v_sh = h - s
    v_ew = w - e
    v_sw = w - s

    def angle_between(v1: torch.Tensor, v2: torch.Tensor) -> float:
        cos_a = (v1 @ v2) / (v1.norm() * v2.norm() + 1e-8)
        return float(torch.rad2deg(torch.acos(cos_a.clamp(-1.0, 1.0))))

    theta1 = angle_between(v_se, v_sh)
    theta2 = angle_between(-v_se, v_ew)
    ext_ratio = float(v_sw.norm() / (v_se.norm() + v_ew.norm() + 1e-8))

    return theta1, theta2, ext_ratio
