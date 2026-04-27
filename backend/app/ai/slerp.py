"""
球面线性插值 (SLERP) 模块

用于 MediaPipe 手掌法向量的平滑，禁止欧氏空间直接 lerp。
"""

import logging
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


def slerp(n_prev: np.ndarray, n_curr: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    球面线性插值：在球面上平滑两个单位向量。

    Args:
        n_prev: 上一帧单位法向量 (3,)
        n_curr: 当前帧单位法向量 (3,)
        alpha: 插值权重，0 = 全取 prev，1 = 全取 curr

    Returns:
        平滑后的单位法向量 (3,)
    """
    n_prev = np.asarray(n_prev, dtype=float)
    n_curr = np.asarray(n_curr, dtype=float)

    # 确保单位长度
    len_prev = np.linalg.norm(n_prev)
    len_curr = np.linalg.norm(n_curr)
    if len_prev > 1e-6:
        n_prev = n_prev / len_prev
    if len_curr > 1e-6:
        n_curr = n_curr / len_curr

    dot = np.clip(np.dot(n_prev, n_curr), -1.0, 1.0)

    # 如果夹角极小，退化为欧氏 lerp（在切平面上等价）
    if dot > 0.9995:
        result = n_prev * (1 - alpha) + n_curr * alpha
        result_len = np.linalg.norm(result)
        if result_len > 1e-6:
            return result / result_len
        return n_prev

    theta_0 = np.arccos(dot)
    theta = theta_0 * alpha

    sin_theta_0 = np.sin(theta_0)
    if abs(sin_theta_0) < 1e-6:
        return n_prev

    result = (n_prev * np.sin(theta_0 - theta) + n_curr * np.sin(theta)) / sin_theta_0
    result_len = np.linalg.norm(result)
    if result_len > 1e-6:
        return result / result_len
    return n_prev


def compute_palm_normal(
    hand_landmarks: list,
) -> Tuple[Optional[np.ndarray], bool]:
    """
    从 MediaPipe 21 点 landmarks 计算手掌平面法向量。

    Args:
        hand_landmarks: 21 个 (x, y, z) 元组/列表

    Returns:
        (normal, ok)
        - normal: 3D 单位法向量 (3,) numpy array，或 None
        - ok: 是否成功计算
    """
    if not hand_landmarks or len(hand_landmarks) < 21:
        return None, False

    pts = np.array(hand_landmarks)
    wrist_3d = pts[0]
    index_mcp_3d = pts[5]
    pinky_mcp_3d = pts[17]

    v1 = index_mcp_3d - wrist_3d
    v2 = pinky_mcp_3d - wrist_3d
    normal = np.cross(v1[:3], v2[:3])
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-6:
        return None, False

    return normal / norm_len, True


def angle_to_camera_z(normal: np.ndarray) -> float:
    """
    法向量与摄像头视线方向的夹角（度）。

    MediaPipe 坐标系中 Z 轴指向屏幕外（朝 viewer）。
    掌心朝车（朝摄像头）时，法向量大致指向 Z 轴负方向，
    因此这里计算的是法向量与 Z 轴负方向 [0, 0, -1] 的夹角。
    夹角越小 → 掌心越正对摄像头/车。
    """
    n = np.asarray(normal, dtype=float)
    n_len = np.linalg.norm(n)
    if n_len < 1e-6:
        return 180.0
    n = n / n_len
    # 摄像头视线方向 = Z 轴负方向（从场景指向摄像头）
    view_dir = np.array([0.0, 0.0, -1.0])
    dot = np.clip(np.dot(n, view_dir), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


class NormalSmoother:
    """
    法向量球面指数移动平均平滑器。
    每帧新法向量 n_new 与历史平滑值 n_smooth 做 SLERP。
    """

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._smooth: Optional[np.ndarray] = None

    def update(self, n_new: np.ndarray) -> np.ndarray:
        """喂入新法向量，返回平滑后的单位法向量。"""
        n_new = np.asarray(n_new, dtype=float)
        n_len = np.linalg.norm(n_new)
        if n_len < 1e-6:
            return self._smooth.copy() if self._smooth is not None else n_new
        n_new = n_new / n_len

        if self._smooth is None:
            self._smooth = n_new.copy()
            return self._smooth.copy()

        self._smooth = slerp(self._smooth, n_new, self.alpha)
        return self._smooth.copy()

    def reset(self) -> None:
        self._smooth = None

    @property
    def value(self) -> Optional[np.ndarray]:
        return self._smooth.copy() if self._smooth is not None else None
