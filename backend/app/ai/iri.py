"""
IRI —— Intent Rigor Index（意图刚性指数）

在手臂局部坐标系中，计算 MediaPipe 手掌法向量相对于手臂的稳定性。
滑动窗口（15帧）内，法向量在手臂局部标架中的球面方差 → R ∈ [0,1]。

最终意图分数：S = Pose_score * R * Motion_score * F_human

注意：IRI 是加分项，不是阻塞项。R 接近 0 时压低置信度，R 接近 1 时保留原置信度。
"""

import logging
from typing import Optional, Tuple
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


def build_arm_frame(
    keypoints: np.ndarray,
    side: str = "right",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool]:
    """
    构建手臂局部正交标架 (e_x, e_y, e_z)。

    原点：elbow
    e_x = normalize(shoulder - elbow)   # 上臂方向（指向肩膀）
    e_y = normalize(wrist - elbow)      # 前臂方向（指向手腕）
    e_z = cross(e_x, e_y)               # 垂直于手臂平面

    注意：e_x 与 e_y 在肘关节处不严格正交（theta2 可能 < 180°），
    但 e_z 仍定义了手臂平面的法向，足够稳定用于 IRI。

    Returns:
        (origin, e_x, e_y, e_z, ok)
        origin: elbow 位置 (2D 像素)
        e_x, e_y, e_z: 3D 标架基向量（z=0，因为 keypoints 只有 x,y）
        ok: 是否成功构建
    """
    if keypoints is None or len(keypoints) < 17:
        return None, None, None, None, False

    s_idx = 5 if side == "left" else 6
    e_idx = 7 if side == "left" else 8
    w_idx = 9 if side == "left" else 10

    shoulder = keypoints[s_idx]
    elbow = keypoints[e_idx]
    wrist = keypoints[w_idx]

    if any(kp[2] < 0.3 for kp in [shoulder, elbow, wrist]):
        return None, None, None, None, False

    origin = np.array(elbow[:2], dtype=float)
    s_vec = np.array(shoulder[:2], dtype=float)
    w_vec = np.array(wrist[:2], dtype=float)

    e_x = s_vec - origin
    e_y = w_vec - origin

    norm_ex = np.linalg.norm(e_x)
    norm_ey = np.linalg.norm(e_y)
    if norm_ex < 1e-6 or norm_ey < 1e-6:
        return None, None, None, None, False

    e_x = e_x / norm_ex
    e_y = e_y / norm_ey

    # e_z 垂直于手臂平面（2D 中 z=0，所以 e_z = [0,0,±1]）
    cross_z = e_x[0] * e_y[1] - e_x[1] * e_y[0]
    e_z = np.array([0.0, 0.0, np.sign(cross_z) if abs(cross_z) > 1e-6 else 1.0])

    # 升维到 3D：e_x, e_y 的 z 分量为 0
    e_x_3d = np.array([e_x[0], e_x[1], 0.0])
    e_y_3d = np.array([e_y[0], e_y[1], 0.0])

    return origin, e_x_3d, e_y_3d, e_z, True


def world_normal_to_arm_frame(
    normal_world: np.ndarray,
    e_x: np.ndarray,
    e_y: np.ndarray,
    e_z: np.ndarray,
) -> Optional[np.ndarray]:
    """
    将世界坐标系中的手掌法向量转换到手臂局部标架。

    Returns:
        n_local (3,) —— 单位向量，或 None
    """
    n = np.asarray(normal_world, dtype=float)
    n_len = np.linalg.norm(n)
    if n_len < 1e-6:
        return None
    n = n / n_len

    # 由于 e_x, e_y 的 z=0，e_z=[0,0,±1]，直接投影
    nx = float(np.dot(n, e_x))
    ny = float(np.dot(n, e_y))
    nz = float(np.dot(n, e_z))

    local = np.array([nx, ny, nz])
    local_len = np.linalg.norm(local)
    if local_len < 1e-6:
        return None
    return local / local_len


class IRICalculator:
    """
    意图刚性指数计算器。

    滑动窗口内，法向量在手臂局部标架中的球面集中度 → R ∈ [0,1]。
    """

    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self._history: deque = deque(maxlen=window_size)

    def feed(
        self,
        keypoints: np.ndarray,
        side: str,
        palm_normal_world: Optional[np.ndarray],
    ) -> float:
        """
        喂入新一帧，返回当前 R 值。

        Returns:
            R ∈ [0, 1]（1=最稳定，0=最不稳定）
        """
        if palm_normal_world is None:
            # 无法向量时不清空历史，而是让 R 自然衰减
            return self._compute_r()

        origin, e_x, e_y, e_z, ok = build_arm_frame(keypoints, side)
        if not ok or e_x is None:
            return self._compute_r()

        n_local = world_normal_to_arm_frame(palm_normal_world, e_x, e_y, e_z)
        if n_local is None:
            return self._compute_r()

        self._history.append(n_local)
        return self._compute_r()

    def _compute_r(self) -> float:
        """基于历史法向量计算球面集中度 R。"""
        if len(self._history) < 3:
            return 1.0  # neutral: insufficient data, do not penalize score

        # 平均向量长度作为集中度度量
        mean_vec = np.mean(list(self._history), axis=0)
        mean_len = np.linalg.norm(mean_vec)

        # 归一化到 [0, 1]
        # 理论上最大值为 1.0（所有向量相同），最小值接近 0（均匀分布）
        r = float(np.clip(mean_len, 0.0, 1.0))
        return r

    def reset(self) -> None:
        self._history.clear()
