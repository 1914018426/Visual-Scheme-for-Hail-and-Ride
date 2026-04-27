"""
Torso-Normalized Local Frame (TNLF) —— 人体局部参考系

所有轨迹、速度、周期性判断必须在人体局部参考系中进行，
禁止直接使用画面像素坐标。
"""

import logging
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


def wrist_to_local_frame(
    keypoints: np.ndarray,
    side: str = "right",
) -> Tuple[Optional[Tuple[float, float]], Optional[float], bool]:
    """
    将手腕坐标从画面像素转换到人体躯干归一化局部坐标系。

    Args:
        keypoints: YOLO11-Pose 17点 [x, y, conf]
        side: "left" 或 "right"

    Returns:
        (wrist_local, torso_scale, frame_valid)
        - wrist_local: (x_local, y_local)，单位：躯干长度
        - torso_scale: 肩中点到髋中点的距离（像素），作为归一化单位
        - frame_valid: 关键点是否足够可信
    """
    if keypoints is None or len(keypoints) < 17:
        return None, None, False

    l_shoulder = keypoints[5][:2]
    r_shoulder = keypoints[6][:2]
    l_hip = keypoints[11][:2]
    r_hip = keypoints[12][:2]

    # 关键点置信度检查
    min_conf = 0.3
    confs = [keypoints[i][2] for i in [5, 6, 11, 12]]
    if any(c < min_conf for c in confs):
        return None, None, False

    wrist_idx = 9 if side == "left" else 10
    wrist = keypoints[wrist_idx][:2]
    if keypoints[wrist_idx][2] < min_conf:
        return None, None, False

    # 原点：肩中点
    origin = (l_shoulder + r_shoulder) / 2.0

    # 局部基向量（非严格正交，但足够稳定）
    e_x = r_shoulder - l_shoulder  # 肩宽方向
    e_y = (l_hip + r_hip) / 2.0 - origin  # 躯干方向

    # 躯干尺度：肩中点到髋中点的距离，作为归一化单位
    torso_scale = float(np.linalg.norm(e_y))
    if torso_scale < 10.0:  # 检测失效
        return None, None, False

    # 归一化
    norm_ex = np.linalg.norm(e_x)
    if norm_ex < 1e-6:
        return None, None, False
    e_x = e_x / norm_ex
    e_y = e_y / torso_scale

    # 手腕相对向量
    w_vec = wrist - origin

    # 投影到局部坐标系（以 torso_scale 为单位长度）
    x_local = float(np.dot(w_vec, e_x) / torso_scale)
    y_local = float(np.dot(w_vec, e_y) / torso_scale)

    return (x_local, y_local), torso_scale, True


def local_velocity(
    wrist_local_prev: Tuple[float, float],
    wrist_local_curr: Tuple[float, float],
    dt: float,
) -> Tuple[float, float, float]:
    """
    计算 wrist_local 的速度，单位：躯干长度/秒 (torso_units/s)。

    Args:
        wrist_local_prev: 上一帧局部坐标
        wrist_local_curr: 当前帧局部坐标
        dt: 时间差（秒）

    Returns:
        (vx_tu, vy_tu, magnitude_tu)
    """
    if dt < 1e-6:
        return 0.0, 0.0, 0.0
    vx = (wrist_local_curr[0] - wrist_local_prev[0]) / dt
    vy = (wrist_local_curr[1] - wrist_local_prev[1]) / dt
    mag = float(np.hypot(vx, vy))
    return vx, vy, mag


def local_to_pixel(
    wrist_local: Tuple[float, float],
    keypoints: np.ndarray,
) -> Optional[Tuple[int, int]]:
    """
    将 wrist_local 反投影回画面像素坐标（仅用于可视化）。

    Args:
        wrist_local: (x_local, y_local)
        keypoints: YOLO11-Pose 17点 [x, y, conf]

    Returns:
        (px, py) 或 None
    """
    if keypoints is None or len(keypoints) < 17:
        return None

    l_shoulder = keypoints[5][:2]
    r_shoulder = keypoints[6][:2]
    l_hip = keypoints[11][:2]
    r_hip = keypoints[12][:2]

    min_conf = 0.3
    confs = [keypoints[i][2] for i in [5, 6, 11, 12]]
    if any(c < min_conf for c in confs):
        return None

    origin = (l_shoulder + r_shoulder) / 2.0
    e_x = r_shoulder - l_shoulder
    e_y = (l_hip + r_hip) / 2.0 - origin

    torso_scale = float(np.linalg.norm(e_y))
    if torso_scale < 10.0:
        return None

    norm_ex = np.linalg.norm(e_x)
    if norm_ex < 1e-6:
        return None
    e_x = e_x / norm_ex
    e_y = e_y / torso_scale

    w_vec = (
        origin
        + wrist_local[0] * torso_scale * e_x
        + wrist_local[1] * torso_scale * e_y
    )
    return int(w_vec[0]), int(w_vec[1])
