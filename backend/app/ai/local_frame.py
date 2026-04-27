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
    （向后兼容包装，内部调用 full 版本）
    """
    wl, _, _, ts, valid = wrist_to_local_frame_full(keypoints, side)
    return wl, ts, valid


def wrist_to_local_frame_full(
    keypoints: np.ndarray,
    side: str = "right",
) -> Tuple[
    Optional[Tuple[float, float]],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[float],
    bool,
]:
    """
    将手腕坐标从画面像素转换到人体躯干归一化局部坐标系，并返回完整标架。

    Args:
        keypoints: YOLO11-Pose 17点 [x, y, conf]
        side: "left" 或 "right"

    Returns:
        (wrist_local, origin, e_x, e_y, torso_scale, frame_valid)
        - wrist_local: (x_local, y_local)，单位：躯干长度
        - origin: 肩中点 (2,)
        - e_x: 肩宽方向单位向量 (2,)
        - e_y: 躯干方向单位向量 (2,)，与 e_x 不正交
        - torso_scale: 肩中点到髋中点的距离（像素）
        - frame_valid: 关键点是否足够可信
    """
    if keypoints is None or len(keypoints) < 17:
        return None, None, None, None, None, False

    l_shoulder = keypoints[5][:2]
    r_shoulder = keypoints[6][:2]
    l_hip = keypoints[11][:2]
    r_hip = keypoints[12][:2]

    # 关键点置信度检查
    min_conf = 0.3
    confs = [keypoints[i][2] for i in [5, 6, 11, 12]]
    if any(c < min_conf for c in confs):
        return None, None, None, None, None, False

    wrist_idx = 9 if side == "left" else 10
    wrist = keypoints[wrist_idx][:2]
    if keypoints[wrist_idx][2] < min_conf:
        return None, None, None, None, None, False

    # 原点：肩中点
    origin = (l_shoulder + r_shoulder) / 2.0

    # 局部基向量
    e_x_raw = r_shoulder - l_shoulder  # 肩宽方向
    e_y_raw = (l_hip + r_hip) / 2.0 - origin  # 躯干方向

    # 躯干尺度
    torso_scale = float(np.linalg.norm(e_y_raw))
    if torso_scale < 10.0:
        return None, None, None, None, None, False

    # 归一化
    norm_ex = np.linalg.norm(e_x_raw)
    if norm_ex < 1e-6:
        return None, None, None, None, None, False
    e_x_unit = e_x_raw / norm_ex
    e_y_unit = e_y_raw / torso_scale

    # 手腕相对向量
    w_vec = wrist - origin

    # 投影到局部坐标系（均以 torso_scale 为单位长度）
    x_local = float(np.dot(w_vec, e_x_unit) / torso_scale)
    y_local = float(np.dot(w_vec, e_y_unit) / torso_scale)

    return (
        (x_local, y_local),
        origin.astype(float),
        e_x_unit.astype(float),
        e_y_unit.astype(float),
        torso_scale,
        True,
    )


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


def local_to_pixel_with_frame(
    wrist_local: Tuple[float, float],
    origin: np.ndarray,
    e_x: np.ndarray,
    e_y: np.ndarray,
    torso_scale: float,
) -> Optional[Tuple[int, int]]:
    """
    将 wrist_local 反投影回画面像素坐标（使用对应帧的标架快照）。

    Args:
        wrist_local: (x_local, y_local)
        origin: 肩中点 (2,)
        e_x: 肩宽方向单位向量 (2,)
        e_y: 躯干方向单位向量 (2,)
        torso_scale: 躯干尺度（像素）

    Returns:
        (px, py) 或 None
    """
    if origin is None or e_x is None or e_y is None or torso_scale is None or torso_scale < 1.0:
        return None

    # 投影公式：
    #   x_local = dot(w_vec, e_x) / torso_scale
    #   y_local = dot(w_vec, e_y) / torso_scale
    # 反投影：
    #   w_vec = x_local * torso_scale * e_x + y_local * torso_scale * e_y
    #   wrist_pixel = origin + w_vec

    w_vec = (
        wrist_local[0] * torso_scale * e_x
        + wrist_local[1] * torso_scale * e_y
    )
    px = int(origin[0] + w_vec[0])
    py = int(origin[1] + w_vec[1])
    return px, py


def local_to_pixel(
    wrist_local: Tuple[float, float],
    keypoints: np.ndarray,
) -> Optional[Tuple[int, int]]:
    """
    将 wrist_local 反投影回画面像素坐标（使用当前帧标架）。
    
    注意：此函数存在反投影错位问题，推荐改用 local_to_pixel_with_frame。
    保留用于向后兼容。
    """
    wl, origin, e_x, e_y, torso_scale, valid = wrist_to_local_frame_full(keypoints)
    if not valid or wl is None:
        return None
    return local_to_pixel_with_frame(wrist_local, origin, e_x, e_y, torso_scale)
