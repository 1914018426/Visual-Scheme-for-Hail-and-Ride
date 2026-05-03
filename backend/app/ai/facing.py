"""
面部过滤层 —— zero-model 人体朝向评分

基于 YOLO11-Pose 17 点关键点，无需额外模型。
禁止引入 YOLO-Face / MediaPipe Face / 深度相机。
"""

import logging
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


def human_facing_score(kpts: np.ndarray) -> float:
    """
    计算人体面向摄像头的程度 F_human ∈ [0, 1]。

    1 = 正脸正对摄像头，0 = 背对摄像头。

    逻辑：
    - 面部关键点置信度门控（低置信度 → 侧/背面，降级为躯干推断）
    - 双眼到鼻子距离对称性
    - 肩宽 / 髋宽解剖比（正常人体约 1.25）
    """
    if kpts is None or len(kpts) < 17:
        return 0.0

    # 面部关键点置信度门控：低置信度意味着侧/背面视角，几何计算不可靠
    face_keypoint_conf = float(np.mean([kpts[i][2] for i in [0, 1, 2]]))

    if face_keypoint_conf < 0.3:
        # 面部关键点不可靠，降级为躯干比例推断（上限 0.3 防止侧视误判）
        sc = float(np.mean([kpts[5][2], kpts[6][2]]))
        hc = float(np.mean([kpts[11][2], kpts[12][2]]))
        if sc < 0.3 or hc < 0.3:
            return 0.0
        shoulder_w = float(np.linalg.norm(kpts[5, :2] - kpts[6, :2]))
        hip_w = float(np.linalg.norm(kpts[11, :2] - kpts[12, :2]))
        body_score = 1.0 - abs(shoulder_w / (hip_w + 1e-6) - 1.25) / 0.8
        return float(np.clip(0.3 * max(0.0, body_score), 0.0, 1.0))

    # 面部对称性：左眼、右眼到鼻子的距离
    nose_xy = kpts[0][:2]
    l_eye_xy = kpts[1][:2]
    r_eye_xy = kpts[2][:2]

    d_leye = float(np.linalg.norm(l_eye_xy - nose_xy))
    d_reye = float(np.linalg.norm(r_eye_xy - nose_xy))

    eye_sym = min(d_leye, d_reye) / (max(d_leye, d_reye) + 1e-6)

    # 躯干比例（带置信度检查）
    sc = float(np.mean([kpts[5][2], kpts[6][2]]))
    hc = float(np.mean([kpts[11][2], kpts[12][2]]))
    if sc < 0.3 or hc < 0.3:
        body_score = 0.5  # 躯干不可靠时取中性值
    else:
        shoulder_w = float(np.linalg.norm(kpts[5, :2] - kpts[6, :2]))
        hip_w = float(np.linalg.norm(kpts[11, :2] - kpts[12, :2]))
        body_score = 1.0 - abs(shoulder_w / (hip_w + 1e-6) - 1.25) / 0.8

    score = 0.6 * face_keypoint_conf * eye_sym + 0.4 * max(0.0, body_score)
    return float(np.clip(score, 0.0, 1.0))


def facing_gate(
    kpts: np.ndarray,
    hard_threshold: float = 0.25,
    soft_threshold: float = 0.6,
) -> Tuple[float, bool, float]:
    """
    面向度门控：硬过滤 + 软调制。

    Args:
        hard_threshold: 硬过滤阈值，F_human < 此值直接丢弃
        soft_threshold: 软过滤上限，F_human ∈ [hard, soft] 时线性衰减

    Returns:
        (f_human, is_hard_rejected, soft_multiplier)
        - f_human: 原始面向分数
        - is_hard_rejected: True 则直接丢弃该目标
        - soft_multiplier: 软调制系数，最终意图分数 *= multiplier
    """
    f_human = human_facing_score(kpts)

    # 硬过滤
    if f_human < hard_threshold:
        return f_human, True, 0.0

    # 软调制
    if f_human < soft_threshold:
        multiplier = 0.5 + 0.5 * (f_human - hard_threshold) / (soft_threshold - hard_threshold)
    else:
        multiplier = 1.0

    return f_human, False, multiplier
