"""
手势识别模块

提供招手打车手势的检测功能。
基于人体关键点和手部轨迹分析，判断是否为招手动作。

招手判定规则:
1. 举手判定 (hand_up): 手臂伸直上举，手腕在肩膀上方
2. 挥动判定 (wave): 手腕在肩膀上方且有明显的水平挥动
"""

import logging
import time
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from app.config import get_config

logger = logging.getLogger(__name__)


class GestureType(str, Enum):
    """手势类型枚举。"""

    NONE = "none"        # 无手势
    HAND_UP = "hand_up"  # 举手
    WAVE = "wave"        # 挥手（招手）
    UNKNOWN = "unknown"  # 未知


@dataclass
class GestureResult:
    """手势识别结果。"""

    gesture_type: GestureType = GestureType.NONE  # 手势类型
    confidence: float = 0.0                        # 置信度 (0-1)
    wrist_pos: Optional[Tuple[float, float]] = None  # 手腕位置 (x, y) 归一化坐标
    is_hailing: bool = False                       # 是否为招手打车手势


class GestureRecognizer:
    """
    手势识别器

    基于人体姿态关键点和历史帧轨迹分析，识别招手打车手势。
    维护手腕位置历史队列，用于检测挥动动作。
    """

    # COCO姿态关键点索引
    KEYPOINT_NOSE = 0
    KEYPOINT_LEFT_SHOULDER = 5
    KEYPOINT_RIGHT_SHOULDER = 6
    KEYPOINT_LEFT_ELBOW = 7
    KEYPOINT_RIGHT_ELBOW = 8
    KEYPOINT_LEFT_WRIST = 9
    KEYPOINT_RIGHT_WRIST = 10
    KEYPOINT_LEFT_HIP = 11
    KEYPOINT_RIGHT_HIP = 12

    def __init__(self, history_frames: int = 10) -> None:
        """
        初始化手势识别器。

        Args:
            history_frames: 历史帧数量，用于挥动检测
        """
        self.config = get_config()
        self.history_frames = history_frames
        self.wave_threshold = self.config.ai.wave_threshold

        # 手腕位置历史队列 {track_id: deque[(x, y, timestamp), ...]}
        self._wrist_history: Dict[str, deque] = {}

    def _get_history(self, track_id: str) -> deque:
        """
        获取指定跟踪ID的手腕位置历史队列。

        Args:
            track_id: 跟踪目标唯一ID

        Returns:
            手腕位置历史双端队列
        """
        if track_id not in self._wrist_history:
            self._wrist_history[track_id] = deque(maxlen=self.history_frames)
        return self._wrist_history[track_id]

    def _clear_history(self, track_id: str) -> None:
        """清除指定跟踪ID的历史记录。"""
        self._wrist_history.pop(track_id, None)

    def _detect_hand_up(
        self, keypoints: np.ndarray, side: str = "right"
    ) -> Tuple[bool, float, Optional[Tuple[float, float]]]:
        """
        检测举手动作：手臂伸直上举。

        判定条件:
        - 手腕y坐标 < 手肘y坐标 < 肩膀y坐标（图像坐标系y轴向下）
        - 手腕在肩膀上方

        Args:
            keypoints: 姿态关键点数组 (17, 3) 每行 [x, y, confidence]
            side: 检测哪只手 'left' 或 'right'

        Returns:
            (是否举手, 置信度, 手腕位置)
        """
        if side == "left":
            shoulder_idx = self.KEYPOINT_LEFT_SHOULDER
            elbow_idx = self.KEYPOINT_LEFT_ELBOW
            wrist_idx = self.KEYPOINT_LEFT_WRIST
        else:
            shoulder_idx = self.KEYPOINT_RIGHT_SHOULDER
            elbow_idx = self.KEYPOINT_RIGHT_ELBOW
            wrist_idx = self.KEYPOINT_RIGHT_WRIST

        # 获取关键点（归一化坐标 0-1）
        shoulder = keypoints[shoulder_idx]
        elbow = keypoints[elbow_idx]
        wrist = keypoints[wrist_idx]

        # 检查关键点置信度
        min_conf = 0.3
        if shoulder[2] < min_conf or elbow[2] < min_conf or wrist[2] < min_conf:
            return False, 0.0, None

        shoulder_y, elbow_y, wrist_y = shoulder[1], elbow[1], wrist[1]
        wrist_pos = (float(wrist[0]), float(wrist[1]))

        # 举手判定：手腕y < 手肘y < 肩膀y（y轴向下，数值越小越靠上）
        is_arm_straight_up = (wrist_y < elbow_y) and (elbow_y < shoulder_y)

        # 手腕在肩膀上方
        wrist_above_shoulder = wrist_y < shoulder_y

        # 综合判定
        is_hand_up = is_arm_straight_up and wrist_above_shoulder

        # 计算置信度：手臂伸直程度
        if is_hand_up:
            # 越直越高置信度
            straightness = max(0.0, 1.0 - abs(elbow_y - (shoulder_y + wrist_y) / 2) * 2)
            confidence = 0.5 + straightness * 0.5
            return True, min(confidence, 1.0), wrist_pos

        return False, 0.0, wrist_pos

    def _detect_wave(
        self,
        keypoints: np.ndarray,
        track_id: str,
        side: str = "right",
    ) -> Tuple[bool, float, Optional[Tuple[float, float]]]:
        """
        检测挥手（招手）动作。

        判定条件:
        - 手腕在肩膀上方（举高手臂）
        - 历史帧中手腕x坐标有明显变化（挥动）

        Args:
            keypoints: 姿态关键点数组 (17, 3)
            track_id: 跟踪目标ID（用于关联历史帧）
            side: 检测哪只手 'left' 或 'right'

        Returns:
            (是否挥手, 置信度, 手腕位置)
        """
        if side == "left":
            shoulder_idx = self.KEYPOINT_LEFT_SHOULDER
            wrist_idx = self.KEYPOINT_LEFT_WRIST
            hip_idx = self.KEYPOINT_LEFT_HIP
        else:
            shoulder_idx = self.KEYPOINT_RIGHT_SHOULDER
            wrist_idx = self.KEYPOINT_RIGHT_WRIST
            hip_idx = self.KEYPOINT_RIGHT_HIP

        shoulder = keypoints[shoulder_idx]
        wrist = keypoints[wrist_idx]
        hip = keypoints[hip_idx]
        l_shoulder = keypoints[self.KEYPOINT_LEFT_SHOULDER]
        r_shoulder = keypoints[self.KEYPOINT_RIGHT_SHOULDER]

        if shoulder[2] < 0.3 or wrist[2] < 0.3:
            return False, 0.0, None

        wrist_pos = (float(wrist[0]), float(wrist[1]))
        wrist_x, wrist_y = wrist_pos
        shoulder_y = float(shoulder[1])
        ts = time.time()

        # 以人体尺度自适应阈值，避免像素坐标下误触发
        if l_shoulder[2] > 0.3 and r_shoulder[2] > 0.3:
            shoulder_width = abs(float(r_shoulder[0]) - float(l_shoulder[0]))
        else:
            shoulder_width = 0.0
        if shoulder_width < 8.0:
            shoulder_width = max(
                8.0, abs(float(shoulder[0]) - wrist_x) * 2.2
            )

        # 手腕必须在肩膀上方
        torso_h = (
            abs(float(hip[1]) - shoulder_y)
            if hip[2] > 0.3
            else shoulder_width * 0.8
        )
        min_raise = max(8.0, torso_h * 0.08)
        if wrist_y >= shoulder_y - min_raise:
            return False, 0.0, wrist_pos

        # 获取历史队列并记录当前位置
        history = self._get_history(track_id)
        history.append((wrist_x, wrist_y, ts))

        # 需要足够历史帧才能检测稳定挥手
        if len(history) < 4:
            return False, 0.0, wrist_pos

        # 计算历史帧中手腕x坐标的最大差值
        wrist_xs = [pos[0] for pos in history]
        wrist_ys = [pos[1] for pos in history]
        x_diff = max(wrist_xs) - min(wrist_xs)
        y_diff = max(wrist_ys) - min(wrist_ys)

        # 水平振荡: 需要出现至少一次明显方向反转
        min_step = max(3.0, shoulder_width * 0.08)
        direction_changes = 0
        prev_sign = 0
        for i in range(1, len(wrist_xs)):
            dx = wrist_xs[i] - wrist_xs[i - 1]
            sign = 1 if dx > min_step else (-1 if dx < -min_step else 0)
            if sign == 0:
                continue
            if prev_sign != 0 and sign != prev_sign:
                direction_changes += 1
            prev_sign = sign

        dynamic_threshold = max(
            float(self.wave_threshold), shoulder_width * 0.35
        )
        is_waving = (
            x_diff > dynamic_threshold
            and direction_changes >= 1
            and y_diff < shoulder_width * 0.6
        )

        if is_waving:
            # 置信度：挥动幅度越大置信度越高
            amp_score = min(1.0, x_diff / (dynamic_threshold * 1.8))
            dir_score = min(1.0, 0.55 + 0.25 * direction_changes)
            stability_score = max(0.0, 1.0 - (y_diff / (shoulder_width * 0.7)))
            confidence = min(
                1.0, 0.55 * amp_score + 0.25 * dir_score + 0.20 * stability_score
            )
            return True, confidence, wrist_pos

        return False, 0.0, wrist_pos

    def recognize(
        self, keypoints: np.ndarray, track_id: str = "default"
    ) -> GestureResult:
        """
        识别手势。

        优先检测挥手（招手），其次检测举手。

        Args:
            keypoints: 姿态关键点数组 (17, 3) [x, y, confidence]
            track_id: 跟踪目标唯一ID

        Returns:
            GestureResult: 手势识别结果
        """
        if keypoints is None or len(keypoints) < 17:
            return GestureResult(gesture_type=GestureType.UNKNOWN, confidence=0.0)

        # 分别检测左右手
        best_result: Optional[GestureResult] = None

        for side in ["right", "left"]:
            # 优先检测挥手（招手打车的主要动作）
            is_wave, wave_conf, wrist_pos = self._detect_wave(
                keypoints, track_id, side
            )
            if is_wave and wave_conf > self.config.ai.gesture_conf_threshold:
                result = GestureResult(
                    gesture_type=GestureType.WAVE,
                    confidence=wave_conf,
                    wrist_pos=wrist_pos,
                    is_hailing=True,
                )
                if best_result is None or result.confidence > best_result.confidence:
                    best_result = result
                continue

            # 检测举手
            is_up, up_conf, wrist_pos = self._detect_hand_up(keypoints, side)
            if is_up and up_conf > self.config.ai.gesture_conf_threshold:
                result = GestureResult(
                    gesture_type=GestureType.HAND_UP,
                    confidence=up_conf,
                    wrist_pos=wrist_pos,
                    is_hailing=False,  # 举手但不是挥手
                )
                if best_result is None or result.confidence > best_result.confidence:
                    best_result = result

        # 返回最佳结果
        if best_result is not None:
            return best_result

        return GestureResult(gesture_type=GestureType.NONE, confidence=0.0)

    def reset(self) -> None:
        """重置所有历史记录。"""
        self._wrist_history.clear()
        logger.debug("手势识别器历史记录已重置")


# 全局手势识别器实例
_recognizer: Optional[GestureRecognizer] = None


def get_recognizer() -> GestureRecognizer:
    """
    获取全局手势识别器实例（单例模式）。

    Returns:
        GestureRecognizer: 手势识别器实例
    """
    global _recognizer
    if _recognizer is None:
        config = get_config()
        _recognizer = GestureRecognizer(
            history_frames=config.ai.gesture_history_frames
        )
    return _recognizer


def is_hailing_gesture(
    keypoints: np.ndarray, track_id: str = "default"
) -> Tuple[str, float]:
    """
    便捷函数：判断是否为招手打车手势。

    Args:
        keypoints: 姿态关键点数组 (17, 3) [x, y, confidence]
        track_id: 跟踪目标唯一ID

    Returns:
        (gesture_type, confidence)
        gesture_type: 'none' | 'hand_up' | 'wave'
        confidence: 0.0 - 1.0
    """
    recognizer = get_recognizer()
    result = recognizer.recognize(keypoints, track_id)
    return result.gesture_type.value, result.confidence
