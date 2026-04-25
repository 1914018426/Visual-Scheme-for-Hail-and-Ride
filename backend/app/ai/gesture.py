"""
手势识别模块

基于人体关键点 + MediaPipe Hands 手部关键点，精确识别"打招呼"与"打车"手势。

招手判定规则:
1. 打招呼 (greeting): 手掌面朝向画面，左右挥动，持续 2.5s+
2. 打车 (hailing): 手掌面朝向画面，手臂自然伸直或高举，手掌/手臂上下挥动，持续 2.5s+
3. 举手 (hand_up): 手臂伸直上举，但无持续周期性挥动

所有时间/占比/角度阈值均可通过环境变量全局配置。
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

    NONE = "none"          # 无手势
    GREETING = "greeting"  # 打招呼：手掌朝前 + 左右挥动
    HAILING = "hailing"    # 打车：手掌朝前 + 手臂伸直/高举 + 上下挥动
    HAND_UP = "hand_up"    # 举手（无持续挥动）
    UNKNOWN = "unknown"    # 未知


@dataclass
class GestureResult:
    """手势识别结果。"""

    gesture_type: GestureType = GestureType.NONE   # 手势类型
    confidence: float = 0.0                         # 置信度 (0-1)
    wrist_pos: Optional[Tuple[float, float]] = None  # 手腕位置 (x, y) 像素坐标
    is_hailing: bool = False                        # 是否为招手类意图（greeting/hailing）


class GestureRecognizer:
    """
    手势识别器

    基于人体姿态关键点 + MediaPipe Hands 手部关键点，识别打招呼/打车手势。
    使用基于时间戳的滑动窗口，确保动作持续 2.5s+，并检测周期性挥动。
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

    def __init__(self) -> None:
        """初始化手势识别器，读取全局配置。"""
        self.config = get_config()

        # 时间窗口参数
        self.min_duration_s = self.config.ai.gesture_min_duration_s
        self.palm_facing_ratio = self.config.ai.gesture_palm_facing_ratio
        self.arm_pose_ratio = self.config.ai.gesture_arm_pose_ratio
        self.motion_purity = self.config.ai.gesture_motion_purity
        self.min_cycles = self.config.ai.gesture_min_cycles
        self.cycle_max_period_s = self.config.ai.gesture_cycle_max_period_s
        self.straight_arm_angle = self.config.ai.gesture_straight_arm_angle
        self.conf_threshold = self.config.ai.gesture_conf_threshold

        # 若 MediaPipe Hands 未启用，自动关闭手掌朝向检测以允许回退
        if not self.config.ai.enable_hand_detection:
            self.palm_facing_ratio = 0.0
            logger.warning(
                "MediaPipe Hands 未启用，手掌朝向检测失效，手势识别将回退到纯姿态模式"
            )

        # 手腕位置与特征历史队列 {track_id_side: deque[(timestamp, x, y, palm_facing, arm_posed, is_raised), ...]}
        self._history: Dict[str, deque] = {}

    def _history_key(self, track_id: str, side: str) -> str:
        """生成历史队列的唯一键。"""
        return f"{track_id}_{side}"

    def _get_history(self, track_id: str, side: str) -> deque:
        """获取指定跟踪ID和侧别的历史队列。"""
        key = self._history_key(track_id, side)
        if key not in self._history:
            self._history[key] = deque()
        return self._history[key]

    def _clear_history(self, track_id: str) -> None:
        """清除指定跟踪ID的所有历史记录（左右手）。"""
        for side in ["left", "right"]:
            self._history.pop(self._history_key(track_id, side), None)

    def _prune_history(self, history: deque, now: float) -> None:
        """清理超过 1.5 倍最小持续时间的旧记录。"""
        cutoff = now - self.min_duration_s * 1.5
        while history and history[0][0] < cutoff:
            history.popleft()

    @staticmethod
    def _is_palm_facing_camera(
        hand_landmarks: List[Tuple[float, float, float]],
    ) -> Tuple[bool, float]:
        """
        判断手掌是否朝向摄像头（掌心朝前）。

        基于 MediaPipe Hands 21 点 landmark 综合判断：
        1. 手指张开程度（4指）
        2. 手指在 2D 平面上的扇形分布角度
        3. 手指关节深度一致性（保守判断）

        Args:
            hand_landmarks: 21 个 (x, y, z) 归一化或像素坐标

        Returns:
            (是否朝前, 置信度)
        """
        if not hand_landmarks or len(hand_landmarks) < 21:
            return False, 0.0

        pts = np.array(hand_landmarks)
        wrist = pts[0]

        # 四指: (tip, pip, mcp)
        fingers = [(8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)]

        # 1. 手指张开检测
        open_count = 0
        for tip_idx, pip_idx, mcp_idx in fingers:
            tip = pts[tip_idx]
            mcp = pts[mcp_idx]
            tip_dist = np.linalg.norm(tip[:2] - wrist[:2])
            mcp_dist = np.linalg.norm(mcp[:2] - wrist[:2])
            if tip_dist > mcp_dist * 1.15:
                open_count += 1

        # 2. 扇形分布角度
        angles = []
        for tip_idx, _, mcp_idx in fingers:
            vec = pts[tip_idx][:2] - pts[mcp_idx][:2]
            angle = np.arctan2(vec[1], vec[0])
            angles.append(angle)

        angles = np.sort(np.array(angles))
        angle_span = float(angles[-1] - angles[0])
        if angle_span > np.pi:
            angle_span = 2 * np.pi - angle_span

        # 3. 深度一致性（保守：指尖不比指根靠后太多）
        depth_ok = 0
        for tip_idx, _, mcp_idx in fingers:
            if pts[tip_idx][2] < pts[mcp_idx][2] + 0.15:
                depth_ok += 1

        is_facing = (
            open_count >= 3
            and angle_span > np.radians(50)
            and depth_ok >= 2
        )

        confidence = (
            (open_count / 4.0) * 0.5
            + min(1.0, angle_span / np.radians(100)) * 0.3
            + (depth_ok / 4.0) * 0.2
        )
        return is_facing, confidence

    def _detect_arm_pose(
        self, keypoints: np.ndarray, side: str = "right"
    ) -> Tuple[bool, bool, float]:
        """
        检测手臂姿势：自然伸直 / 高举 / 伸出。

        Args:
            keypoints: 姿态关键点数组 (17, 3)
            side: 'left' 或 'right'

        Returns:
            (is_posed, is_raised, confidence)
            is_posed: 自然伸直 or 高举 or 伸出
            is_raised: 手腕明显高于肩膀
        """
        if side == "left":
            s_idx = self.KEYPOINT_LEFT_SHOULDER
            e_idx = self.KEYPOINT_LEFT_ELBOW
            w_idx = self.KEYPOINT_LEFT_WRIST
            hip_idx = self.KEYPOINT_LEFT_HIP
            opp_s_idx = self.KEYPOINT_RIGHT_SHOULDER
        else:
            s_idx = self.KEYPOINT_RIGHT_SHOULDER
            e_idx = self.KEYPOINT_RIGHT_ELBOW
            w_idx = self.KEYPOINT_RIGHT_WRIST
            hip_idx = self.KEYPOINT_RIGHT_HIP
            opp_s_idx = self.KEYPOINT_LEFT_SHOULDER

        shoulder = keypoints[s_idx]
        elbow = keypoints[e_idx]
        wrist = keypoints[w_idx]
        hip = keypoints[hip_idx]
        opp_shoulder = keypoints[opp_s_idx]

        min_conf = 0.3
        if (
            shoulder[2] < min_conf
            or elbow[2] < min_conf
            or wrist[2] < min_conf
        ):
            return False, False, 0.0

        # 肩宽（用于归一化）
        if shoulder[2] > min_conf and opp_shoulder[2] > min_conf:
            shoulder_width = abs(float(opp_shoulder[0]) - float(shoulder[0]))
        else:
            shoulder_width = 50.0
        if shoulder_width < 8.0:
            shoulder_width = 50.0

        # 1. 自然伸直：shoulder-elbow-wrist 夹角
        def _angle(a, b, c):
            ba = np.array(a[:2]) - np.array(b[:2])
            bc = np.array(c[:2]) - np.array(b[:2])
            norm_ba = np.linalg.norm(ba)
            norm_bc = np.linalg.norm(bc)
            if norm_ba < 1e-6 or norm_bc < 1e-6:
                return 180.0
            cos_ang = np.dot(ba, bc) / (norm_ba * norm_bc)
            return float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))

        arm_angle = _angle(shoulder, elbow, wrist)
        is_straight = arm_angle > self.straight_arm_angle

        # 2. 高举：手腕在肩膀上方
        shoulder_y = float(shoulder[1])
        wrist_y = float(wrist[1])
        torso_h = (
            abs(float(hip[1]) - shoulder_y)
            if hip[2] > min_conf
            else shoulder_width * 1.2
        )
        is_raised = wrist_y < shoulder_y - torso_h * 0.15

        # 3. 伸出：手腕到肩膀距离较大
        arm_length = float(
            np.linalg.norm(np.array(wrist[:2]) - np.array(shoulder[:2]))
        )
        is_extended = arm_length > shoulder_width * 0.8

        is_posed = is_straight or is_raised or is_extended

        # 置信度
        straight_score = min(
            1.0, max(0.0, (arm_angle - self.straight_arm_angle + 30) / 60)
        )
        raised_score = 1.0 if is_raised else 0.0
        extended_score = min(1.0, arm_length / (shoulder_width * 1.5 + 1e-6))
        confidence = max(straight_score, raised_score * 0.9, extended_score * 0.7)

        return is_posed, is_raised, confidence

    def _analyze_motion(
        self, history: List[Tuple]
    ) -> Optional[Dict[str, Any]]:
        """
        分析时间窗口内的运动方向与周期。

        Args:
            history: [(timestamp, wrist_x, wrist_y, palm_facing, arm_posed, is_raised), ...]

        Returns:
            运动分析字典，包含 direction/x_diff/y_diff/reversals/purity/cycles 等
        """
        if len(history) < 4:
            return None

        times = [h[0] for h in history]
        xs = [h[1] for h in history]
        ys = [h[2] for h in history]

        duration = times[-1] - times[0]
        if duration < self.min_duration_s:
            return None

        x_diff = max(xs) - min(xs)
        y_diff = max(ys) - min(ys)
        total_diff = x_diff + y_diff + 1e-6

        # 方向反转计数（忽略微小移动）
        def _count_reversals(values):
            if len(values) < 3:
                return 0
            reversals = 0
            prev_sign = 0
            for i in range(1, len(values)):
                dv = values[i] - values[i - 1]
                if abs(dv) < 1.0:
                    continue
                sign = 1 if dv > 0 else -1
                if prev_sign != 0 and sign != prev_sign:
                    reversals += 1
                prev_sign = sign
            return reversals

        x_reversals = _count_reversals(xs)
        y_reversals = _count_reversals(ys)

        # 周期检测：计算相邻反转的时间间隔
        def _cycle_periods(ts, values):
            if len(values) < 3:
                return []
            periods = []
            prev_sign = 0
            last_rev_t = None
            for i in range(1, len(values)):
                dv = values[i] - values[i - 1]
                if abs(dv) < 1.0:
                    continue
                sign = 1 if dv > 0 else -1
                if prev_sign != 0 and sign != prev_sign:
                    if last_rev_t is not None:
                        periods.append(ts[i] - last_rev_t)
                    last_rev_t = ts[i]
                prev_sign = sign
            return periods

        x_periods = _cycle_periods(times, xs)
        y_periods = _cycle_periods(times, ys)

        # 判定主方向
        if x_diff > y_diff:
            main_periods = x_periods
            main_reversals = x_reversals
            direction = "horizontal"
            purity = x_diff / total_diff
        else:
            main_periods = y_periods
            main_reversals = y_reversals
            direction = "vertical"
            purity = y_diff / total_diff

        avg_cycle_period = (
            float(np.mean(main_periods)) if main_periods else 0.0
        )

        # 过滤过慢周期，统计完整周期数（2 次反转 = 1 个周期）
        valid_cycles = sum(
            1 for p in main_periods if p <= self.cycle_max_period_s
        )
        full_cycles = valid_cycles // 2

        return {
            "direction": direction,
            "x_diff": x_diff,
            "y_diff": y_diff,
            "x_reversals": x_reversals,
            "y_reversals": y_reversals,
            "avg_cycle_period_s": avg_cycle_period,
            "purity": purity,
            "full_cycles": full_cycles,
            "main_reversals": main_reversals,
        }

    def recognize(
        self,
        keypoints: np.ndarray,
        track_id: str = "default",
        left_hand_landmarks: Optional[List[Tuple[float, float, float]]] = None,
        right_hand_landmarks: Optional[List[Tuple[float, float, float]]] = None,
        frame_timestamp: Optional[float] = None,
    ) -> GestureResult:
        """
        识别手势。

        优先检测打车(hailing)，其次打招呼(greeting)，最后举手(hand_up)。
        左右手分别检测，取置信度最高者。

        Args:
            keypoints: 姿态关键点数组 (17, 3) [x, y, confidence]
            track_id: 跟踪目标唯一ID
            left_hand_landmarks: 左手 MediaPipe 21 点 (x,y,z)，可选
            right_hand_landmarks: 右手 MediaPipe 21 点 (x,y,z)，可选
            frame_timestamp: 当前帧时间戳（秒），未提供则使用 time.time()

        Returns:
            GestureResult: 手势识别结果
        """
        if keypoints is None or len(keypoints) < 17:
            return GestureResult(
                gesture_type=GestureType.UNKNOWN, confidence=0.0
            )

        now = frame_timestamp if frame_timestamp is not None else time.time()
        best_result: Optional[GestureResult] = None

        for side in ["right", "left"]:
            # 1. 手臂姿势检测
            is_posed, is_raised, arm_conf = self._detect_arm_pose(
                keypoints, side
            )

            if side == "left":
                wrist_idx = self.KEYPOINT_LEFT_WRIST
                shoulder_idx = self.KEYPOINT_LEFT_SHOULDER
                hand_landmarks = left_hand_landmarks
            else:
                wrist_idx = self.KEYPOINT_RIGHT_WRIST
                shoulder_idx = self.KEYPOINT_RIGHT_SHOULDER
                hand_landmarks = right_hand_landmarks

            wrist = keypoints[wrist_idx]
            shoulder = keypoints[shoulder_idx]

            if wrist[2] < 0.3 or shoulder[2] < 0.3:
                continue

            wrist_pos = (float(wrist[0]), float(wrist[1]))

            # 2. 手掌朝向检测
            palm_facing = False
            palm_conf = 0.0
            if hand_landmarks and len(hand_landmarks) >= 21:
                palm_facing, palm_conf = self._is_palm_facing_camera(
                    hand_landmarks
                )

            # 3. 记录历史（按 track_id + side 独立存储）
            history = self._get_history(track_id, side)
            history.append(
                (now, wrist_pos[0], wrist_pos[1], palm_facing, is_posed, is_raised)
            )
            self._prune_history(history, now)

            if len(history) < 4:
                continue

            # 4. 统计时间窗口
            palm_count = sum(1 for h in history if h[3])
            posed_count = sum(1 for h in history if h[4])
            total = len(history)
            duration = history[-1][0] - history[0][0]

            palm_ratio = palm_count / total if total > 0 else 0.0
            pose_ratio = posed_count / total if total > 0 else 0.0

            # 时间不足时，若手臂姿势良好可临时返回 HAND_UP（即时反馈）
            if duration < self.min_duration_s:
                if (
                    is_posed
                    and arm_conf > self.conf_threshold
                    and pose_ratio >= self.arm_pose_ratio
                ):
                    result = GestureResult(
                        gesture_type=GestureType.HAND_UP,
                        confidence=min(arm_conf * 0.7, 1.0),
                        wrist_pos=wrist_pos,
                        is_hailing=False,
                    )
                    if best_result is None or result.confidence > best_result.confidence:
                        best_result = result
                continue

            # 手掌朝向占比不足 → 无法判定为 greeting/hailing
            if palm_ratio < self.palm_facing_ratio:
                continue

            # 手臂姿势占比不足
            if pose_ratio < self.arm_pose_ratio:
                continue

            # 5. 运动分析
            motion = self._analyze_motion(list(history))
            if motion is None:
                continue

            # 运动纯度不足
            if motion["purity"] < self.motion_purity:
                continue

            # 周期数不足 → 归为 HAND_UP
            if motion["full_cycles"] < self.min_cycles:
                conf = min(1.0, (palm_ratio + pose_ratio) * 0.5)
                result = GestureResult(
                    gesture_type=GestureType.HAND_UP,
                    confidence=conf,
                    wrist_pos=wrist_pos,
                    is_hailing=False,
                )
                if best_result is None or result.confidence > best_result.confidence:
                    best_result = result
                continue

            # 6. 判定手势类型
            if motion["direction"] == "horizontal":
                # 打招呼：左右挥动
                conf = min(
                    1.0,
                    0.5
                    + motion["purity"] * 0.3
                    + palm_ratio * 0.2,
                )
                result = GestureResult(
                    gesture_type=GestureType.GREETING,
                    confidence=conf,
                    wrist_pos=wrist_pos,
                    is_hailing=True,
                )
            else:
                # 打车：上下挥动（需确认手臂姿势）
                conf = min(
                    1.0,
                    0.55
                    + motion["purity"] * 0.25
                    + palm_ratio * 0.2
                    + pose_ratio * 0.15,
                )
                result = GestureResult(
                    gesture_type=GestureType.HAILING,
                    confidence=conf,
                    wrist_pos=wrist_pos,
                    is_hailing=True,
                )

            if best_result is None or result.confidence > best_result.confidence:
                best_result = result

        if best_result is not None:
            return best_result

        return GestureResult(gesture_type=GestureType.NONE, confidence=0.0)

    def reset(self) -> None:
        """重置所有历史记录。"""
        self._history.clear()
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
        _recognizer = GestureRecognizer()
    return _recognizer


def is_hailing_gesture(
    keypoints: np.ndarray,
    track_id: str = "default",
    left_hand_landmarks: Optional[List[Tuple[float, float, float]]] = None,
    right_hand_landmarks: Optional[List[Tuple[float, float, float]]] = None,
    frame_timestamp: Optional[float] = None,
) -> Tuple[str, float]:
    """
    便捷函数：判断是否为招手打车手势。

    Args:
        keypoints: 姿态关键点数组 (17, 3) [x, y, confidence]
        track_id: 跟踪目标唯一ID
        left_hand_landmarks: 左手 MediaPipe 21 点，可选
        right_hand_landmarks: 右手 MediaPipe 21 点，可选
        frame_timestamp: 当前帧时间戳（秒），可选

    Returns:
        (gesture_type, confidence)
        gesture_type: 'none' | 'greeting' | 'hailing' | 'hand_up'
        confidence: 0.0 - 1.0
    """
    recognizer = get_recognizer()
    result = recognizer.recognize(
        keypoints,
        track_id,
        left_hand_landmarks,
        right_hand_landmarks,
        frame_timestamp,
    )
    return result.gesture_type.value, result.confidence
