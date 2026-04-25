"""
手势识别模块 —— 基于速度向量的帧级状态机

核心设计：
1. 每帧计算 wrist 速度向量，用速度而非位置作为动态特征
2. 帧级状态机：IDLE → HAND_UP → WAVING → CONFIRMED → IDLE
3. 响应延迟 5-8 帧（约 0.3-0.5s @ 15fps），远优于原 2.5s 时间窗口
4. 意图语义：hailing = 高举 + 垂直挥动；greeting = 水平挥动（不限高度）

所有阈值均可通过环境变量配置。
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

    NONE = "none"
    GREETING = "greeting"  # 打招呼：水平方向挥动
    HAILING = "hailing"    # 打车：高举 + 垂直方向挥动（向下招手）
    HAND_UP = "hand_up"    # 举手：手臂举起但无挥动


@dataclass
class GestureResult:
    """手势识别结果。"""

    gesture_type: GestureType = GestureType.NONE
    confidence: float = 0.0
    wrist_pos: Optional[Tuple[float, float]] = None


@dataclass
class SideStateMachine:
    """
    单侧手臂状态机（每 track_id + side 一个实例）。

    状态流转：
        IDLE → HAND_UP → WAVING → CONFIRMED → IDLE
    """

    state: str = "idle"                 # idle / hand_up / waving / confirmed
    frames_in_state: int = 0             # 在当前状态的累计帧数
    consecutive_wave_frames: int = 0     # 连续挥动帧数
    stop_frames: int = 0                 # 连续停止帧数
    last_wrist_pos: Optional[Tuple[float, float]] = None
    last_timestamp: Optional[float] = None
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=20))
    direction_history: deque = field(default_factory=lambda: deque(maxlen=20))
    main_direction: str = "none"         # horizontal / vertical / none
    confirmed_gesture: Optional[str] = None
    peak_confidence: float = 0.0


class GestureRecognizer:
    """
    帧级手势识别器。

    基于速度向量 + 帧级状态机，每帧实时更新意图状态。
    """

    # COCO 姿态关键点索引
    NOSE = 0
    L_SHOULDER = 5
    R_SHOULDER = 6
    L_ELBOW = 7
    R_ELBOW = 8
    L_WRIST = 9
    R_WRIST = 10
    L_HIP = 11
    R_HIP = 12

    def __init__(self) -> None:
        self.config = get_config()
        c = self.config.ai

        # 手臂姿势阈值
        self.straight_arm_angle = c.gesture_straight_arm_angle

        # 速度阈值（像素/秒），低于此视为静止
        self.velocity_threshold = getattr(c, "gesture_velocity_threshold", 80.0)
        # 进入 WAVING 所需的最小速度反转次数（1 帧内）
        self.waving_trigger_reversals = getattr(c, "gesture_waving_trigger_reversals", 1)
        # 连续挥动 N 帧后确认意图
        self.confirm_frames = getattr(c, "gesture_confirm_frames", 5)
        # 停止挥动 M 帧后重置
        self.stop_reset_frames = getattr(c, "gesture_stop_reset_frames", 8)
        # 最小挥动幅度（相对肩宽倍数）
        self.min_amplitude = getattr(c, "gesture_min_amplitude", 0.25)
        # 手掌朝向判定：0=禁用  其他=所需手掌朝前帧占比（但状态机里每帧独立判断）
        self.palm_facing_required = getattr(c, "gesture_palm_facing_required", 0)
        # hailing 判定：手腕必须高于肩膀下方 torso_h * ratio
        self.hailing_min_height_ratio = getattr(c, "gesture_hailing_min_height_ratio", 0.3)

        # 每 track_id_side 一个状态机
        self._machines: Dict[str, SideStateMachine] = {}

        logger.info(
            "GestureRecognizer(状态机): vel_thresh=%.1f confirm_frames=%d "
            "stop_reset=%d min_amp=%.2f",
            self.velocity_threshold,
            self.confirm_frames,
            self.stop_reset_frames,
            self.min_amplitude,
        )

    # ------------------------------------------------------------------ #
    # 工具方法
    # ------------------------------------------------------------------ #

    def _machine_key(self, track_id: str, side: str) -> str:
        return f"{track_id}_{side}"

    def _get_machine(self, track_id: str, side: str) -> SideStateMachine:
        key = self._machine_key(track_id, side)
        if key not in self._machines:
            self._machines[key] = SideStateMachine()
        return self._machines[key]

    def _clear_machine(self, track_id: str) -> None:
        for side in ["left", "right"]:
            self._machines.pop(self._machine_key(track_id, side), None)

    @staticmethod
    def _keypoint_conf(kp) -> float:
        return float(kp[2]) if len(kp) > 2 else 1.0

    @staticmethod
    def _angle_3pt(a, b, c) -> float:
        """计算三点夹角（度）。"""
        ba = np.array(a[:2]) - np.array(b[:2])
        bc = np.array(c[:2]) - np.array(b[:2])
        n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
        if n1 < 1e-6 or n2 < 1e-6:
            return 180.0
        cos_ang = np.dot(ba, bc) / (n1 * n2)
        return float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))

    # ------------------------------------------------------------------ #
    # 手掌朝向检测（改进版）
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_palm_facing_camera(
        hand_landmarks: List[Tuple[float, float, float]],
    ) -> Tuple[bool, float]:
        """
        判断手掌是否朝向摄像头（掌心朝前）。
        改进：降低张开手指要求，增加手指指向判断。
        """
        if not hand_landmarks or len(hand_landmarks) < 21:
            return False, 0.0

        pts = np.array(hand_landmarks)
        wrist = pts[0]

        # 四指: (tip, pip, mcp)
        fingers = [(8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)]

        open_count = 0
        for tip_idx, pip_idx, mcp_idx in fingers:
            tip = pts[tip_idx]
            mcp = pts[mcp_idx]
            if np.linalg.norm(tip[:2] - wrist[:2]) > np.linalg.norm(mcp[:2] - wrist[:2]) * 1.1:
                open_count += 1

        # 扇形分布角度
        angles = []
        for tip_idx, _, mcp_idx in fingers:
            vec = pts[tip_idx][:2] - pts[mcp_idx][:2]
            angles.append(np.arctan2(vec[1], vec[0]))
        angles = np.sort(np.array(angles))
        angle_span = float(angles[-1] - angles[0])
        if angle_span > np.pi:
            angle_span = 2 * np.pi - angle_span

        # 深度一致性（指尖不比指根靠后太多）
        depth_ok = sum(
            1 for tip_idx, _, mcp_idx in fingers
            if pts[tip_idx][2] < pts[mcp_idx][2] + 0.2
        )

        # 手指指向判断：手指整体是否指向画面外（z < 0 表示朝前）
        finger_tips_z = [pts[i][2] for i in [8, 12, 16, 20]]
        pointing_out = sum(1 for z in finger_tips_z if z < 0.05) >= 2

        is_facing = (
            open_count >= 2
            and (angle_span > np.radians(35) or pointing_out)
            and depth_ok >= 1
        )

        confidence = (
            (open_count / 4.0) * 0.4
            + min(1.0, angle_span / np.radians(100)) * 0.3
            + (depth_ok / 4.0) * 0.15
            + (0.25 if pointing_out else 0.0)
        )
        return is_facing, confidence

    # ------------------------------------------------------------------ #
    # 手臂姿势检测
    # ------------------------------------------------------------------ #

    def _detect_arm_pose(
        self, keypoints: np.ndarray, side: str
    ) -> Tuple[bool, bool, float]:
        """
        检测手臂姿势。
        Returns: (is_posed, is_raised, confidence)
        """
        if side == "left":
            s_idx, e_idx, w_idx = self.L_SHOULDER, self.L_ELBOW, self.L_WRIST
        else:
            s_idx, e_idx, w_idx = self.R_SHOULDER, self.R_ELBOW, self.R_WRIST

        shoulder, elbow, wrist = keypoints[s_idx], keypoints[e_idx], keypoints[w_idx]
        min_conf = 0.3
        if shoulder[2] < min_conf or elbow[2] < min_conf or wrist[2] < min_conf:
            return False, False, 0.0

        # 肩宽归一化
        opp_s_idx = self.R_SHOULDER if side == "left" else self.L_SHOULDER
        opp_shoulder = keypoints[opp_s_idx]
        shoulder_width = (
            abs(float(opp_shoulder[0]) - float(shoulder[0]))
            if opp_shoulder[2] > min_conf else 50.0
        )
        if shoulder_width < 8.0:
            shoulder_width = 50.0

        # 1. 自然伸直
        arm_angle = self._angle_3pt(shoulder, elbow, wrist)
        is_straight = arm_angle > self.straight_arm_angle

        # 2. 高举
        shoulder_y, wrist_y = float(shoulder[1]), float(wrist[1])
        is_raised = wrist_y < shoulder_y + shoulder_width * 0.1

        # 3. 伸出
        arm_length = float(np.linalg.norm(np.array(wrist[:2]) - np.array(shoulder[:2])))
        is_extended = arm_length > shoulder_width * 0.6

        is_posed = is_straight or is_raised or is_extended

        straight_score = min(1.0, max(0.0, (arm_angle - self.straight_arm_angle + 30) / 60))
        raised_score = 1.0 if is_raised else 0.0
        extended_score = min(1.0, arm_length / (shoulder_width * 1.5 + 1e-6))
        confidence = max(straight_score, raised_score * 0.9, extended_score * 0.7)

        return is_posed, is_raised, confidence

    # ------------------------------------------------------------------ #
    # 速度计算
    # ------------------------------------------------------------------ #

    def _compute_velocity(
        self,
        machine: SideStateMachine,
        wrist_pos: Tuple[float, float],
        timestamp: float,
    ) -> Tuple[float, float, float]:
        """
        计算 wrist 速度。
        Returns: (vx, vy, magnitude)
        """
        if machine.last_wrist_pos is None or machine.last_timestamp is None:
            machine.last_wrist_pos = wrist_pos
            machine.last_timestamp = timestamp
            return 0.0, 0.0, 0.0

        dt = timestamp - machine.last_timestamp
        if dt < 1e-6:
            return 0.0, 0.0, 0.0

        vx = (wrist_pos[0] - machine.last_wrist_pos[0]) / dt
        vy = (wrist_pos[1] - machine.last_wrist_pos[1]) / dt
        mag = np.hypot(vx, vy)

        machine.last_wrist_pos = wrist_pos
        machine.last_timestamp = timestamp
        return vx, vy, mag

    # ------------------------------------------------------------------ #
    # 主识别方法
    # ------------------------------------------------------------------ #

    def recognize(
        self,
        keypoints: np.ndarray,
        track_id: str = "default",
        left_hand_landmarks: Optional[List[Tuple[float, float, float]]] = None,
        right_hand_landmarks: Optional[List[Tuple[float, float, float]]] = None,
        frame_timestamp: Optional[float] = None,
    ) -> GestureResult:
        """
        帧级手势识别。
        左右手分别运行状态机，取置信度最高者。
        """
        if keypoints is None or len(keypoints) < 17:
            return GestureResult(gesture_type=GestureType.NONE, confidence=0.0)

        now = frame_timestamp if frame_timestamp is not None else time.time()
        best_result: Optional[GestureResult] = None

        for side in ["right", "left"]:
            result = self._recognize_side(
                keypoints, side, track_id,
                left_hand_landmarks if side == "left" else None,
                right_hand_landmarks if side == "right" else None,
                now,
            )
            if result and (best_result is None or result.confidence > best_result.confidence):
                best_result = result

        return best_result if best_result else GestureResult(
            gesture_type=GestureType.NONE, confidence=0.0
        )

    def _recognize_side(
        self,
        keypoints: np.ndarray,
        side: str,
        track_id: str,
        hand_landmarks: Optional[List[Tuple[float, float, float]]],
        timestamp: float,
    ) -> Optional[GestureResult]:
        """单侧状态机处理一帧。"""
        if side == "left":
            w_idx, s_idx = self.L_WRIST, self.L_SHOULDER
        else:
            w_idx, s_idx = self.R_WRIST, self.R_SHOULDER

        wrist = keypoints[w_idx]
        shoulder = keypoints[s_idx]

        # 关键点不可信时重置状态机
        if wrist[2] < 0.3 or shoulder[2] < 0.3:
            self._clear_machine(track_id)
            return None

        wrist_pos = (float(wrist[0]), float(wrist[1]))
        shoulder_pos = (float(shoulder[0]), float(shoulder[1]))

        # 1. 手臂姿势
        is_posed, is_raised, arm_conf = self._detect_arm_pose(keypoints, side)

        # 2. 手掌朝向（每帧独立判断，不依赖历史）
        palm_facing = False
        palm_conf = 0.0
        if hand_landmarks and len(hand_landmarks) >= 21:
            palm_facing, palm_conf = self._is_palm_facing_camera(hand_landmarks)

        # 3. 速度计算
        machine = self._get_machine(track_id, side)
        vx, vy, v_mag = self._compute_velocity(machine, wrist_pos, timestamp)

        # 记录速度和方向
        machine.velocity_history.append((vx, vy, v_mag))
        if v_mag > self.velocity_threshold:
            direction = "horizontal" if abs(vx) > abs(vy) else "vertical"
            machine.direction_history.append(direction)
        else:
            machine.direction_history.append("none")

        # 4. 状态机流转
        gesture, confidence = self._state_transition(
            machine, is_posed, is_raised, arm_conf,
            palm_facing, palm_conf, v_mag,
            wrist_pos, shoulder_pos,
        )

        if gesture != "none":
            logger.info(
                "gesture[%s/%s]: %s conf=%.2f state=%s frames=%d "
                "v=%.1f raised=%s palm=%s",
                track_id, side, gesture, confidence,
                machine.state, machine.frames_in_state,
                v_mag, is_raised, palm_facing,
            )

        if gesture == "greeting":
            return GestureResult(GestureType.GREETING, confidence, wrist_pos)
        elif gesture == "hailing":
            return GestureResult(GestureType.HAILING, confidence, wrist_pos)
        elif gesture == "hand_up":
            return GestureResult(GestureType.HAND_UP, confidence, wrist_pos)
        return None

    # ------------------------------------------------------------------ #
    # 状态机流转逻辑
    # ------------------------------------------------------------------ #

    def _state_transition(
        self,
        machine: SideStateMachine,
        is_posed: bool,
        is_raised: bool,
        arm_conf: float,
        palm_facing: bool,
        palm_conf: float,
        v_mag: float,
        wrist_pos: Tuple[float, float],
        shoulder_pos: Tuple[float, float],
    ) -> Tuple[str, float]:
        """
        状态机核心逻辑。
        Returns: (gesture_string, confidence)
        """
        state = machine.state
        machine.frames_in_state += 1

        # 手臂放下 → 重置
        if not is_posed:
            machine.state = "idle"
            machine.frames_in_state = 0
            machine.consecutive_wave_frames = 0
            machine.stop_frames = 0
            machine.confirmed_gesture = None
            return "none", 0.0

        # 速度显著？
        is_waving = v_mag > self.velocity_threshold

        if state == "idle":
            if is_posed:
                machine.state = "hand_up"
                machine.frames_in_state = 1
                return "hand_up", min(arm_conf * 0.7, 1.0)
            return "none", 0.0

        if state == "hand_up":
            if is_waving:
                machine.state = "waving"
                machine.frames_in_state = 1
                machine.consecutive_wave_frames = 1
                machine.stop_frames = 0
            return "hand_up", min(arm_conf * 0.7, 1.0)

        if state == "waving":
            if is_waving:
                machine.consecutive_wave_frames += 1
                machine.stop_frames = 0
                # 连续挥动 N 帧 → 确认意图
                if machine.consecutive_wave_frames >= self.confirm_frames:
                    machine.state = "confirmed"
                    machine.frames_in_state = 1
                    gesture, conf = self._classify_intent(
                        machine, is_raised, palm_facing, arm_conf, palm_conf,
                        wrist_pos, shoulder_pos,
                    )
                    machine.confirmed_gesture = gesture
                    machine.peak_confidence = conf
                    return gesture, conf
            else:
                machine.stop_frames += 1
                if machine.stop_frames >= self.stop_reset_frames:
                    machine.state = "idle"
                    machine.frames_in_state = 0
                    machine.consecutive_wave_frames = 0
                    machine.stop_frames = 0
                    return "none", 0.0
            return "hand_up", min(arm_conf * 0.7, 1.0)

        if state == "confirmed":
            if is_waving:
                machine.stop_frames = 0
                # 维持确认的手势，置信度可逐渐衰减
                decay = max(0.5, 1.0 - machine.frames_in_state * 0.02)
                conf = machine.peak_confidence * decay
                machine.frames_in_state += 1
                if machine.confirmed_gesture:
                    return machine.confirmed_gesture, conf
            else:
                machine.stop_frames += 1
                if machine.stop_frames >= self.stop_reset_frames:
                    machine.state = "idle"
                    machine.frames_in_state = 0
                    machine.consecutive_wave_frames = 0
                    machine.stop_frames = 0
                    machine.confirmed_gesture = None
                    return "none", 0.0
                # 短暂静止时维持低置信度输出
                if machine.confirmed_gesture:
                    return machine.confirmed_gesture, machine.peak_confidence * 0.5
            return "none", 0.0

        return "none", 0.0

    # ------------------------------------------------------------------ #
    # 意图分类
    # ------------------------------------------------------------------ #

    def _classify_intent(
        self,
        machine: SideStateMachine,
        is_raised: bool,
        palm_facing: bool,
        arm_conf: float,
        palm_conf: float,
        wrist_pos: Tuple[float, float],
        shoulder_pos: Tuple[float, float],
    ) -> Tuple[str, float]:
        """
        根据挥动方向、手臂高度、手掌朝向综合判定 greeting vs hailing。
        """
        # 统计方向历史中的主方向
        dirs = [d for d in machine.direction_history if d != "none"]
        if not dirs:
            return "hand_up", min(arm_conf * 0.7, 1.0)

        h_count = sum(1 for d in dirs if d == "horizontal")
        v_count = sum(1 for d in dirs if d == "vertical")
        total = h_count + v_count
        if total == 0:
            return "hand_up", min(arm_conf * 0.7, 1.0)

        h_ratio = h_count / total
        v_ratio = v_count / total

        # 手腕相对肩膀的高度（像素，正数表示在肩膀下方）
        wrist_below_shoulder = wrist_pos[1] - shoulder_pos[1]

        # hailing: 垂直挥动为主 + 手臂高举（手腕在肩膀附近或上方）
        if v_ratio > 0.55 and is_raised and wrist_below_shoulder < shoulder_pos[1] * self.hailing_min_height_ratio:
            conf = min(1.0, 0.5 + v_ratio * 0.3 + arm_conf * 0.2)
            if palm_facing:
                conf = min(1.0, conf + 0.15)
            return "hailing", conf

        # greeting: 水平挥动为主 或 垂直挥动但手臂不高举
        if h_ratio > 0.55 or (v_ratio > 0.55 and not is_raised):
            conf = min(1.0, 0.5 + max(h_ratio, v_ratio) * 0.3 + arm_conf * 0.2)
            if palm_facing:
                conf = min(1.0, conf + 0.1)
            return "greeting", conf

        # 方向不明确， fallback 到 hand_up
        return "hand_up", min(arm_conf * 0.7, 1.0)

    def reset(self) -> None:
        """重置所有状态机。"""
        self._machines.clear()
        logger.info("手势识别器已重置")


# =====================================================================
# 全局单例 + 便捷函数
# =====================================================================

_recognizer: Optional[GestureRecognizer] = None


def get_recognizer() -> GestureRecognizer:
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
    """便捷函数。"""
    recognizer = get_recognizer()
    result = recognizer.recognize(
        keypoints, track_id,
        left_hand_landmarks, right_hand_landmarks,
        frame_timestamp,
    )
    return result.gesture_type.value, result.confidence
