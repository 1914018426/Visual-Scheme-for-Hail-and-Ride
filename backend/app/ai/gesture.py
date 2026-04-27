"""
手势识别模块 —— 三锁合取机制 (Triple-Lock Conjunction)

删除 idle → posed → oscillating → confirmed 状态机，改为：
  姿态锁 + 朝向锁 + 运动锁 同时满足 → confirmed_hailing

所有轨迹、速度、周期性判断基于 Torso-Normalized Local Frame (TNLF)。
禁止直接使用画面像素坐标。
"""

import logging
import time
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from app.config import get_config
from app.ai.local_frame import wrist_to_local_frame, local_velocity
from app.ai.facing import facing_gate
from app.ai.slerp import (
    compute_palm_normal,
    NormalSmoother,
    angle_to_camera_z,
)
from app.ai.iri import IRICalculator

logger = logging.getLogger(__name__)


# =============================================================================
# 周期性运动检测引擎（基于 wrist_local，单位 torso_units）
# =============================================================================

class PeriodicMotionDetector:
    """
    基于 wrist_local 序列的周期性运动检测器。

    使用 zero-crossing + 自相关函数(ACF) 检测稳定的周期性运动。
    人类挥手/招手的典型频率：0.5-3 Hz。
    """

    def __init__(
        self,
        buffer_seconds: float = 2.0,
        fps: float = 15.0,
        min_freq_hz: float = 0.5,
        max_freq_hz: float = 3.0,
        min_cycles: int = 2,
        max_cycle_variation: float = 0.35,
    ) -> None:
        self.fps = fps
        self.min_freq_hz = min_freq_hz
        self.max_freq_hz = max_freq_hz
        self.min_cycles = min_cycles
        self.max_cycle_variation = max_cycle_variation
        self._maxlen = int(buffer_seconds * fps)
        self._xs: deque = deque(maxlen=self._maxlen)
        self._ys: deque = deque(maxlen=self._maxlen)
        self._last_result: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        self._xs.clear()
        self._ys.clear()
        self._last_result = None

    def feed(self, wrist_local: Tuple[float, float]) -> None:
        """喂入新一帧的 wrist_local（躯干归一化坐标）。"""
        self._xs.append(float(wrist_local[0]))
        self._ys.append(float(wrist_local[1]))

    def detect(self) -> Optional[Dict[str, Any]]:
        """
        检测周期性运动。

        Returns:
            dict with keys:
                - is_periodic: bool
                - dominant_axis: "x" | "y" | "none"
                - frequency_hz: float
                - amplitude_tu: float   # 振幅（已是 torso_units）
                - cycle_count: int
                - consistency: float    # 0-1，周期一致性
                - zero_crossings: int
        """
        if len(self._xs) < self._maxlen * 0.6:
            return None

        x_result = self._analyze_axis(np.array(self._xs))
        y_result = self._analyze_axis(np.array(self._ys))

        if x_result is None and y_result is None:
            self._last_result = None
            return None

        if x_result is None:
            best = y_result
            best["dominant_axis"] = "y"
        elif y_result is None:
            best = x_result
            best["dominant_axis"] = "x"
        elif x_result.get("consistency", 0) >= y_result.get("consistency", 0):
            best = x_result
            best["dominant_axis"] = "x"
        else:
            best = y_result
            best["dominant_axis"] = "y"

        self._last_result = best
        return best

    def _analyze_axis(self, series: np.ndarray) -> Optional[Dict[str, Any]]:
        """分析单轴序列的周期性。"""
        if len(series) < 10:
            return None

        # 1. 去趋势
        x = np.arange(len(series))
        slope, intercept = np.polyfit(x, series, 1)
        detrended = series - (slope * x + intercept)

        # 2. Zero-crossing 检测
        zero_crossings = self._count_zero_crossings(detrended)
        if zero_crossings < self.min_cycles * 2:
            return None

        # 3. 从 zero-crossing 估计频率
        duration_sec = len(series) / self.fps
        freq_zc = zero_crossings / (2 * duration_sec)
        if not (self.min_freq_hz <= freq_zc <= self.max_freq_hz):
            return None

        # 4. ACF 峰值检测
        acf_result = PeriodicMotionDetector._acf_peak_period(
            detrended, self.fps, self.min_freq_hz, self.max_freq_hz
        )
        if acf_result is None:
            return None
        acf_period_frames, acf_peak_val = acf_result

        freq_acf = self.fps / acf_period_frames if acf_period_frames > 0 else 0
        if not (self.min_freq_hz <= freq_acf <= self.max_freq_hz):
            return None

        # 5. 频率一致性
        if freq_zc > 0 and abs(freq_acf - freq_zc) / freq_zc > 0.4:
            return None

        # 6. 周期一致性
        cycle_lengths = self._extract_cycle_lengths(detrended)
        consistency = self._compute_consistency(cycle_lengths)
        if consistency < 0.5:
            return None

        # 7. 振幅（已是 torso_units）
        amplitude = float((np.max(detrended) - np.min(detrended)) / 2.0)

        cycle_count = len(cycle_lengths)

        return {
            "is_periodic": True,
            "frequency_hz": float((freq_zc + freq_acf) / 2.0),
            "amplitude_tu": amplitude,
            "cycle_count": cycle_count,
            "consistency": consistency,
            "zero_crossings": zero_crossings,
            "acf_peak": acf_peak_val,
        }

    @staticmethod
    def _count_zero_crossings(series: np.ndarray) -> int:
        signs = np.sign(series)
        diff = np.diff(signs)
        return int(np.sum(diff != 0))

    @staticmethod
    def _acf_peak_period(
        series: np.ndarray, fps: float, min_freq: float, max_freq: float
    ) -> Optional[Tuple[int, float]]:
        n = len(series)
        if n < 10:
            return None
        s = series - np.mean(series)
        fft_result = np.fft.fft(s, n=n * 2)
        acf = np.fft.ifft(fft_result * np.conjugate(fft_result)).real[:n]
        acf = acf / acf[0]
        min_lag = max(3, int(fps / max_freq) - 1)
        max_lag = min(n // 2, int(fps / min_freq) + 3)
        if max_lag <= min_lag:
            return None
        peak_idx = min_lag + int(np.argmax(acf[min_lag:max_lag]))
        peak_val = float(acf[peak_idx])
        if peak_val < 0.25:
            return None
        return peak_idx, peak_val

    @staticmethod
    def _extract_cycle_lengths(series: np.ndarray) -> List[float]:
        signs = np.sign(series)
        crossings = []
        for i in range(1, len(signs)):
            if signs[i] == 0:
                continue
            if signs[i - 1] != 0 and signs[i] != signs[i - 1]:
                crossings.append(i)
        if len(crossings) < 3:
            return []
        return [crossings[i] - crossings[i - 1] for i in range(1, len(crossings))]

    @staticmethod
    def _compute_consistency(lengths: List[float]) -> float:
        if len(lengths) < 2:
            return 0.0
        mean_len = np.mean(lengths)
        if mean_len < 1e-6:
            return 0.0
        std_len = np.std(lengths)
        cv = std_len / mean_len
        return float(max(0.0, 1.0 - cv))


# =============================================================================
# 归一化姿态特征提取器（角度链）
# =============================================================================

class NormalizedPoseFeatures:
    """将原始 COCO 关键点转换为躯干归一化坐标系，并计算关节角度链。"""

    L_SHOULDER = 5
    R_SHOULDER = 6
    L_ELBOW = 7
    R_ELBOW = 8
    L_WRIST = 9
    R_WRIST = 10
    L_HIP = 11
    R_HIP = 12

    def __init__(self, keypoints: np.ndarray) -> None:
        self.kpts = keypoints
        self.torso_size = self._compute_torso_size()

    def _kp(self, idx: int) -> Optional[np.ndarray]:
        if self.kpts is None or len(self.kpts) <= idx:
            return None
        kp = self.kpts[idx]
        if len(kp) >= 3 and kp[2] < 0.3:
            return None
        return np.array(kp[:2], dtype=float)

    def _compute_torso_size(self) -> float:
        ls = self._kp(self.L_SHOULDER)
        rs = self._kp(self.R_SHOULDER)
        lh = self._kp(self.L_HIP)
        rh = self._kp(self.R_HIP)
        if ls is None or rs is None or lh is None or rh is None:
            if ls is not None and rs is not None:
                return float(np.linalg.norm(ls - rs))
            return 100.0
        d1 = np.linalg.norm(ls - lh)
        d2 = np.linalg.norm(ls - rh)
        d3 = np.linalg.norm(rs - lh)
        d4 = np.linalg.norm(rs - rh)
        return float((d1 + d2 + d3 + d4) / 4.0)

    @staticmethod
    def _angle_3pt(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba = a - b
        bc = c - b
        n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
        if n1 < 1e-6 or n2 < 1e-6:
            return 180.0
        cos_ang = np.dot(ba, bc) / (n1 * n2)
        return float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))

    def theta1(self, side: str) -> Optional[float]:
        if side == "left":
            hip = self._kp(self.L_HIP)
            shoulder = self._kp(self.L_SHOULDER)
            elbow = self._kp(self.L_ELBOW)
        else:
            hip = self._kp(self.R_HIP)
            shoulder = self._kp(self.R_SHOULDER)
            elbow = self._kp(self.R_ELBOW)
        if shoulder is None or elbow is None:
            return None
        if hip is None:
            hip = np.array([shoulder[0], shoulder[1] + self.torso_size])
        return self._angle_3pt(hip, shoulder, elbow)

    def theta2(self, side: str) -> Optional[float]:
        if side == "left":
            s, e, w = self._kp(self.L_SHOULDER), self._kp(self.L_ELBOW), self._kp(self.L_WRIST)
        else:
            s, e, w = self._kp(self.R_SHOULDER), self._kp(self.R_ELBOW), self._kp(self.R_WRIST)
        if s is None or e is None or w is None:
            return None
        return self._angle_3pt(s, e, w)

    def arm_extension_ratio(self, side: str) -> Optional[float]:
        if side == "left":
            s, e, w = self._kp(self.L_SHOULDER), self._kp(self.L_ELBOW), self._kp(self.L_WRIST)
        else:
            s, e, w = self._kp(self.R_SHOULDER), self._kp(self.R_ELBOW), self._kp(self.R_WRIST)
        if s is None or e is None or w is None:
            return None
        d_se = np.linalg.norm(s - e)
        d_ew = np.linalg.norm(e - w)
        d_sw = np.linalg.norm(s - w)
        if d_se + d_ew < 1e-6:
            return None
        return float(d_sw / (d_se + d_ew))


# =============================================================================
# 三锁合取引擎
# =============================================================================

@dataclass
class TripleLockState:
    """单侧三锁状态（每 track_id + side 一个实例）。"""

    # 锁计数器（连续满足帧数）
    pose_count: int = 0
    orientation_count: int = 0
    motion_count: int = 0

    # 释放计数器（连续不满足帧数）
    pose_release: int = 0
    orientation_release: int = 0
    motion_release: int = 0

    # 确认后的保持
    hold_frames: int = 0
    confirmed: bool = False
    peak_confidence: float = 0.0

    # 历史数据
    wrist_local_history: deque = field(default_factory=lambda: deque(maxlen=30))
    timestamp_history: deque = field(default_factory=lambda: deque(maxlen=30))
    periodic_detector: PeriodicMotionDetector = field(default_factory=lambda: PeriodicMotionDetector())
    normal_smoother: NormalSmoother = field(default_factory=lambda: NormalSmoother(alpha=0.3))
    iri_calculator: IRICalculator = field(default_factory=lambda: IRICalculator(window_size=15))

    # 缓存
    last_wrist_local: Optional[Tuple[float, float]] = None
    last_timestamp: Optional[float] = None
    smoothed_confidence: float = 0.0

    def reset(self) -> None:
        self.pose_count = 0
        self.orientation_count = 0
        self.motion_count = 0
        self.pose_release = 0
        self.orientation_release = 0
        self.motion_release = 0
        self.hold_frames = 0
        self.confirmed = False
        self.peak_confidence = 0.0
        self.wrist_local_history.clear()
        self.timestamp_history.clear()
        self.periodic_detector.reset()
        self.normal_smoother.reset()
        self.iri_calculator.reset()
        self.last_wrist_local = None
        self.last_timestamp = None
        self.smoothed_confidence = 0.0


class TripleLockEngine:
    """
    三锁合取机制。

    锁表：
    ┌──────────┬─────────────────────────────────────┬──────────┬─────────────────┐
    │ 锁       │ 判定条件                            │ 最小持续 │ 释放条件        │
    ├──────────┼─────────────────────────────────────┼──────────┼─────────────────┤
    │ 姿态锁   │ θ1 > 25° 且 θ2 > 15° 且 ext>0.1    │ 3帧      │ 任一条件不满足  │
    │ 朝向锁   │ n_smooth 与 Z 轴夹角 < 45°          │ 5帧      │ 夹角 > 60°      │
    │ 运动锁   │ FFT 主导频 0.5~3Hz 且幅度 > 0.1     │ 3帧      │ 周期消失或速<0.05│
    └──────────┴─────────────────────────────────────┴──────────┴─────────────────┘
    """

    def __init__(self) -> None:
        self.config = get_config()
        c = self.config.ai

        # 姿态锁阈值
        self.theta1_min = c.gesture_theta1_hailing_min
        self.theta2_min = c.gesture_theta2_straight_min
        self.ext_min = c.gesture_arm_extension_min

        # 朝向锁阈值
        self.orientation_lock_angle = c.gesture_orientation_lock_angle
        self.orientation_release_angle = c.gesture_orientation_release_angle

        # 运动锁阈值（torso_units/s）
        self.motion_freq_min = c.gesture_motion_freq_min
        self.motion_freq_max = c.gesture_motion_freq_max
        self.motion_amp_min = c.gesture_motion_amp_min
        self.motion_speed_min = c.gesture_motion_speed_min

        # 锁持续帧数
        self.pose_min_frames = c.gesture_pose_min_frames
        self.orientation_min_frames = c.gesture_orientation_min_frames
        self.motion_min_frames = c.gesture_motion_min_frames

        # 确认后保持
        self.hold_max_frames = c.gesture_hold_max_frames

        # 置信度平滑
        self.ema_alpha = c.gesture_ema_alpha

        # 每 track_id_side 一个状态
        self._states: Dict[str, TripleLockState] = {}

    def _state_key(self, track_id: str, side: str) -> str:
        return f"{track_id}_{side}"

    def _get_state(self, track_id: str, side: str) -> TripleLockState:
        key = self._state_key(track_id, side)
        if key not in self._states:
            self._states[key] = TripleLockState()
        return self._states[key]

    def _clear_state(self, track_id: str, side: str) -> None:
        key = self._state_key(track_id, side)
        if key in self._states:
            self._states[key].reset()
            del self._states[key]

    def gc_states(self, active_track_ids: set) -> None:
        stale = [k for k in self._states if k.rsplit("_", 1)[0] not in active_track_ids]
        for k in stale:
            self._states[k].reset()
            del self._states[k]

    def reset(self) -> None:
        for s in self._states.values():
            s.reset()
        self._states.clear()

    def process_frame(
        self,
        keypoints: np.ndarray,
        side: str,
        track_id: str,
        palm_normal: Optional[np.ndarray],
        timestamp: float,
        f_human: float,
        f_human_multiplier: float,
    ) -> Tuple[str, float]:
        """
        处理单帧，返回 (gesture, confidence)。
        """
        state = self._get_state(track_id, side)

        # ---- 1. 局部参考系转换 ----
        wrist_local, torso_scale, frame_valid = wrist_to_local_frame(keypoints, side)
        if not frame_valid or wrist_local is None:
            state.reset()
            return "none", 0.0

        # ---- 2. 速度计算（基于 wrist_local） ----
        v_mag = 0.0
        if state.last_wrist_local is not None and state.last_timestamp is not None:
            dt = timestamp - state.last_timestamp
            if dt > 1e-6:
                _, _, v_mag = local_velocity(state.last_wrist_local, wrist_local, dt)

        state.last_wrist_local = wrist_local
        state.last_timestamp = timestamp

        # ---- 3. 喂入周期性检测器 ----
        state.periodic_detector.feed(wrist_local)
        period_info = state.periodic_detector.detect()

        # ---- 4. 姿态锁判定 ----
        feat = NormalizedPoseFeatures(keypoints)
        theta1 = feat.theta1(side)
        theta2 = feat.theta2(side)
        ext_ratio = feat.arm_extension_ratio(side)

        pose_ok = False
        pose_score = 0.0
        if theta1 is not None and theta2 is not None and ext_ratio is not None:
            if theta1 > self.theta1_min and theta2 > self.theta2_min and ext_ratio > self.ext_min:
                pose_ok = True
                pose_score = min(1.0, 0.5 + (theta1 - self.theta1_min) / 120.0)
                pose_score = min(1.0, pose_score + (theta2 - self.theta2_min) / 120.0)

        if pose_ok:
            state.pose_count += 1
            state.pose_release = 0
        else:
            state.pose_release += 1
            if state.pose_release >= 2:  # 允许 1 帧抖动
                state.pose_count = 0

        pose_locked = state.pose_count >= self.pose_min_frames

        # ---- 5. 朝向锁判定 ----
        orientation_ok = False
        if palm_normal is not None:
            angle = angle_to_camera_z(palm_normal)
            orientation_ok = angle < self.orientation_lock_angle

        if orientation_ok:
            state.orientation_count += 1
            state.orientation_release = 0
        else:
            state.orientation_release += 1
            if state.orientation_release >= 2:
                state.orientation_count = 0

        orientation_locked = state.orientation_count >= self.orientation_min_frames

        # ---- 6. 运动锁判定 ----
        motion_ok = False
        motion_score = 0.0
        if period_info and period_info.get("is_periodic"):
            freq = period_info.get("frequency_hz", 0.0)
            amp = period_info.get("amplitude_tu", 0.0)
            consistency = period_info.get("consistency", 0.0)
            freq_ok = self.motion_freq_min <= freq <= self.motion_freq_max
            amp_ok = amp > self.motion_amp_min
            if freq_ok and amp_ok:
                motion_ok = True
                motion_score = min(1.0, amp / 0.3) * min(1.0, consistency)

        # 若速度过低也释放运动锁
        if v_mag < self.motion_speed_min:
            motion_ok = False

        if motion_ok:
            state.motion_count += 1
            state.motion_release = 0
        else:
            state.motion_release += 1
            if state.motion_release >= 2:
                state.motion_count = 0

        motion_locked = state.motion_count >= self.motion_min_frames

        # ---- 7. 三锁合取 / 保持 ----
        all_locked = pose_locked and orientation_locked and motion_locked

        # ---- IRI（意图刚性指数）----
        iri_r = state.iri_calculator.feed(keypoints, side, palm_normal)

        if all_locked:
            state.confirmed = True
            state.hold_frames = self.hold_max_frames
            # 记录峰值置信度：S = Pose_score * R * Motion_score * F_human
            state.peak_confidence = pose_score * iri_r * motion_score * f_human_multiplier

        gesture = "none"
        raw_conf = 0.0

        if state.confirmed and state.hold_frames > 0:
            # 保持期内输出衰减中的峰值置信度
            decay = state.hold_frames / self.hold_max_frames
            raw_conf = state.peak_confidence * decay
            gesture = "waving"
            state.hold_frames -= 1
            # 若保持期耗尽，释放 confirmed
            if state.hold_frames <= 0:
                state.confirmed = False
        else:
            state.confirmed = False
            state.hold_frames = 0
            state.peak_confidence = 0.0

        # EMA 平滑
        state.smoothed_confidence = (
            self.ema_alpha * raw_conf + (1 - self.ema_alpha) * state.smoothed_confidence
        )

        logger.debug(
            "triple-lock[%s/%s] pose=%s|%d orient=%s|%d motion=%s|%d "
            "v=%.3fTU/s conf=%.3f f_human=%.2f",
            track_id, side,
            "Y" if pose_locked else "N", state.pose_count,
            "Y" if orientation_locked else "N", state.orientation_count,
            "Y" if motion_locked else "N", state.motion_count,
            v_mag, state.smoothed_confidence, f_human,
        )

        return gesture, state.smoothed_confidence


# =============================================================================
# 手势识别器主类（兼容旧接口）
# =============================================================================

class GestureRecognizer:
    """
    帧级手势识别器 —— 三锁合取机制。
    """

    L_WRIST = 9
    R_WRIST = 10

    def __init__(self) -> None:
        self.config = get_config()
        self.engine = TripleLockEngine()

        logger.info(
            "GestureRecognizer(三锁合取): pose_min=%d orient_min=%d motion_min=%d hold=%d",
            self.engine.pose_min_frames,
            self.engine.orientation_min_frames,
            self.engine.motion_min_frames,
            self.engine.hold_max_frames,
        )

    def recognize(
        self,
        keypoints: np.ndarray,
        track_id: str = "default",
        left_palm_normal: Optional[np.ndarray] = None,
        right_palm_normal: Optional[np.ndarray] = None,
        frame_timestamp: Optional[float] = None,
        active_track_ids: Optional[set] = None,
    ) -> "GestureResult":
        """
        帧级手势识别。
        左右手分别运行三锁引擎，取置信度最高者。
        """
        if keypoints is None or len(keypoints) < 17:
            return GestureResult(gesture_type=GestureType.NONE, confidence=0.0)

        if active_track_ids is not None:
            self.engine.gc_states(active_track_ids)

        now = frame_timestamp if frame_timestamp is not None else time.time()

        # 面部过滤（零模型）
        c = self.config.ai
        f_human, is_hard_rejected, f_human_multiplier = facing_gate(
            keypoints,
            hard_threshold=c.gesture_facing_hard_threshold,
            soft_threshold=c.gesture_facing_soft_threshold,
        )
        if is_hard_rejected:
            logger.debug("facing-gate[%s] hard-rejected f_human=%.2f", track_id, f_human)
            return GestureResult(gesture_type=GestureType.NONE, confidence=0.0)

        best_result: Optional[GestureResult] = None

        for side in ["right", "left"]:
            raw_normal = left_palm_normal if side == "left" else right_palm_normal

            # ---- MediaPipe 法向量平滑 ----
            palm_normal = None
            if raw_normal is not None:
                state = self.engine._get_state(track_id, side)
                palm_normal = state.normal_smoother.update(raw_normal)

            gesture, confidence = self.engine.process_frame(
                keypoints=keypoints,
                side=side,
                track_id=track_id,
                palm_normal=palm_normal,
                timestamp=now,
                f_human=f_human,
                f_human_multiplier=f_human_multiplier,
            )

            result = GestureResult(
                gesture_type=GestureType.WAVING if gesture == "waving" else GestureType.NONE,
                confidence=confidence,
            )

            if best_result is None or result.confidence > best_result.confidence:
                best_result = result

        return best_result if best_result else GestureResult(
            gesture_type=GestureType.NONE, confidence=0.0
        )

    def reset(self) -> None:
        """重置所有状态。"""
        self.engine.reset()
        self._mp_skip_counter.clear()
        logger.info("手势识别器已重置")


# =============================================================================
# 手势类型与结果定义
# =============================================================================

class GestureType(str, Enum):
    NONE = "none"
    WAVING = "waving"
    GREETING = "greeting"
    HAILING = "hailing"
    HAND_UP = "hand_up"


@dataclass
class GestureResult:
    gesture_type: GestureType = GestureType.NONE
    confidence: float = 0.0
    wrist_pos: Optional[Tuple[float, float]] = None


# =============================================================================
# 全局单例 + 便捷函数
# =============================================================================

_recognizer: Optional[GestureRecognizer] = None


def get_recognizer() -> GestureRecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = GestureRecognizer()
    return _recognizer


def is_hailing_gesture(
    keypoints: np.ndarray,
    track_id: str = "default",
    left_palm_normal: Optional[np.ndarray] = None,
    right_palm_normal: Optional[np.ndarray] = None,
    frame_timestamp: Optional[float] = None,
    active_track_ids: Optional[set] = None,
) -> Tuple[str, float]:
    """便捷函数。"""
    recognizer = get_recognizer()
    result = recognizer.recognize(
        keypoints, track_id,
        left_palm_normal, right_palm_normal,
        frame_timestamp,
        active_track_ids=active_track_ids,
    )
    return result.gesture_type.value, result.confidence
