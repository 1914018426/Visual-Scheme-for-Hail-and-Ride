"""
手势识别模块 —— 增强型帧级状态机 + 周期性检测引擎

基于前沿文献改进：
1. 躯干归一化坐标系（Leeds Univ. 2025）
2. 周期性运动检测（SFU Bruce et al. CRV 2016）
3. 关节角度链 θ1-θ2（Tunis taxi-hailing, MDPI 2023）
4. 方向符号变化追踪（来回摆动检测）
5. 手掌朝向纯 2D 几何判断（不依赖 z 坐标）
6. 置信度 EMA 时序平滑

核心设计：
- 状态机：IDLE → POSED → OSCILLATING → CONFIRMED → IDLE
- 响应延迟：8-12 帧（约 0.5-0.8s @ 15fps，周期性检测需要更长窗口）
- 意图语义：hailing = 高举 + 垂直周期性挥动；greeting = 平伸 + 水平周期性挥动

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


# =============================================================================
# 周期性运动检测引擎
# =============================================================================

class PeriodicMotionDetector:
    """
    基于 wrist 位置序列的周期性运动检测器。

    使用 zero-crossing + 自相关函数(ACF) 检测稳定的周期性运动。
    人类挥手/招手的典型频率：1-3 Hz。
    """

    def __init__(
        self,
        buffer_seconds: float = 2.0,
        fps: float = 15.0,
        min_freq_hz: float = 0.8,
        max_freq_hz: float = 3.5,
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

    def feed(self, wrist_pos: Tuple[float, float]) -> None:
        """喂入新一帧的 wrist 位置（像素坐标）。"""
        self._xs.append(float(wrist_pos[0]))
        self._ys.append(float(wrist_pos[1]))

    def detect(self) -> Optional[Dict[str, Any]]:
        """
        检测周期性运动。

        Returns:
            dict with keys:
                - is_periodic: bool
                - dominant_axis: "x" | "y" | "none"  # 周期性更强的轴
                - frequency_hz: float
                - amplitude_tu: float   # 振幅（已归一化，调用方需提供 torso_size）
                - cycle_count: int
                - consistency: float    # 0-1，周期一致性
                - zero_crossings: int
            or None if insufficient data.
        """
        if len(self._xs) < self._maxlen * 0.6:
            return None

        # 分别检测 x 和 y 轴的周期性
        x_result = self._analyze_axis(np.array(self._xs))
        y_result = self._analyze_axis(np.array(self._ys))

        # 选择周期性更强的轴
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

        # 1. 去趋势（减去线性漂移）
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

        # 4. 自相关函数 (ACF) 峰值检测
        acf_result = PeriodicMotionDetector._acf_peak_period(
            detrended, self.fps, self.min_freq_hz, self.max_freq_hz
        )
        if acf_result is None:
            return None
        acf_period_frames, acf_peak_val = acf_result

        # ACF 周期对应的频率
        freq_acf = self.fps / acf_period_frames if acf_period_frames > 0 else 0
        if not (self.min_freq_hz <= freq_acf <= self.max_freq_hz):
            return None

        # 5. 频率一致性：zero-crossing 和 ACF 估计的频率应接近
        if freq_zc > 0 and abs(freq_acf - freq_zc) / freq_zc > 0.4:
            return None

        # 6. 周期一致性：检测相邻周期长度是否稳定
        cycle_lengths = self._extract_cycle_lengths(detrended)
        consistency = self._compute_consistency(cycle_lengths)
        if consistency < 0.5:
            return None

        # 7. 振幅（峰值到谷值）
        amplitude = float((np.max(detrended) - np.min(detrended)) / 2.0)

        # 8. 周期数
        cycle_count = len(cycle_lengths)

        return {
            "is_periodic": True,
            "frequency_hz": float((freq_zc + freq_acf) / 2.0),
            "amplitude_pixels": amplitude,
            "cycle_count": cycle_count,
            "consistency": consistency,
            "zero_crossings": zero_crossings,
            "acf_peak": acf_peak_val,
        }

    @staticmethod
    def _count_zero_crossings(series: np.ndarray) -> int:
        """统计序列穿过零点的次数（符号变化）。"""
        signs = np.sign(series)
        # 忽略恰好在零上的点
        diff = np.diff(signs)
        return int(np.sum(diff != 0))

    @staticmethod
    def _acf_peak_period(
        series: np.ndarray, fps: float, min_freq: float, max_freq: float
    ) -> Optional[Tuple[int, float]]:
        """
        计算自相关函数并找到第一个显著峰值（排除滞后0）。
        返回 (period_in_frames, peak_correlation)。
        """
        n = len(series)
        if n < 10:
            return None

        # 零均值化
        s = series - np.mean(series)
        # 自相关（使用 FFT 加速）
        fft_result = np.fft.fft(s, n=n * 2)
        acf = np.fft.ifft(fft_result * np.conjugate(fft_result)).real[:n]
        acf = acf / acf[0]  # 归一化

        # 寻找第一个显著峰值
        # 人类挥手频率 1-3Hz，@ fps 对应周期 fps/max_freq ~ fps/min_freq 帧
        min_lag = max(3, int(fps / max_freq) - 1)
        max_lag = min(n // 2, int(fps / min_freq) + 3)
        if max_lag <= min_lag:
            return None

        peak_idx = min_lag + int(np.argmax(acf[min_lag:max_lag]))
        peak_val = float(acf[peak_idx])

        # 峰值必须显著高于周围
        if peak_val < 0.25:
            return None

        return peak_idx, peak_val

    @staticmethod
    def _extract_cycle_lengths(series: np.ndarray) -> List[float]:
        """从 zero-crossing 提取相邻周期长度（帧数）。"""
        signs = np.sign(series)
        crossings = []
        for i in range(1, len(signs)):
            if signs[i] == 0:
                continue
            if signs[i - 1] != 0 and signs[i] != signs[i - 1]:
                crossings.append(i)

        if len(crossings) < 3:
            return []

        # 相邻 crossing 的间隔
        lengths = [crossings[i] - crossings[i - 1] for i in range(1, len(crossings))]
        return lengths

    @staticmethod
    def _compute_consistency(lengths: List[float]) -> float:
        """计算周期长度的一致性（0-1）。"""
        if len(lengths) < 2:
            return 0.0
        mean_len = np.mean(lengths)
        if mean_len < 1e-6:
            return 0.0
        std_len = np.std(lengths)
        # 变异系数 CV = std / mean，一致性 = 1 - min(CV, 1)
        cv = std_len / mean_len
        return float(max(0.0, 1.0 - cv))


# =============================================================================
# 归一化姿态特征提取器
# =============================================================================

class NormalizedPoseFeatures:
    """
    将原始 COCO 关键点转换为躯干归一化坐标系，并计算关节角度链。
    """

    # COCO 姿态关键点索引
    NOSE = 0
    L_EYE = 1
    R_EYE = 2
    L_EAR = 3
    R_EAR = 4
    L_SHOULDER = 5
    R_SHOULDER = 6
    L_ELBOW = 7
    R_ELBOW = 8
    L_WRIST = 9
    R_WRIST = 10
    L_HIP = 11
    R_HIP = 12
    L_KNEE = 13
    R_KNEE = 14
    L_ANKLE = 15
    R_ANKLE = 16

    def __init__(self, keypoints: np.ndarray) -> None:
        self.kpts = keypoints
        self.torso_size = self._compute_torso_size()
        self.mid_hip = self._compute_mid_hip()

    def _kp(self, idx: int) -> Optional[np.ndarray]:
        if self.kpts is None or len(self.kpts) <= idx:
            return None
        kp = self.kpts[idx]
        if len(kp) >= 3 and kp[2] < 0.3:
            return None
        return np.array(kp[:2], dtype=float)

    def _compute_torso_size(self) -> float:
        """
        躯干大小 = 肩-髋四角距离的平均值。
        Leeds Univ. 论文公式：ts = (|LS-LH| + |LS-RH| + |RS-LH| + |RS-RH|) / 4
        """
        ls = self._kp(self.L_SHOULDER)
        rs = self._kp(self.R_SHOULDER)
        lh = self._kp(self.L_HIP)
        rh = self._kp(self.R_HIP)

        if ls is None or rs is None or lh is None or rh is None:
            # 回退：用肩宽
            if ls is not None and rs is not None:
                return float(np.linalg.norm(ls - rs))
            return 100.0  # 默认像素值

        d1 = np.linalg.norm(ls - lh)
        d2 = np.linalg.norm(ls - rh)
        d3 = np.linalg.norm(rs - lh)
        d4 = np.linalg.norm(rs - rh)
        return float((d1 + d2 + d3 + d4) / 4.0)

    def _compute_mid_hip(self) -> np.ndarray:
        lh = self._kp(self.L_HIP)
        rh = self._kp(self.R_HIP)
        if lh is not None and rh is not None:
            return (lh + rh) / 2.0
        if lh is not None:
            return lh
        if rh is not None:
            return rh
        # 回退到画面中心下方
        return np.array([400.0, 400.0])

    def normalize_point(self, idx: int) -> Optional[np.ndarray]:
        """返回躯干归一化坐标（相对于 mid_hip，单位 TU）。"""
        p = self._kp(idx)
        if p is None:
            return None
        if self.torso_size < 1e-6:
            return p - self.mid_hip
        return (p - self.mid_hip) / self.torso_size

    def normalize_distance(self, dist_pixels: float) -> float:
        """将像素距离转换为躯干单位。"""
        if self.torso_size < 1e-6:
            return dist_pixels / 100.0
        return dist_pixels / self.torso_size

    @staticmethod
    def _angle_3pt(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """三点夹角（度）。b 为顶点。"""
        ba = a - b
        bc = c - b
        n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
        if n1 < 1e-6 or n2 < 1e-6:
            return 180.0
        cos_ang = np.dot(ba, bc) / (n1 * n2)
        return float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))

    def theta1(self, side: str) -> Optional[float]:
        """
        θ1 = hip-shoulder-elbow 夹角（评估手臂整体抬起程度）。
        Tunis 论文中定义为 θ1 = hip^shoulder_elbow。
        """
        if side == "left":
            hip = self._kp(self.L_HIP)
            shoulder = self._kp(self.L_SHOULDER)
            elbow = self._kp(self.L_ELBOW)
        else:
            hip = self._kp(self.R_HIP)
            shoulder = self._kp(self.R_SHOULDER)
            elbow = self._kp(self.R_ELBOW)

        if hip is None or shoulder is None or elbow is None:
            return None
        return self._angle_3pt(hip, shoulder, elbow)

    def theta2(self, side: str) -> Optional[float]:
        """
        θ2 = shoulder-elbow-wrist 夹角（评估手臂伸直程度）。
        伸直时接近 180°，弯曲时减小。
        """
        if side == "left":
            shoulder = self._kp(self.L_SHOULDER)
            elbow = self._kp(self.L_ELBOW)
            wrist = self._kp(self.L_WRIST)
        else:
            shoulder = self._kp(self.R_SHOULDER)
            elbow = self._kp(self.R_ELBOW)
            wrist = self._kp(self.R_WRIST)

        if shoulder is None or elbow is None or wrist is None:
            return None
        return self._angle_3pt(shoulder, elbow, wrist)

    def arm_extension_ratio(self, side: str) -> Optional[float]:
        """
        手臂伸展比例 = |shoulder-wrist| / (|shoulder-elbow| + |elbow-wrist|)。
        完全伸直时 ≈ 1.0，弯曲时 < 1.0。
        """
        if side == "left":
            s = self._kp(self.L_SHOULDER)
            e = self._kp(self.L_ELBOW)
            w = self._kp(self.L_WRIST)
        else:
            s = self._kp(self.R_SHOULDER)
            e = self._kp(self.R_ELBOW)
            w = self._kp(self.R_WRIST)

        if s is None or e is None or w is None:
            return None
        d_se = np.linalg.norm(s - e)
        d_ew = np.linalg.norm(e - w)
        d_sw = np.linalg.norm(s - w)
        if d_se + d_ew < 1e-6:
            return None
        return float(d_sw / (d_se + d_ew))

    def wrist_height_relative(self, side: str) -> Optional[float]:
        """
        手腕相对 mid_hip 的归一化高度（TU）。
        正值 = 在 mid_hip 上方（y 坐标更小），负值 = 下方。
        """
        if side == "left":
            wrist = self._kp(self.L_WRIST)
        else:
            wrist = self._kp(self.R_WRIST)

        if wrist is None:
            return None
        # 在图像坐标中 y 向下为正，所以上方是负值
        dy = self.mid_hip[1] - wrist[1]
        return self.normalize_distance(dy)


# =============================================================================
# 手势类型与状态机定义
# =============================================================================

class GestureType(str, Enum):
    """手势类型枚举。"""

    NONE = "none"
    GREETING = "greeting"  # 打招呼：水平方向周期性挥动
    HAILING = "hailing"    # 打车：高举 + 垂直方向周期性挥动
    HAND_UP = "hand_up"    # 举手：手臂举起但无周期性挥动


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
        IDLE → POSED → OSCILLATING → CONFIRMED → IDLE
    """

    state: str = "idle"                 # idle / posed / oscillating / confirmed / hand_up
    frames_in_state: int = 0             # 在当前状态的累计帧数
    consecutive_wave_frames: int = 0     # 连续挥动帧数
    stop_frames: int = 0                 # 连续停止帧数
    last_wrist_pos: Optional[Tuple[float, float]] = None
    last_timestamp: Optional[float] = None
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=30))
    direction_history: deque = field(default_factory=lambda: deque(maxlen=30))
    sign_changes: int = 0                # 主方向上的符号变化次数
    main_direction: str = "none"         # horizontal / vertical / none
    confirmed_gesture: Optional[str] = None
    peak_confidence: float = 0.0
    smoothed_confidence: float = 0.0     # EMA 平滑后的置信度
    periodic_detector: PeriodicMotionDetector = field(
        default_factory=lambda: PeriodicMotionDetector()
    )
    # 缓存上一帧的归一化特征（用于日志和调试）
    last_features: Dict[str, Any] = field(default_factory=dict)



# =============================================================================
# 手势识别器主类
# =============================================================================

class GestureRecognizer:
    """
    增强型帧级手势识别器。

    基于速度向量 + 周期性检测 + 关节角度链 + 帧级状态机。
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

        # ---- 手臂姿势阈值（角度，度）----
        self.theta1_hailing_min = getattr(c, "gesture_theta1_hailing_min", 110.0)
        self.theta1_greeting_min = getattr(c, "gesture_theta1_greeting_min", 75.0)
        self.theta1_greeting_max = getattr(c, "gesture_theta1_greeting_max", 115.0)
        self.theta2_straight_min = getattr(c, "gesture_theta2_straight_min", 140.0)
        self.arm_extension_min = getattr(c, "gesture_arm_extension_min", 0.85)

        # ---- 速度阈值（躯干单位 TU/秒）----
        self.velocity_threshold = getattr(c, "gesture_velocity_threshold", 2.5)
        self.velocity_idle_ratio = getattr(c, "gesture_velocity_idle_ratio", 0.25)

        # ---- 状态机参数 ----
        self.confirm_frames = getattr(c, "gesture_confirm_frames", 5)
        self.stop_reset_frames = getattr(c, "gesture_stop_reset_frames", 6)
        self.idle_reset_frames = getattr(c, "gesture_idle_reset_frames", 4)

        # ---- 周期性检测参数 ----
        self.period_min_cycles = getattr(c, "gesture_period_min_cycles", 2)
        self.period_min_amplitude = getattr(c, "gesture_period_min_amplitude", 0.35)
        self.period_consistency_min = getattr(c, "gesture_period_consistency_min", 0.45)
        self.period_min_freq = getattr(c, "gesture_period_min_freq", 0.8)
        self.period_max_freq = getattr(c, "gesture_period_max_freq", 3.5)

        # ---- 方向追踪参数 ----
        self.sign_change_min = getattr(c, "gesture_sign_change_min", 2)
        self.direction_consistency_min = getattr(c, "gesture_direction_consistency_min", 0.65)

        # ---- 手掌朝向 ----
        self.palm_fan_angle_min = getattr(c, "gesture_palm_fan_angle_min", 50.0)
        self.palm_finger_ratio_min = getattr(c, "gesture_palm_finger_ratio_min", 1.2)

        # ---- 置信度平滑 ----
        self.ema_alpha = getattr(c, "gesture_ema_alpha", 0.35)
        self.confidence_threshold = getattr(c, "gesture_confidence_threshold", 0.55)

        # ---- 快速模式（跳过周期性检测）----
        self.fast_mode = getattr(c, "gesture_fast_mode", False)

        # 每 track_id_side 一个状态机
        self._machines: Dict[str, SideStateMachine] = {}

        logger.info(
            "GestureRecognizer(增强型): vel_thresh=%.2fTU/s theta1_hail=%.0f° "
            "theta2_straight=%.0f° period_min_cycles=%d fast_mode=%s",
            self.velocity_threshold,
            self.theta1_hailing_min,
            self.theta2_straight_min,
            self.period_min_cycles,
            self.fast_mode,
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
            # 用当前参数初始化 periodic_detector
            pd = self._machines[key].periodic_detector
            pd.min_cycles = self.period_min_cycles
            pd.min_freq_hz = self.period_min_freq
            pd.max_freq_hz = self.period_max_freq
        return self._machines[key]

    def _clear_machine(self, track_id: str) -> None:
        for side in ["left", "right"]:
            key = self._machine_key(track_id, side)
            if key in self._machines:
                self._machines[key].periodic_detector.reset()
                del self._machines[key]

    @staticmethod
    def _keypoint_conf(kp) -> float:
        return float(kp[2]) if len(kp) > 2 else 1.0

    # ------------------------------------------------------------------ #
    # 手掌朝向检测（纯 2D 几何，不依赖 z 坐标）
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_palm_facing_camera(
        hand_landmarks: List[Tuple[float, float, float]],
    ) -> Tuple[bool, float]:
        """
        判断手掌是否朝向摄像头（掌心朝前）。
        纯 2D 几何判断，完全不依赖 z 坐标。

        策略：
        1. 手指展开扇形角 > threshold（手掌张开）
        2. 指尖到 wrist 距离 > 对应 PIP 到 wrist 距离（手指伸出）
        3. 拇指在四指"外侧"
        """
        if not hand_landmarks or len(hand_landmarks) < 21:
            return False, 0.0

        pts = np.array(hand_landmarks)
        wrist = pts[0][:2]

        # 四指: tip, pip, mcp
        fingers = [(8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)]

        # 1. 手指展开扇形角（相对于 wrist 的极坐标角度）
        angles = []
        for tip_idx, _, mcp_idx in fingers:
            vec = pts[tip_idx][:2] - wrist
            ang = np.arctan2(vec[1], vec[0])
            angles.append(ang)
        angles = np.sort(np.array(angles))
        # 计算角度跨度（考虑环绕）
        span = float(angles[-1] - angles[0])
        if span > np.pi:
            span = 2 * np.pi - span
        span_deg = np.degrees(span)

        # 2. 指尖-指根距离比
        finger_ratios = []
        open_count = 0
        for tip_idx, pip_idx, mcp_idx in fingers:
            d_tip = np.linalg.norm(pts[tip_idx][:2] - wrist)
            d_pip = np.linalg.norm(pts[pip_idx][:2] - wrist)
            if d_pip > 1e-6:
                ratio = d_tip / d_pip
                finger_ratios.append(ratio)
                if ratio > 1.15:
                    open_count += 1

        avg_ratio = np.mean(finger_ratios) if finger_ratios else 0.0

        # 3. 拇指位置判断（thumb tip 应在 index mcp 的"外侧"）
        thumb_tip = pts[4][:2]
        thumb_ip = pts[3][:2]
        thumb_mcp = pts[2][:2]
        index_mcp = pts[5][:2]
        # 简单判断：thumb tip 到 wrist 的距离是否显著大于 thumb mcp 到 wrist
        thumb_extended = np.linalg.norm(thumb_tip - wrist) > np.linalg.norm(thumb_mcp - wrist) * 1.1

        # 综合判断
        is_facing = (
            span_deg > 45.0          # 手指有一定展开
            and avg_ratio > 1.05     # 指尖比指根远
            and open_count >= 2      # 至少 2 根手指展开
            and thumb_extended       # 拇指也伸出
        )

        confidence = (
            min(1.0, span_deg / 120.0) * 0.3
            + min(1.0, avg_ratio / 1.5) * 0.3
            + (open_count / 4.0) * 0.25
            + (0.15 if thumb_extended else 0.0)
        )
        return is_facing, confidence

    # ------------------------------------------------------------------ #
    # 手臂姿势检测（θ1-θ2 角度链）
    # ------------------------------------------------------------------ #

    def _detect_arm_pose(
        self, keypoints: np.ndarray, side: str
    ) -> Tuple[bool, bool, bool, float, Dict[str, Any]]:
        """
        检测手臂姿势（基于 θ1-θ2 角度链）。

        Returns: (is_posed, is_raised, is_forward, confidence, features_dict)

        is_raised : θ1 大（手臂从躯干大幅抬起），对应 hailing
        is_forward: θ1 中等（手臂平伸），对应 greeting
        is_posed  : is_raised or is_forward
        """
        feat = NormalizedPoseFeatures(keypoints)

        theta1 = feat.theta1(side)
        theta2 = feat.theta2(side)
        ext_ratio = feat.arm_extension_ratio(side)
        wrist_h = feat.wrist_height_relative(side)

        features = {
            "theta1": theta1,
            "theta2": theta2,
            "ext_ratio": ext_ratio,
            "wrist_h": wrist_h,
            "torso_size": feat.torso_size,
        }

        # 关键点不可信
        if theta1 is None or theta2 is None:
            return False, False, False, 0.0, features

        # 手臂必须充分伸直
        is_straight = theta2 > self.theta2_straight_min
        if ext_ratio is not None:
            is_straight = is_straight and (ext_ratio > self.arm_extension_min)

        if not is_straight:
            # 手臂弯曲，不认为是招手姿势
            return False, False, False, 0.0, features

        # θ1 大 = 手臂高举（hailing）
        is_raised = theta1 > self.theta1_hailing_min

        # θ1 中等 = 手臂平伸（greeting）
        is_forward = (
            self.theta1_greeting_min < theta1 <= self.theta1_greeting_max
        )

        is_posed = is_raised or is_forward

        # 置信度计算
        if is_raised:
            conf = min(1.0, 0.6 + (theta1 - self.theta1_hailing_min) / 120.0)
        elif is_forward:
            center = (self.theta1_greeting_min + self.theta1_greeting_max) / 2.0
            dist_from_center = abs(theta1 - center)
            conf = min(1.0, 0.7 - dist_from_center / 100.0)
        else:
            conf = 0.0

        return is_posed, is_raised, is_forward, conf, features

    # ------------------------------------------------------------------ #
    # 速度计算（归一化）
    # ------------------------------------------------------------------ #

    def _compute_velocity(
        self,
        machine: SideStateMachine,
        wrist_pos: Tuple[float, float],
        timestamp: float,
        torso_size: float,
    ) -> Tuple[float, float, float]:
        """
        计算 wrist 速度（躯干单位 TU/秒）。
        Returns: (vx_tu, vy_tu, magnitude_tu)
        """
        if machine.last_wrist_pos is None or machine.last_timestamp is None:
            machine.last_wrist_pos = wrist_pos
            machine.last_timestamp = timestamp
            return 0.0, 0.0, 0.0

        dt = timestamp - machine.last_timestamp
        if dt < 1e-6:
            return 0.0, 0.0, 0.0

        # 像素速度
        vx_px = (wrist_pos[0] - machine.last_wrist_pos[0]) / dt
        vy_px = (wrist_pos[1] - machine.last_wrist_pos[1]) / dt

        # 归一化为躯干单位
        ts = torso_size if torso_size > 1e-6 else 100.0
        vx = vx_px / ts
        vy = vy_px / ts
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
            hand_landmarks = (
                left_hand_landmarks if side == "left"
                else right_hand_landmarks
            )
            result = self._recognize_side(
                keypoints, side, track_id, hand_landmarks, now,
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
        if wrist[2] < 0.4 or shoulder[2] < 0.4:
            self._clear_machine(track_id)
            return None

        wrist_pos = (float(wrist[0]), float(wrist[1]))

        # 1. 手臂姿势（θ1-θ2 角度链）
        is_posed, is_raised, is_forward, arm_conf, features = self._detect_arm_pose(
            keypoints, side
        )

        # 2. 手掌朝向（纯 2D 几何）
        palm_facing = False
        palm_conf = 0.0
        if hand_landmarks and len(hand_landmarks) >= 21:
            palm_facing, palm_conf = self._is_palm_facing_camera(hand_landmarks)

        # 3. 速度计算（归一化）
        machine = self._get_machine(track_id, side)
        torso_size = features.get("torso_size", 100.0)
        vx, vy, v_mag = self._compute_velocity(machine, wrist_pos, timestamp, torso_size)

        # 4. 喂入周期性检测器
        machine.periodic_detector.feed(wrist_pos)
        period_info = machine.periodic_detector.detect()

        # 5. 记录速度和方向历史
        machine.velocity_history.append((vx, vy, v_mag))

        direction = "none"
        if v_mag > self.velocity_threshold:
            # 用更精细的方向判断（考虑速度符号）
            if abs(vx) > abs(vy) * 1.2:
                direction = "horizontal"
            elif abs(vy) > abs(vx) * 1.2:
                direction = "vertical"
            else:
                direction = "diagonal"
        machine.direction_history.append(direction)

        # 6. 追踪方向符号变化
        self._update_sign_changes(machine, vx, vy)

        # 缓存特征用于调试
        machine.last_features = {
            **features,
            "vx": vx, "vy": vy, "v_mag": v_mag,
            "palm_facing": palm_facing,
            "period_info": period_info,
            "sign_changes": machine.sign_changes,
        }

        # 7. 状态机流转
        gesture, confidence = self._state_transition(
            machine, is_posed, is_raised, is_forward, arm_conf,
            palm_facing, palm_conf, v_mag, period_info,
            vx, vy,
        )

        # 8. EMA 置信度平滑
        machine.smoothed_confidence = (
            self.ema_alpha * confidence
            + (1 - self.ema_alpha) * machine.smoothed_confidence
        )

        if gesture != "none":
            logger.info(
                "gesture[%s/%s]: %s raw_conf=%.2f smooth_conf=%.2f state=%s "
                "v=%.2fTU/s theta1=%s theta2=%s period=%s sign_changes=%d",
                track_id, side, gesture, confidence, machine.smoothed_confidence,
                machine.state, v_mag,
                features.get("theta1"),
                features.get("theta2"),
                "Y" if period_info and period_info.get("is_periodic") else "N",
                machine.sign_changes,
            )

        if gesture == "greeting":
            return GestureResult(GestureType.GREETING, machine.smoothed_confidence, wrist_pos)
        elif gesture == "hailing":
            return GestureResult(GestureType.HAILING, machine.smoothed_confidence, wrist_pos)
        elif gesture == "hand_up":
            return GestureResult(GestureType.HAND_UP, machine.smoothed_confidence, wrist_pos)
        return None

    def _update_sign_changes(
        self, machine: SideStateMachine, vx: float, vy: float
    ) -> None:
        """追踪主方向上的速度符号变化次数。"""
        hist = list(machine.velocity_history)
        if len(hist) < 2:
            return

        # 确定主方向（基于历史速度的平均方向）
        avg_vx = np.mean([h[0] for h in hist[-10:]])
        avg_vy = np.mean([h[1] for h in hist[-10:]])
        is_horizontal = abs(avg_vx) > abs(avg_vy)

        # 取主方向上的速度序列
        if is_horizontal:
            vals = [h[0] for h in hist]
        else:
            vals = [h[1] for h in hist]

        # 统计符号变化（忽略接近零的值）
        changes = 0
        threshold = self.velocity_threshold * 0.3
        last_sign = 0
        for v in vals:
            if abs(v) < threshold:
                continue
            curr_sign = 1 if v > 0 else -1
            if last_sign != 0 and curr_sign != last_sign:
                changes += 1
            last_sign = curr_sign

        machine.sign_changes = changes

    # ------------------------------------------------------------------ #
    # 状态机流转逻辑（增强版）
    # ------------------------------------------------------------------ #

    def _state_transition(
        self,
        machine: SideStateMachine,
        is_posed: bool,
        is_raised: bool,
        is_forward: bool,
        arm_conf: float,
        palm_facing: bool,
        palm_conf: float,
        v_mag: float,
        period_info: Optional[Dict[str, Any]],
        vx: float,
        vy: float,
    ) -> Tuple[str, float]:
        """
        增强型状态机。

        状态定义：
          idle       : 手臂自然下垂
          posed      : 手臂姿势符合（举起或平伸），等待挥动
          oscillating: 检测到来回摆动（符号变化），等待周期性确认
          confirmed  : 周期性确认，输出 greeting/hailing
          hand_up    : 手臂举起但无周期性挥动
        """
        state = machine.state
        machine.frames_in_state += 1

        # 手臂完全放下 → 立即重置
        if not is_posed:
            machine.state = "idle"
            machine.frames_in_state = 0
            machine.consecutive_wave_frames = 0
            machine.stop_frames = 0
            machine.sign_changes = 0
            machine.confirmed_gesture = None
            machine.smoothed_confidence = 0.0
            machine.periodic_detector.reset()
            return "none", 0.0

        # 是否处于运动状态
        is_moving = v_mag > self.velocity_threshold
        is_almost_still = v_mag < self.velocity_threshold * self.velocity_idle_ratio

        # 静止检测
        if is_almost_still:
            machine.stop_frames += 1
            if machine.stop_frames >= self.idle_reset_frames and state in ("posed", "hand_up"):
                machine.state = "idle"
                machine.frames_in_state = 0
                machine.consecutive_wave_frames = 0
                machine.stop_frames = 0
                machine.sign_changes = 0
                return "none", 0.0
        else:
            machine.stop_frames = max(0, machine.stop_frames - 1)

        # ---- 周期性检测通过？ ----
        period_ok = False
        if period_info and period_info.get("is_periodic"):
            amp = period_info.get("amplitude_pixels", 0.0)
            # 振幅需要归一化（但 periodic_detector 不知道 torso_size，这里用原始像素）
            # 典型肩宽 60-100 像素，振幅 > 0.35 TU ≈ 25-35 像素
            amp_ok = amp > 25.0
            consistency_ok = period_info.get("consistency", 0.0) > self.period_consistency_min
            cycles_ok = period_info.get("cycle_count", 0) >= self.period_min_cycles
            period_ok = amp_ok and consistency_ok and cycles_ok

        # ---- 方向追踪通过？ ----
        direction_ok = machine.sign_changes >= self.sign_change_min

        if state == "idle":
            if is_raised:
                machine.state = "hand_up"
                machine.frames_in_state = 1
                return "hand_up", min(arm_conf * 0.6, 0.85)
            elif is_forward:
                machine.state = "posed"
                machine.frames_in_state = 1
            return "none", 0.0

        if state == "posed":
            if is_moving and direction_ok:
                machine.state = "oscillating"
                machine.frames_in_state = 1
                machine.consecutive_wave_frames = 1
            elif is_raised:
                machine.state = "hand_up"
                machine.frames_in_state = 1
                return "hand_up", min(arm_conf * 0.6, 0.85)
            return "none", 0.0

        if state == "hand_up":
            if is_moving and direction_ok:
                machine.state = "oscillating"
                machine.frames_in_state = 1
                machine.consecutive_wave_frames = 1
                machine.stop_frames = 0
            return "hand_up", min(arm_conf * 0.6, 0.85)

        if state == "oscillating":
            if is_moving:
                machine.consecutive_wave_frames += 1
                machine.stop_frames = 0

                # 快速模式：跳过周期性检测，仅依赖方向变化
                if self.fast_mode:
                    if machine.consecutive_wave_frames >= self.confirm_frames:
                        machine.state = "confirmed"
                        machine.frames_in_state = 1
                        gesture, conf = self._classify_intent(
                            machine, is_raised, is_forward, palm_facing, arm_conf, palm_conf
                        )
                        machine.confirmed_gesture = gesture
                        machine.peak_confidence = conf
                        return gesture, conf
                else:
                    # 标准模式：需要周期性检测通过
                    if period_ok and machine.consecutive_wave_frames >= self.confirm_frames:
                        machine.state = "confirmed"
                        machine.frames_in_state = 1
                        gesture, conf = self._classify_intent(
                            machine, is_raised, is_forward, palm_facing, arm_conf, palm_conf
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
                    machine.sign_changes = 0
                    return "none", 0.0
            return "none", 0.0

        if state == "confirmed":
            if is_moving:
                machine.stop_frames = 0
                # 如果姿势不再符合原手势类型，提前降级
                if machine.confirmed_gesture == "hailing" and not is_raised:
                    # 高举变平伸，可能变成 greeting
                    if is_forward:
                        machine.confirmed_gesture = "greeting"
                    else:
                        # 姿势不符合任何手势，开始衰减
                        decay = max(0.3, 1.0 - machine.frames_in_state * 0.04)
                        machine.frames_in_state += 1
                        if machine.confirmed_gesture:
                            return machine.confirmed_gesture, machine.peak_confidence * decay
                        return "none", 0.0
                elif machine.confirmed_gesture == "greeting" and not is_forward and not is_raised:
                    decay = max(0.3, 1.0 - machine.frames_in_state * 0.04)
                    machine.frames_in_state += 1
                    if machine.confirmed_gesture:
                        return machine.confirmed_gesture, machine.peak_confidence * decay
                    return "none", 0.0

                decay = max(0.5, 1.0 - machine.frames_in_state * 0.025)
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
                    machine.sign_changes = 0
                    machine.confirmed_gesture = None
                    return "none", 0.0
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
        is_forward: bool,
        palm_facing: bool,
        arm_conf: float,
        palm_conf: float,
    ) -> Tuple[str, float]:
        """
        根据手臂姿势、周期性方向、手掌朝向综合判定 greeting vs hailing。
        """
        # 统计方向历史中的主方向
        dirs = [d for d in machine.direction_history if d != "none"]
        if not dirs:
            return "hand_up", min(arm_conf * 0.5, 0.7)

        h_count = sum(1 for d in dirs if d == "horizontal")
        v_count = sum(1 for d in dirs if d == "vertical")
        d_count = sum(1 for d in dirs if d == "diagonal")
        total = h_count + v_count + d_count
        if total == 0:
            return "hand_up", min(arm_conf * 0.5, 0.7)

        h_ratio = h_count / total
        v_ratio = v_count / total

        # hailing: 垂直挥动为主 + 手臂高举
        if v_ratio >= 0.55 and is_raised:
            conf = min(1.0, 0.55 + v_ratio * 0.25 + arm_conf * 0.2)
            if palm_facing:
                conf = min(1.0, conf + 0.1)
            return "hailing", conf

        # greeting: 水平挥动为主 + 手臂平伸（或高举但水平挥动）
        if h_ratio >= 0.55 and (is_forward or is_raised):
            conf = min(1.0, 0.55 + h_ratio * 0.25 + arm_conf * 0.2)
            if palm_facing:
                conf = min(1.0, conf + 0.08)
            return "greeting", conf

        # diagonal 为主时，根据手臂姿势判断
        if d_count / total >= 0.5:
            if is_raised:
                return "hailing", min(1.0, 0.6 + arm_conf * 0.2)
            elif is_forward:
                return "greeting", min(1.0, 0.6 + arm_conf * 0.2)

        # fallback
        return "hand_up", min(arm_conf * 0.5, 0.7)

    def reset(self) -> None:
        """重置所有状态机。"""
        for m in self._machines.values():
            m.periodic_detector.reset()
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
