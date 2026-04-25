import mediapipe as mp
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

@dataclass
class GestureResult:
    is_waving: bool
    confidence: float
    hand_landmarks: List[Tuple[float, float]]
    gesture_type: str  # "wave", "open_palm", "none"
    # 细分样式，便于调试；两种均 is_waving=True、gesture_type="wave"
    wave_style: Optional[str] = None  # "greeting" | "hail" | None

class GestureRecognizer:
    def __init__(self, 
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 min_frames_for_wave: int = 6,
                 history_size: int = 22,
                 finger_extend_ratio: float = 1.22,
                 min_open_fingers: int = 3,
                 bbox_quantize_px: int = 112,
                 # 打招呼式：腕部小幅度、较快摆动（ROI 归一化坐标）
                 greeting_range_min: float = 0.09,
                 greeting_range_max: float = 0.44,
                 greeting_rev_total_min: int = 2,
                 # 打车式：肘/肩带动、大幅度、可较慢（上下或侧向在 ROI 内体现为较大位移）
                 hail_range_min: float = 0.20,
                 hail_rev_max_min: int = 1):
        """
        bbox_quantize_px: 人体框量化步长，过小会导致 YOLO 框抖动时换轨、历史无法累积。
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.min_frames_for_wave = min_frames_for_wave
        self.history_size = history_size
        self.finger_extend_ratio = finger_extend_ratio
        self.min_open_fingers = min_open_fingers
        self._bbox_quantize_px = max(32, bbox_quantize_px)
        self.greeting_range_min = greeting_range_min
        self.greeting_range_max = greeting_range_max
        self.greeting_rev_total_min = greeting_rev_total_min
        self.hail_range_min = hail_range_min
        self.hail_rev_max_min = hail_rev_max_min
        # 手腕 (x,y) 序列，按人体轨道分轨
        self._histories: Dict[str, List[Tuple[float, float]]] = {}

    def _track_key(self, bbox: Optional[List[int]], track_index: Optional[int] = None) -> str:
        if not bbox or len(bbox) != 4:
            return f"t{track_index}" if track_index is not None else "_0"
        x1, y1, x2, y2 = bbox
        q = self._bbox_quantize_px
        bucket = f"{x1 // q}_{y1 // q}_{x2 // q}_{y2 // q}"
        if track_index is not None:
            return f"{track_index}_{bucket}"
        return bucket
        
    def recognize(
        self,
        roi: np.ndarray,
        bbox: Optional[List[int]] = None,
        track_index: Optional[int] = None,
    ) -> GestureResult:
        """
        识别 ROI 内招手：手掌张开 +（打招呼式小快摆 或 打车式大慢摆），二者均记为 wave。
        """
        if roi is None or roi.size == 0:
            return GestureResult(False, 0.0, [], "none", None)
        
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.hands.process(roi_rgb)
        
        if not results.multi_hand_landmarks:
            return GestureResult(False, 0.0, [], "none", None)
        
        hand_landmarks = results.multi_hand_landmarks[0]
        
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append((lm.x, lm.y))
        
        key = self._track_key(bbox, track_index=track_index)
        is_waving, confidence, gesture_type, wave_style = self._analyze_wave(landmarks, key)
        
        return GestureResult(
            is_waving=is_waving,
            confidence=confidence,
            hand_landmarks=landmarks,
            gesture_type=gesture_type,
            wave_style=wave_style,
        )
    
    def _analyze_wave(
        self, landmarks: List[Tuple[float, float]], track_key: str
    ) -> Tuple[bool, float, str, Optional[str]]:
        """
        打招呼式：max(x_range,y_range) 落在较小区间 + 轴向反转次数较多（腕部快摆）。
        打车式：max(x_range,y_range) 足够大 + 至少一条轴上有来回（可较慢、次数少）。
        """
        wrist = landmarks[0]
        
        finger_tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        finger_mcps = [landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
        
        open_fingers = 0
        for tip, mcp in zip(finger_tips, finger_mcps):
            if self._distance(tip, wrist) > self._distance(mcp, wrist) * self.finger_extend_ratio:
                open_fingers += 1
        
        palm_open = open_fingers >= self.min_open_fingers
        
        hist = self._histories.setdefault(track_key, [])
        hist.append((wrist[0], wrist[1]))
        if len(hist) > self.history_size:
            hist.pop(0)
        
        if len(hist) < self.min_frames_for_wave or not palm_open:
            gesture = "open_palm" if palm_open else "none"
            return False, 0.0, gesture, None
        
        xs = [p[0] for p in hist]
        ys = [p[1] for p in hist]
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        rng = max(x_range, y_range)
        
        rev_x = self._count_velocity_reversals(xs)
        rev_y = self._count_velocity_reversals(ys)
        rev_total = rev_x + rev_y
        rev_max = max(rev_x, rev_y)
        
        # 打招呼式：小～中等幅度、双轴累计反转较多
        greeting = (
            self.greeting_range_min <= rng <= self.greeting_range_max
            and rev_total >= self.greeting_rev_total_min
        )
        # 打车式：大幅度、任一条轴上至少一次方向反转（慢挥也可能窗内仅 1 次反转）
        hail = rng >= self.hail_range_min and rev_max >= self.hail_rev_max_min
        
        wave_detected = greeting or hail
        wave_style: Optional[str] = None
        if wave_detected:
            if greeting and hail:
                wave_style = "mixed"
            elif greeting:
                wave_style = "greeting"
            else:
                wave_style = "hail"
        
        wave_confidence = 0.0
        if wave_detected:
            wave_confidence = self._compute_wave_confidence(
                rng=rng,
                rev_total=rev_total,
                rev_max=rev_max,
                open_fingers=open_fingers,
                wave_style=wave_style,
            )
        
        gesture = "wave" if wave_detected else ("open_palm" if palm_open else "none")
        
        return wave_detected, wave_confidence, gesture, wave_style

    def _compute_wave_confidence(
        self,
        rng: float,
        rev_total: int,
        rev_max: int,
        open_fingers: int,
        wave_style: Optional[str],
    ) -> float:
        """
        置信度 ∈ [0.48, 0.97]：由幅度、节奏（反转）、张掌程度加权，并按样式微调。
        边缘通过阈值时偏低，动作清晰时偏高，避免常出现 0.99+。
        """
        # 幅度：在 ROI 归一化坐标下，约 0.08～0.45 为常见有效区间
        motion = min(1.0, max(0.0, (rng - 0.07) / 0.38))
        # 节奏：总反转与单轴反转结合，过高饱和（防抖动刷分）
        rhythm = min(
            1.0,
            max(0.0, (rev_total - 1.5) / 5.0) * 0.55 + min(1.0, rev_max / 3.5) * 0.45,
        )
        # 张掌：3～4 指为常见，4 指略高
        palm = min(1.0, max(0.35, open_fingers / 4.0))

        w = 0.40 * motion + 0.34 * rhythm + 0.26 * palm

        if wave_style == "hail":
            w += min(0.09, max(0.0, (rng - self.hail_range_min) / 0.32) * 0.09)
        elif wave_style in ("greeting", "mixed"):
            extra_rev = max(0, rev_total - self.greeting_rev_total_min)
            w += min(0.07, extra_rev / 5.0 * 0.07)

        if wave_style == "mixed":
            w += 0.03

        return float(max(0.48, min(0.97, w)))
    
    @staticmethod
    def _count_velocity_reversals(xs: List[float]) -> int:
        if len(xs) < 3:
            return 0
        v = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        rev = 0
        for i in range(len(v) - 1):
            if v[i] == 0 or v[i + 1] == 0:
                continue
            if v[i] * v[i + 1] < 0:
                rev += 1
        return rev
    
    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    def reset_history(self):
        self._histories.clear()
