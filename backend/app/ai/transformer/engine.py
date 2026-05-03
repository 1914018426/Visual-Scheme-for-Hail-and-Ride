"""
TransformerGestureEngine — lightweight inference engine that replaces
the rule-based TripleLockEngine with a trained TemporalKeypointTransformer.

Maintains per-track per-side sequence buffers and runs transformer inference
when sufficient frames have accumulated.
"""

import logging
import time
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

N_FEATURES = 12
DEFAULT_SEQ_LEN = 45  # 3 seconds @ 15 fps
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


class TransformerGestureEngine:
    """
    Transformer-based gesture recognition engine.

    Usage:
        engine = TransformerGestureEngine("waving_transformer.pt", device="cuda")
        gesture, confidence = engine.process_frame(
            track_id="person_1",
            side="right",
            wrist_local=np.array([0.2, -0.3]),
            velocity_mag=0.15,
            theta1=45.0,
            theta2=30.0,
            ext_ratio=0.7,
            palm_normal=np.array([0.3, 0.2, 0.93]),
            tnlf_valid=True,
            timestamp=time.time(),
        )
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        seq_len: int = DEFAULT_SEQ_LEN,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        ema_alpha: float = 0.35,
        hold_frames: int = 15,
    ):
        self.seq_len = seq_len
        self.confidence_threshold = confidence_threshold
        self.ema_alpha = ema_alpha
        self.hold_frames = hold_frames

        # Per (track_id, side) sequence buffer
        self._buffers: Dict[str, deque] = {}
        # Per (track_id, side) last inference result
        self._last_result: Dict[str, Tuple[str, float, float]] = {}
        # Per (track_id, side) EMA-smoothed confidence
        self._ema_conf: Dict[str, float] = {}
        # Per (track_id, side) hold counter (frames remaining after confirmation)
        self._hold_counter: Dict[str, int] = {}
        # Track buffer last-update timestamps for GC
        self._last_update: Dict[str, float] = {}

        self._device = device

        # Lazy-loaded model
        self._model = None
        self._model_path = model_path
        self._model_loaded = False

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    def _ensure_model(self):
        """Lazy-load the TorchScript model."""
        if self._model_loaded:
            return
        try:
            import torch
            self._model = torch.jit.load(self._model_path, map_location=self._device)
            self._model.eval()
            self._model_loaded = True
            logger.info(f"Transformer model loaded from {self._model_path}")
        except FileNotFoundError:
            logger.warning(
                f"Transformer model not found at {self._model_path}. "
                "Falling back to TripleLockEngine."
            )
            self._model_loaded = False
        except Exception as e:
            logger.error(f"Failed to load transformer model: {e}")
            self._model_loaded = False

    def process_frame(
        self,
        track_id: str,
        side: str,
        wrist_local: np.ndarray,
        velocity_mag: float,
        theta1: float,
        theta2: float,
        ext_ratio: float,
        palm_normal: np.ndarray,
        tnlf_valid: bool,
        timestamp: float,
        wrist_local_other: Optional[np.ndarray] = None,
    ) -> Tuple[str, float]:
        """
        Process one frame for a given tracked person and body side.

        Returns:
            (gesture_type, confidence)  where gesture_type is "waving" or "none"
        """
        self._ensure_model()
        if not self._model_loaded:
            return "none", 0.0

        key = f"{track_id}_{side}"
        self._last_update[key] = timestamp

        # Build feature vector
        feat = np.zeros(N_FEATURES, dtype=np.float32)

        if side == "right":
            feat[2] = wrist_local[0] if wrist_local is not None else 0.0
            feat[3] = wrist_local[1] if wrist_local is not None else 0.0
            if wrist_local_other is not None:
                feat[0] = wrist_local_other[0]
                feat[1] = wrist_local_other[1]
        else:
            feat[0] = wrist_local[0] if wrist_local is not None else 0.0
            feat[1] = wrist_local[1] if wrist_local is not None else 0.0
            if wrist_local_other is not None:
                feat[2] = wrist_local_other[0]
                feat[3] = wrist_local_other[1]

        feat[4] = velocity_mag
        feat[5] = theta1
        feat[6] = theta2
        feat[7] = ext_ratio
        feat[8] = palm_normal[0] if palm_normal is not None else 0.0
        feat[9] = palm_normal[1] if palm_normal is not None else 0.0
        feat[10] = palm_normal[2] if palm_normal is not None else 0.0
        feat[11] = 1.0 if tnlf_valid else 0.0

        # Append to buffer
        if key not in self._buffers:
            self._buffers[key] = deque(maxlen=self.seq_len)
        self._buffers[key].append(feat)

        # Check hold counter (after confirmation, maintain output for N frames)
        if key in self._hold_counter and self._hold_counter[key] > 0:
            self._hold_counter[key] -= 1
            if key in self._last_result:
                gesture, peak_conf, _ = self._last_result[key]
                decay = self._hold_counter[key] / max(self.hold_frames, 1)
                return gesture, peak_conf * decay

        # Need full buffer to run inference
        if len(self._buffers[key]) < self.seq_len:
            return "none", 0.0

        # Run inference
        try:
            import torch
            buf_array = np.array(list(self._buffers[key]), dtype=np.float32)
            seq = torch.as_tensor(buf_array, device=self._device).unsqueeze(0)
            with torch.no_grad():
                raw_conf = float(self._model(seq).item())
        except Exception as e:
            logger.error(f"Transformer inference error for {key}: {e}")
            return "none", 0.0

        # EMA smoothing
        if key not in self._ema_conf:
            self._ema_conf[key] = raw_conf
        else:
            self._ema_conf[key] = (
                self.ema_alpha * raw_conf + (1 - self.ema_alpha) * self._ema_conf[key]
            )
        smoothed_conf = self._ema_conf[key]

        # Decision
        if smoothed_conf >= self.confidence_threshold:
            self._hold_counter[key] = self.hold_frames
            self._last_result[key] = ("waving", smoothed_conf, timestamp)
            return "waving", smoothed_conf
        else:
            # Clear hold on negative decision
            self._hold_counter.pop(key, None)
            return "none", smoothed_conf

    def cleanup_stale(self, active_keys: set, max_age_seconds: float = 10.0):
        """
        Remove buffers for track_ids that are no longer active.

        Args:
            active_keys: Set of "{track_id}_{side}" keys currently in frame
            max_age_seconds: Max age before forced GC
        """
        now = time.time()
        stale = []
        for key in list(self._buffers.keys()):
            if key not in active_keys:
                last_ts = self._last_update.get(key, 0)
                if now - last_ts > max_age_seconds:
                    stale.append(key)

        for key in stale:
            self._buffers.pop(key, None)
            self._last_result.pop(key, None)
            self._ema_conf.pop(key, None)
            self._hold_counter.pop(key, None)
            self._last_update.pop(key, None)

        if stale:
            logger.debug(f"GC'd {len(stale)} stale transformer buffers")

    def reset_track(self, track_id: str, side: str):
        """Reset buffer for a specific track (e.g., on tracking loss)."""
        key = f"{track_id}_{side}"
        self._buffers.pop(key, None)
        self._last_result.pop(key, None)
        self._ema_conf.pop(key, None)
        self._hold_counter.pop(key, None)
        self._last_update.pop(key, None)

    def reset_all(self):
        """Clear all state (e.g., on model reload)."""
        self._buffers.clear()
        self._last_result.clear()
        self._ema_conf.clear()
        self._hold_counter.clear()
        self._last_update.clear()

    @property
    def buffer_count(self) -> int:
        return len(self._buffers)

    @property
    def total_buffer_size(self) -> int:
        return sum(len(buf) for buf in self._buffers.values())
