"""SGN ONNX temporal judge for waving detection."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# COCO17 body: shoulders/elbows/wrists — waving 判定的关键关节下标
SGN_CRITICAL_JOINT_INDICES = (5, 6, 7, 8, 9, 10)


class SGNTemporalJudge:
    """Run SGN ONNX model as a temporal gesture classifier."""

    def __init__(
        self,
        model_path: str,
        seq_len: int = 30,
        conf_threshold: float = 0.6,
        expected_nodes: int = 21,
        min_kpt_conf: float = 0.25,
        smooth_alpha: float = 0.6,
        motion_gate_min_amp: float = 0.18,
        motion_gate_min_flips: int = 2,
    ) -> None:
        self.seq_len = max(4, seq_len)
        self.conf_threshold = float(conf_threshold)
        self.expected_nodes = max(1, expected_nodes)
        self.min_kpt_conf = float(min_kpt_conf)
        self.smooth_alpha = float(smooth_alpha)
        self.motion_gate_min_amp = float(motion_gate_min_amp)
        self.motion_gate_min_flips = int(motion_gate_min_flips)
        self._buffers: Dict[str, Deque[np.ndarray]] = {}
        self._last_frame: Dict[str, np.ndarray] = {}
        self._enabled = False
        self._ort = None
        self._session = None
        self._input_name = ""

        try:
            import onnxruntime as ort

            self._ort = ort
            self._session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            self._enabled = True
            logger.info(
                "SGN judge enabled: model=%s seq_len=%d nodes=%d",
                model_path,
                self.seq_len,
                self.expected_nodes,
            )
        except Exception as exc:
            logger.warning("SGN judge disabled, model load failed: %s", exc)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _adapt_nodes(self, track_id: str, keypoints: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Adapt incoming keypoints to model-required node count.

        Returns:
            Processed tensor [N, 3] and per-frame debug stats for telemetry.
        """
        meta: Dict[str, Any] = {}
        if keypoints.ndim != 2 or keypoints.shape[1] < 3:
            meta["sgn_raw_visible_ratio"] = 0.0
            meta["sgn_critical_visible_ratio"] = 0.0
            meta["sgn_low_conf_nodes"] = self.expected_nodes
            meta["sgn_imputed_nodes"] = 0
            return np.zeros((self.expected_nodes, 3), dtype=np.float32), meta

        src = keypoints[:, :3].astype(np.float32, copy=False)
        out = np.zeros((self.expected_nodes, 3), dtype=np.float32)
        n = min(src.shape[0], self.expected_nodes)
        out[:n] = src[:n]

        raw_miss = np.ones(self.expected_nodes, dtype=bool)
        raw_miss[:n] = src[:n, 2] < self.min_kpt_conf
        low_conf = int(np.sum(raw_miss))
        visible_ratio = float(np.mean(~raw_miss))
        crit_idx = [i for i in SGN_CRITICAL_JOINT_INDICES if i < self.expected_nodes]
        critical_vis = (
            float(np.mean(~raw_miss[crit_idx])) if crit_idx else 0.0
        )

        prev = self._last_frame.get(track_id)
        imputed = 0
        if prev is not None and prev.shape == out.shape:
            miss = raw_miss.copy()
            imputed = int(np.sum(miss))
            # Impute missing/low-confidence keypoints from previous frame.
            out[miss, :2] = prev[miss, :2]
            # Keep low confidence instead of force-high to let model know quality.
            out[miss, 2] = np.maximum(out[miss, 2], prev[miss, 2] * 0.5)
            # Smooth visible points to reduce jitter.
            vis = ~miss
            out[vis, :2] = (
                self.smooth_alpha * out[vis, :2]
                + (1.0 - self.smooth_alpha) * prev[vis, :2]
            )

        raw_out = out.copy()
        out = self._normalize_for_model(raw_out)
        self._last_frame[track_id] = raw_out
        meta["sgn_raw_visible_ratio"] = round(visible_ratio, 4)
        meta["sgn_critical_visible_ratio"] = round(critical_vis, 4)
        meta["sgn_low_conf_nodes"] = low_conf
        meta["sgn_imputed_nodes"] = imputed
        return out, meta

    def _normalize_for_model(self, nodes: np.ndarray) -> np.ndarray:
        """Normalize to body-local coordinate system to match training domain."""
        out = nodes.copy()
        if out.shape[0] < 7:
            return out

        conf = out[:, 2]
        center = None
        scale = None

        # Prefer shoulder center/width (COCO17).
        if out.shape[0] > 6 and conf[5] >= self.min_kpt_conf and conf[6] >= self.min_kpt_conf:
            center = (out[5, :2] + out[6, :2]) * 0.5
            scale = float(np.linalg.norm(out[6, :2] - out[5, :2]))
        # Fallback to hip center/width.
        elif out.shape[0] > 12 and conf[11] >= self.min_kpt_conf and conf[12] >= self.min_kpt_conf:
            center = (out[11, :2] + out[12, :2]) * 0.5
            scale = float(np.linalg.norm(out[12, :2] - out[11, :2]))
        else:
            vis = conf >= self.min_kpt_conf
            if np.any(vis):
                xy = out[vis, :2]
                center = np.mean(xy, axis=0)
                span = np.max(xy, axis=0) - np.min(xy, axis=0)
                scale = float(max(span[0], span[1]))
            else:
                center = np.array([0.0, 0.0], dtype=np.float32)
                scale = 1.0

        scale = max(scale if scale is not None else 1.0, 1.0)
        out[:, :2] = (out[:, :2] - center) / scale
        return out

    def _motion_gate(self, seq: np.ndarray) -> Dict[str, Any]:
        """Conservative waving gate using wrist motion over temporal window."""
        if seq.ndim != 3 or seq.shape[1] <= 10:
            return {
                "sgn_motion_amp": 0.0,
                "sgn_motion_flips": 0,
                "sgn_motion_gate_pass": False,
            }

        left_wrist = seq[:, 9, :]
        right_wrist = seq[:, 10, :]
        left_shoulder = seq[:, 5, :]
        right_shoulder = seq[:, 6, :]

        shoulder_width = np.abs(right_shoulder[:, 0] - left_shoulder[:, 0])
        scale = float(np.median(shoulder_width[shoulder_width > 1e-6])) if np.any(shoulder_width > 1e-6) else 1.0
        scale = max(scale, 1.0)

        def wrist_stats(w: np.ndarray) -> Tuple[float, int]:
            vis = w[:, 2] >= self.min_kpt_conf
            if int(np.sum(vis)) < 4:
                return 0.0, 0
            x = w[:, 0].astype(np.float32)
            xv = x[vis]
            amp = float((np.max(xv) - np.min(xv)) / scale)
            dx = np.diff(xv)
            eps = 0.015 * scale
            signs = np.sign(dx[np.abs(dx) > eps])
            if len(signs) < 2:
                return amp, 0
            flips = int(np.sum(signs[1:] * signs[:-1] < 0))
            return amp, flips

        l_amp, l_flips = wrist_stats(left_wrist)
        r_amp, r_flips = wrist_stats(right_wrist)
        amp = max(l_amp, r_amp)
        flips = max(l_flips, r_flips)
        passed = amp >= self.motion_gate_min_amp and flips >= self.motion_gate_min_flips
        return {
            "sgn_motion_amp": round(float(amp), 4),
            "sgn_motion_flips": int(flips),
            "sgn_motion_gate_pass": bool(passed),
        }

    def infer(self, track_id: str, keypoints: np.ndarray) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Infer waving state for one track.

        Returns:
            (gesture, confidence, ready, debug_dict)
        """
        empty_debug: Dict[str, Any] = {
            "sgn_source": "sgn",
            "sgn_ready": False,
            "sgn_buffer_len": 0,
            "sgn_seq_len": self.seq_len,
            "sgn_raw_visible_ratio": 0.0,
            "sgn_critical_visible_ratio": 0.0,
            "sgn_low_conf_nodes": self.expected_nodes,
            "sgn_imputed_nodes": 0,
            "sgn_pos_prob": 0.0,
            "sgn_motion_amp": 0.0,
            "sgn_motion_flips": 0,
            "sgn_motion_gate_pass": False,
        }
        if not self._enabled or self._session is None:
            return "none", 0.0, False, empty_debug

        buf = self._buffers.get(track_id)
        if buf is None:
            buf = deque(maxlen=self.seq_len)
            self._buffers[track_id] = buf

        adapted, frame_meta = self._adapt_nodes(track_id, keypoints)
        buf.append(adapted)

        buf_len = len(buf)
        ready = buf_len >= self.seq_len
        debug: Dict[str, Any] = {
            "sgn_source": "sgn",
            "sgn_ready": ready,
            "sgn_buffer_len": buf_len,
            "sgn_seq_len": self.seq_len,
            "sgn_pos_prob": 0.0,
            "sgn_motion_amp": 0.0,
            "sgn_motion_flips": 0,
            "sgn_motion_gate_pass": False,
            **frame_meta,
        }

        if not ready:
            return "none", 0.0, False, debug

        x = np.stack(list(buf), axis=0).astype(np.float32)  # [T, N, 3]
        x = np.expand_dims(x, axis=0)  # [1, T, N, 3]

        try:
            logits = self._session.run(None, {self._input_name: x})[0]
            logits = np.asarray(logits, dtype=np.float32)
            if logits.ndim == 1:
                logits = logits[np.newaxis, :]
            if logits.shape[-1] < 2:
                return "none", 0.0, True, debug
            vec = logits[0]
            exp = np.exp(vec - np.max(vec))
            probs = exp / (np.sum(exp) + 1e-8)
            pos_conf = float(probs[1])
            debug["sgn_pos_prob"] = round(pos_conf, 4)
            motion_meta = self._motion_gate(np.stack(list(buf), axis=0))
            debug.update(motion_meta)
            gesture = "none"
            if pos_conf >= self.conf_threshold and motion_meta["sgn_motion_gate_pass"]:
                gesture = "waving"
            return gesture, pos_conf, True, debug
        except Exception as exc:
            logger.warning("SGN inference failed: %s", exc)
            return "none", 0.0, True, debug
