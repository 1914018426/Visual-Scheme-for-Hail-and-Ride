"""
Real-world dataset downloader + TNLF feature extractor for transformer training.

Datasets downloaded via HuggingFace (hf-mirror.com):
- HMDB51 (MichiganNLP/hmdb): "wave" class (~100 clips) + negative classes
- UCF101 (quchenyuan/UCF101-ZIP): walking/standing/running classes (negatives)

Usage:
    # First download datasets:
    #   HF_ENDPOINT=https://hf-mirror.com hf download MichiganNLP/hmdb --repo-type dataset --local-dir datasets/hmdb51
    #   HF_ENDPOINT=https://hf-mirror.com hf download quchenyuan/UCF101-ZIP --repo-type dataset --local-dir datasets/ucf101
    #
    # Then process:
    #   python3 real_data_pipeline.py --datasets hmdb51,ucf101 --data_dir ../../datasets --output_dir ../../data
"""

import argparse
import glob
import json
import math
import os
import random
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import cv2

# TNLF feature indices
N_FEATURES = 12
F_WLX_L, F_WLY_L = 0, 1
F_WLX_R, F_WLY_R = 2, 3
F_VEL_MAG = 4
F_THETA1 = 5
F_THETA2 = 6
F_EXT_RATIO = 7
F_PN_X, F_PN_Y, F_PN_Z = 8, 9, 10
F_VALID = 11

DEFAULT_SEQ_LEN = 45
DEFAULT_FPS = 15

# ============================================================================
# Dataset Extraction (from HuggingFace-downloaded ZIP files)
# ============================================================================

# HMDB51 classes relevant for our task
HMDB51_WAVE_CLASS = "wave"
HMDB51_NEGATIVE_CLASSES = [
    "walk", "run", "stand", "sit", "turn", "talk", "smile", "laugh",
    "chew", "eat", "drink", "climb", "climb_stairs", "jump", "pullup",
    "push", "pull", "pick", "carry", "throw", "catch", "kick", "punch",
    "swing_baseball", "handstand", "cartwheel", "shake_hands", "hug",
]


def extract_zip(zip_path: str, dest_dir: str, desc: str = "") -> bool:
    """Extract a ZIP file to dest_dir. Returns True on success."""
    if not os.path.exists(zip_path):
        print(f"  [{desc}] ZIP not found: {zip_path}")
        return False

    os.makedirs(dest_dir, exist_ok=True)
    print(f"  [{desc}] Extracting {os.path.getsize(zip_path) / 1e6:.1f} MB...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        print(f"  [{desc}] Extraction complete")
        return True
    except Exception as e:
        print(f"  [{desc}] Extraction failed: {e}")
        return False


def extract_hmdb51(datasets_dir: str) -> Optional[str]:
    """Extract HMDB51 from HF-downloaded ZIP file."""
    hmdb_zip = os.path.join(datasets_dir, "hmdb51", "hmdb51_org.zip")
    hmdb_dir = os.path.join(datasets_dir, "hmdb51_extracted")

    if os.path.isdir(os.path.join(hmdb_dir, "wave")):
        print("[HMDB51] Already extracted")
        return hmdb_dir

    if not os.path.exists(hmdb_zip):
        print(f"[HMDB51] ZIP not found at {hmdb_zip}. "
              "Download first: hf download MichiganNLP/hmdb --repo-type dataset --local-dir datasets/hmdb51")
        return None

    if not extract_zip(hmdb_zip, hmdb_dir, "HMDB51"):
        return None

    return hmdb_dir


def extract_ucf101(datasets_dir: str) -> Optional[str]:
    """Extract UCF101 from HF-downloaded ZIP file."""
    ucf_zip = os.path.join(datasets_dir, "ucf101", "UCF-101.zip")
    ucf_dir = os.path.join(datasets_dir, "ucf101_extracted")

    if os.path.isdir(os.path.join(ucf_dir, "ApplyEyeMakeup")):
        print("[UCF101] Already extracted")
        return ucf_dir

    if not os.path.exists(ucf_zip):
        print(f"[UCF101] ZIP not found at {ucf_zip}. "
              "Download first: hf download quchenyuan/UCF101-ZIP --repo-type dataset --local-dir datasets/ucf101")
        return None

    if not extract_zip(ucf_zip, ucf_dir, "UCF101"):
        return None

    return ucf_dir


# ============================================================================
# Video to TNLF Feature Extractor
# ============================================================================

class VideoToTNLFProcessor:
    """
    Extracts TNLF features from video clips using YOLO11-Pose.
    """

    def __init__(self, yolo_model_path: str = "yolo11x-pose.pt",
                 device: str = "cuda", conf_threshold: float = 0.25):
        self.device = device
        self.conf_threshold = conf_threshold  # e.g. 0.25 for low-res HMDB51
        self.model = None
        self.model_path = yolo_model_path

    def _load_model(self):
        if self.model is not None:
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        except ImportError:
            print("[WARN] ultralytics not installed. Install: pip install ultralytics")
            raise

    def extract_features(
        self,
        video_path: str,
        seq_len: int = DEFAULT_SEQ_LEN,
        target_fps: float = DEFAULT_FPS,
    ) -> List[np.ndarray]:
        """
        Extract TNLF feature sequences from a video.

        For short clips (like HMDB51 ~3 sec), extracts features per-frame from
        the most-prominent detected person, without complex tracking.

        Returns:
            List of [T, N_FEATURES] arrays (typically 1 per video for short clips)
        """
        self._load_model()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if src_fps <= 0:
            src_fps = 30.0

        stride = max(1, int(src_fps / target_fps))
        frame_count = 0
        all_feats = []  # List of feature vectors per frame

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % stride != 0:
                frame_count += 1
                continue
            frame_count += 1

            results = self.model(frame, conf=self.conf_threshold, verbose=False,
                                device=self.device, half=True)

            if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                # Pick the person with most valid keypoints (likely the subject)
                best_kpts, best_valid = None, 0
                for kpt_data in results[0].keypoints.data:
                    kpts = kpt_data.cpu().numpy()
                    n_valid = int((kpts[:, 2] > 0.2).sum())
                    if n_valid > best_valid and kpts.shape[0] >= 17:
                        best_valid = n_valid
                        best_kpts = kpts

                if best_kpts is not None and best_valid >= 2:
                    feat = self._compute_tnlf_features(best_kpts)
                    if feat is not None:
                        all_feats.append(feat)
                    else:
                        all_feats.append(np.zeros(N_FEATURES, dtype=np.float32))
                else:
                    all_feats.append(None)
            else:
                all_feats.append(None)

        cap.release()

        if not all_feats:
            return []

        # Fill gaps (up to 3 consecutive missing frames) with interpolation
        feats_filled = []
        i = 0
        while i < len(all_feats):
            if all_feats[i] is not None:
                feats_filled.append(all_feats[i])
                i += 1
            else:
                # Look for next valid frame within 3 steps
                gap_end = i
                for j in range(i, min(i + 4, len(all_feats))):
                    if all_feats[j] is not None:
                        gap_end = j
                        break
                if gap_end > i and all_feats[gap_end] is not None:
                    # Linear interpolate
                    prev = feats_filled[-1] if feats_filled else np.zeros(N_FEATURES, dtype=np.float32)
                    nxt = all_feats[gap_end]
                    for k in range(i, gap_end):
                        alpha = (k - i + 1) / (gap_end - i + 1)
                        feats_filled.append(prev * (1 - alpha) + nxt * alpha)
                    feats_filled.append(nxt)
                    i = gap_end + 1
                else:
                    i += 1

        if len(feats_filled) < 15:
            return []

        feat_array = np.array(feats_filled, dtype=np.float32)
        T = feat_array.shape[0]

        # Velocity (active arm — pick the side with smaller mean y_local over the whole track)
        mean_wly_l = float(np.mean(feat_array[:, F_WLY_L]))
        mean_wly_r = float(np.mean(feat_array[:, F_WLY_R]))
        if mean_wly_l < mean_wly_r:
            wlx_idx, wly_idx = F_WLX_L, F_WLY_L
        else:
            wlx_idx, wly_idx = F_WLX_R, F_WLY_R

        for t in range(1, T):
            wl_curr = np.array([feat_array[t, wlx_idx], feat_array[t, wly_idx]])
            wl_prev = np.array([feat_array[t-1, wlx_idx], feat_array[t-1, wly_idx]])
            feat_array[t, F_VEL_MAG] = float(np.linalg.norm(wl_curr - wl_prev) / (1.0 / target_fps))

        # Build sequences via sliding window
        sequences = []
        if T >= seq_len:
            for start in range(0, T - seq_len + 1, max(1, seq_len // 4)):
                sequences.append(feat_array[start:start + seq_len].copy())
        else:
            # Pad to seq_len
            pad_len = seq_len - T
            pad = np.tile(feat_array[-1:], (pad_len, 1))
            sequences.append(np.concatenate([feat_array, pad], axis=0))

        return sequences

    def _track_and_extract(
        self,
        frames_kpts: List[List[np.ndarray]],
        fps: float,
    ) -> List[np.ndarray]:
        """
        Simple IoU-based tracking + TNLF feature extraction.
        Returns list of [T, N_FEATURES] per tracked person.
        """
        if not frames_kpts:
            return []

        dt = 1.0 / fps
        active_tracks: Dict[int, Dict] = {}  # track_id -> {kpts, last_bbox, features}
        next_track_id = 0
        all_tracks: List[List[np.ndarray]] = []

        for frame_idx, persons in enumerate(frames_kpts):
            matched = set()

            for p_kpts in persons:
                # Compute bbox from keypoints
                valid_kpts = p_kpts[p_kpts[:, 2] > 0.2]
                if len(valid_kpts) < 2:
                    continue
                bbox = np.array([
                    valid_kpts[:, 0].min(), valid_kpts[:, 1].min(),
                    valid_kpts[:, 0].max(), valid_kpts[:, 1].max(),
                ])

                # Match to existing track
                best_iou, best_tid = 0.0, -1
                for tid, track in active_tracks.items():
                    if tid in matched:
                        continue
                    iou = self._box_iou(bbox, track["last_bbox"])
                    if iou > best_iou:
                        best_iou, best_tid = iou, tid

                if best_iou > 0.3:
                    tid = best_tid
                    matched.add(tid)
                else:
                    tid = next_track_id
                    next_track_id += 1
                    active_tracks[tid] = {
                        "features": [],
                        "last_wl": None,
                        "last_ts": frame_idx * dt,
                    }

                # Compute TNLF features
                feat = self._compute_tnlf_features(p_kpts)
                if feat is not None:
                    # Velocity
                    track = active_tracks[tid]
                    wl_curr = np.array([feat[F_WLX_R], feat[F_WLY_R]])
                    if track["last_wl"] is not None:
                        td = frame_idx * dt - track["last_ts"]
                        if td > 0.001:
                            v = np.linalg.norm(wl_curr - track["last_wl"]) / td
                        else:
                            v = 0.0
                        feat[F_VEL_MAG] = v
                    track["last_wl"] = wl_curr
                    track["last_ts"] = frame_idx * dt
                    track["features"].append(feat)

                active_tracks[tid]["last_bbox"] = bbox

            # Handle tracks lost this frame — finalize if they have enough data
            for tid in list(active_tracks.keys()):
                track = active_tracks[tid]
                if tid not in matched:
                    # Track lost — finalize if enough features
                    feats = track["features"]
                    if len(feats) >= 15:
                        all_tracks.append(np.array(feats, dtype=np.float32))
                    del active_tracks[tid]
                # Also check if track is too long (> 10 seconds = 150 frames @ 15fps)
                elif len(track["features"]) > 150:
                    all_tracks.append(np.array(track["features"][:150], dtype=np.float32))
                    track["features"] = track["features"][-30:]  # Keep tail for continuity

        # Finalize remaining tracks
        for tid, track in active_tracks.items():
            feats = track["features"]
            if len(feats) >= 15:
                all_tracks.append(np.array(feats, dtype=np.float32))

        return all_tracks

    def _compute_tnlf_features(self, kpts: np.ndarray) -> Optional[np.ndarray]:
        """Compute 12-dim TNLF feature vector from COCO-17 keypoints."""
        feat = np.zeros(N_FEATURES, dtype=np.float32)

        shoulder_mid = (kpts[5, :2] + kpts[6, :2]) / 2.0
        hip_mid = (kpts[11, :2] + kpts[12, :2]) / 2.0

        e_x = kpts[6, :2] - kpts[5, :2]
        e_x_norm = float(np.linalg.norm(e_x))
        if e_x_norm < 1e-6:
            return None
        e_x = e_x / e_x_norm

        e_y = hip_mid - shoulder_mid  # 与推理 local_frame.py 保持一致：指向图像下方（hip 方向）
        torso_scale = float(np.linalg.norm(e_y))
        if torso_scale < 10.0:
            e_y = np.array([0.0, e_x_norm * 3.0])
            torso_scale = e_x_norm * 3.0
        e_y = e_y / (float(np.linalg.norm(e_y)) + 1e-8)

        # Left wrist TNLF
        wl = kpts[9, :2] - shoulder_mid
        feat[F_WLX_L] = float(np.dot(wl, e_x) / torso_scale)
        feat[F_WLY_L] = float(np.dot(wl, e_y) / torso_scale)

        # Right wrist TNLF
        wr = kpts[10, :2] - shoulder_mid
        feat[F_WLX_R] = float(np.dot(wr, e_x) / torso_scale)
        feat[F_WLY_R] = float(np.dot(wr, e_y) / torso_scale)

        # 选择活跃手臂：手腕在 TNLF 中 y 更小（更靠上）的一侧
        # （e_y 指向下方时，y_local 越小代表手腕越靠上）
        if feat[F_WLY_L] < feat[F_WLY_R]:
            s_idx, e_idx, w_idx, h_idx = 5, 7, 9, 11
        else:
            s_idx, e_idx, w_idx, h_idx = 6, 8, 10, 12

        s, e, w = kpts[s_idx, :2], kpts[e_idx, :2], kpts[w_idx, :2]
        h = kpts[h_idx, :2]
        v_se = e - s
        v_sh = h - s
        v_ew = w - e
        v_sw = w - s

        def angle(v1, v2):
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

        feat[F_THETA1] = angle(v_se, v_sh)
        feat[F_THETA2] = angle(-v_se, v_ew)
        feat[F_EXT_RATIO] = float(np.linalg.norm(v_sw) / (np.linalg.norm(v_se) + np.linalg.norm(v_ew) + 1e-8))

        # Palm normal proxy (forearm direction of active arm)
        forearm = kpts[w_idx, :2] - kpts[e_idx, :2]
        fn = float(np.linalg.norm(forearm)) + 1e-8
        feat[F_PN_X] = float(forearm[0] / fn)
        feat[F_PN_Y] = float(forearm[1] / fn)
        feat[F_PN_Z] = 0.5

        feat[F_VALID] = 1.0
        return feat

    def _box_iou(self, a: np.ndarray, b: np.ndarray) -> float:
        xa = max(a[0], b[0])
        ya = max(a[1], b[1])
        xb = min(a[2], b[2])
        yb = min(a[3], b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-8)


# ============================================================================
# Kinetics processing (YouTube blocked in some regions — skip if unavailable)
# ============================================================================

def has_kinetics_videos(kinetics_dir: str) -> bool:
    """Check if Kinetics videos were previously downloaded."""
    if not kinetics_dir or not os.path.isdir(kinetics_dir):
        return False
    return len(glob.glob(os.path.join(kinetics_dir, "*.mp4"))) > 0


def process_kinetics(kinetics_dir: str, output_dir: str, processor: VideoToTNLFProcessor,
                     label: int = 1) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Process Kinetics waving videos (all positive). Only used if videos exist."""
    X = []
    clips = sorted(glob.glob(os.path.join(kinetics_dir, "*.mp4")))
    print(f"[Kinetics] Processing {len(clips)} waving videos...")

    for i, clip in enumerate(clips):
        seqs = processor.extract_features(clip)
        for seq in seqs:
            X.append(seq)
        if i % 10 == 0 and i > 0:
            print(f"  Processed {i}/{len(clips)}, sequences: {len(X)}")

    print(f"[Kinetics] Total sequences: {len(X)}")
    return (X, []) if label == 1 else ([], X)


# ============================================================================
# Dataset Processing Orchestrator
# ============================================================================

def process_hmdb51(hmdb_dir: str, output_dir: str, processor: VideoToTNLFProcessor,
                   max_clips_per_class: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Process HMDB51 clips into TNLF features."""
    X_pos, X_neg = [], []

    # Positive: "wave" class
    wave_dir = os.path.join(hmdb_dir, HMDB51_WAVE_CLASS)
    if os.path.isdir(wave_dir):
        clips = sorted(glob.glob(os.path.join(wave_dir, "*.avi")))
        print(f"[HMDB51] Processing {min(len(clips), max_clips_per_class)} wave clips...")
        for clip in clips[:max_clips_per_class]:
            seqs = processor.extract_features(clip)
            for seq in seqs:
                X_pos.append(seq)
            if len(X_pos) % 20 == 0:
                print(f"  Positive sequences: {len(X_pos)}")

    # Negative: non-wave classes
    for cls_name in HMDB51_NEGATIVE_CLASSES:
        cls_dir = os.path.join(hmdb_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        clips = sorted(glob.glob(os.path.join(cls_dir, "*.avi")))
        for clip in clips[:max_clips_per_class // 2]:
            seqs = processor.extract_features(clip)
            for seq in seqs:
                X_neg.append(seq)

        if len(X_neg) % 50 == 0:
            print(f"  Negative sequences: {len(X_neg)}")

        # Balance: stop collecting negatives when we have ~2x positives
        if len(X_neg) >= len(X_pos) * 3:
            break

    print(f"[HMDB51] Positives: {len(X_pos)}, Negatives: {len(X_neg)}")
    return X_pos, X_neg


def process_ucf101(ucf_dir: str, output_dir: str, processor: VideoToTNLFProcessor,
                   max_clips: int = 200) -> np.ndarray:
    """Process UCF101 clips as negative samples only (no waving class)."""
    X_neg = []

    # Classes involving human motion (negative samples)
    neg_classes = [
        "Walking", "Running", "Standing", "Sitting", "Jumping",
        "PushUps", "PullUps", "HandStandPushups", "HandStandWalking",
        "Swing", "Punch", "Throw", "Catch",
    ]

    for cls_name in neg_classes:
        cls_dir = os.path.join(ucf_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        clips = sorted(glob.glob(os.path.join(cls_dir, "*.avi")))
        for clip in clips[:max_clips // len(neg_classes) + 1]:
            seqs = processor.extract_features(clip)
            for seq in seqs:
                X_neg.append(seq)

        if len(X_neg) % 50 == 0:
            print(f"[UCF101] Negatives: {len(X_neg)}")
        if len(X_neg) >= max_clips:
            break

    print(f"[UCF101] Total negatives: {len(X_neg)}")
    return X_neg


def _balance_and_pad(
    X_pos: List[np.ndarray],
    X_neg: List[np.ndarray],
    seq_len: int = DEFAULT_SEQ_LEN,
) -> Tuple[np.ndarray, np.ndarray]:
    """Balance classes and pad/truncate to fixed seq_len."""
    if not X_pos or not X_neg:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # Balance
    n = min(len(X_pos), len(X_neg))
    X_pos = random.sample(X_pos, n) if len(X_pos) > n else X_pos
    X_neg = random.sample(X_neg, n) if len(X_neg) > n else X_neg

    X_all = []
    y_all = []
    for x in X_pos:
        if x is None or not hasattr(x, 'shape') or len(x.shape) < 2:
            continue
        T = x.shape[0]
        if T < 15:
            continue
        if T > seq_len:
            start = random.randint(0, T - seq_len)
            x = x[start:start + seq_len]
        elif T < seq_len:
            pad = np.tile(x[-1:], (seq_len - T, 1))
            x = np.concatenate([x, pad], axis=0)
        X_all.append(x)
        y_all.append(1)

    for x in X_neg:
        if x is None or not hasattr(x, 'shape') or len(x.shape) < 2:
            continue
        T = x.shape[0]
        if T < 15:
            continue
        if T > seq_len:
            start = random.randint(0, T - seq_len)
            x = x[start:start + seq_len]
        elif T < seq_len:
            pad = np.tile(x[-1:], (seq_len - T, 1))
            x = np.concatenate([x, pad], axis=0)
        X_all.append(x)
        y_all.append(0)

    X_arr = np.array(X_all, dtype=np.float32)
    y_arr = np.array(y_all, dtype=np.int64)
    return X_arr, y_arr


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Process real gesture datasets into TNLF features")
    parser.add_argument("--datasets", type=str, default="hmdb51",
                        help="Comma-separated: hmdb51,ucf101,kinetics")
    parser.add_argument("--data_dir", type=str, default="../../datasets",
                        help="Directory with downloaded dataset ZIP files")
    parser.add_argument("--output_dir", type=str, default="../../data",
                        help="Output directory for processed .npz files")
    parser.add_argument("--yolo_model", type=str, default="yolo11x-pose.pt",
                        help="Path to YOLO11-pose model")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference")
    parser.add_argument("--max_videos", type=int, default=200,
                        help="Max videos per dataset/class")
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    processed_dir = os.path.join(os.path.abspath(args.output_dir), "processed")
    os.makedirs(processed_dir, exist_ok=True)

    datasets = [d.strip().lower() for d in args.datasets.split(",")]

    # Initialize processor
    processor = VideoToTNLFProcessor(
        yolo_model_path=args.yolo_model,
        device=args.device,
    )

    all_X_pos, all_X_neg = [], []

    # === HMDB51 ===
    if "hmdb51" in datasets:
        hmdb_dir = extract_hmdb51(data_dir)
        if hmdb_dir:
            X_p, X_n = process_hmdb51(hmdb_dir, processed_dir, processor,
                                       max_clips_per_class=args.max_videos)
            if len(X_p) > 0:
                all_X_pos.extend(X_p)
            if len(X_n) > 0:
                all_X_neg.extend(X_n)

    # === UCF101 ===
    if "ucf101" in datasets:
        ucf_dir = extract_ucf101(data_dir)
        if ucf_dir:
            X_n = process_ucf101(ucf_dir, processed_dir, processor,
                                 max_clips=args.max_videos)
            if len(X_n) > 0:
                all_X_neg.extend(X_n)

    # === Kinetics-700 (only if videos already exist — YouTube blocked in many regions) ===
    if "kinetics" in datasets:
        kinetics_dir = os.path.join(data_dir, "kinetics_waving")
        if has_kinetics_videos(kinetics_dir):
            X_p, _ = process_kinetics(kinetics_dir, processed_dir, processor, label=1)
            if len(X_p) > 0:
                all_X_pos.extend(X_p)
        else:
            print("[Kinetics] No videos found. Skipping. (YouTube blocked in this region)")

    # === Combine and save ===
    if all_X_pos or all_X_neg:
        X, y = _balance_and_pad(all_X_pos, all_X_neg, seq_len=args.seq_len)
        print(f"\n[FINAL] Total: {X.shape}, Pos={y.sum()}, Neg={(1-y).sum()}")

        # Train/val split
        n_val = max(1, int(X.shape[0] * 0.2))
        idx = np.random.permutation(X.shape[0])
        X, y = X[idx], y[idx]
        X_val, y_val = X[:n_val], y[:n_val]
        X_train, y_train = X[n_val:], y[n_val:]

        save_path = os.path.join(processed_dir, f"real_data_seq{args.seq_len}.npz")
        np.savez_compressed(save_path,
                            X_train=X_train, y_train=y_train,
                            X_val=X_val, y_val=y_val)
        print(f"Saved to: {save_path}")
        print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    else:
        print("\n[FINAL] No data collected. Check dataset downloads.")


if __name__ == "__main__":
    main()
