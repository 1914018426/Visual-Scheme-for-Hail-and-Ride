"""
Data pipeline for training the TemporalKeypointTransformer.

Provides:
- Synthetic data generator (realistic TNLF feature sequences)
- NTU RGB+D skeleton converter (requires dataset download)
- Kinetics-700 video processor (requires YouTube-DL + YOLO)
- JAAD pedestrian crop processor (requires dataset download)
"""

import math
import os
import pickle
import random
from collections import deque
from typing import Iterator, List, Optional, Tuple

import numpy as np

# Feature indices (must match model.n_features = 12)
F_WLX_L, F_WLY_L = 0, 1       # wrist_local left
F_WLX_R, F_WLY_R = 2, 3       # wrist_local right
F_VEL_MAG = 4                  # velocity magnitude (dominant side)
F_THETA1 = 5                   # arm lift angle (degrees)
F_THETA2 = 6                   # arm straightness (degrees)
F_EXT_RATIO = 7                # arm extension ratio
F_PN_X, F_PN_Y, F_PN_Z = 8, 9, 10  # palm normal (unit vector)
F_VALID = 11                   # TNLF validity

N_FEATURES = 12
DEFAULT_SEQ_LEN = 45           # 3 seconds @ 15 fps
DEFAULT_FPS = 15.0


def generate_waving_sequence(
    seq_len: int = DEFAULT_SEQ_LEN,
    fps: float = DEFAULT_FPS,
    freq_hz: Optional[float] = None,
    amplitude: Optional[float] = None,
    side: str = "right",
    noise_std: float = 0.03,
) -> np.ndarray:
    """
    Generate a realistic waving gesture sequence in TNLF feature space.

    A waving gesture consists of:
    - Rhythmic lateral oscillation of the wrist (x_local in TNLF)
    - Raised arm (negative y_local, above shoulder)
    - Arm extended and reasonably straight
    - Palm normal oscillating (facing toward/away from camera)

    Returns:
        np.ndarray of shape [seq_len, N_FEATURES]
    """
    if freq_hz is None:
        freq_hz = random.uniform(0.4, 2.5)
    if amplitude is None:
        amplitude = random.uniform(0.15, 0.45)

    dt = 1.0 / fps
    t = np.arange(seq_len) * dt
    omega = 2.0 * math.pi * freq_hz

    # Primary oscillation in x_local (lateral waving)
    x_osc = amplitude * np.sin(omega * t)
    # Secondary vertical bounce
    y_osc = amplitude * 0.15 * np.sin(2.0 * omega * t + random.uniform(0, math.pi))

    feat = np.zeros((seq_len, N_FEATURES), dtype=np.float32)

    if side == "right":
        # Dominant right hand
        base_wlx = 0.15 + x_osc
        base_wly = -0.35 + y_osc  # above shoulder
        feat[:, F_WLX_R] = base_wlx
        feat[:, F_WLY_R] = base_wly
        feat[:, F_WLX_L] = np.random.normal(0, 0.05, seq_len)
        feat[:, F_WLY_L] = np.random.normal(-0.05, 0.05, seq_len)
    else:
        base_wlx = -0.15 - x_osc
        base_wly = -0.35 + y_osc
        feat[:, F_WLX_L] = base_wlx
        feat[:, F_WLY_L] = base_wly
        feat[:, F_WLX_R] = np.random.normal(0, 0.05, seq_len)
        feat[:, F_WLY_R] = np.random.normal(-0.05, 0.05, seq_len)

    # Velocity magnitude (via finite difference of dominant wrist)
    if side == "right":
        dx = np.gradient(feat[:, F_WLX_R]) / dt
        dy = np.gradient(feat[:, F_WLY_R]) / dt
    else:
        dx = np.gradient(feat[:, F_WLX_L]) / dt
        dy = np.gradient(feat[:, F_WLY_L]) / dt
    feat[:, F_VEL_MAG] = np.sqrt(dx**2 + dy**2)

    # Arm angles
    theta1_base = random.uniform(35, 90)  # arm lift
    theta2_base = random.uniform(20, 60)  # arm straightness
    feat[:, F_THETA1] = theta1_base + amplitude * 8 * np.sin(omega * t + 0.3)
    feat[:, F_THETA2] = theta2_base + amplitude * 5 * np.sin(omega * t + 0.8)
    feat[:, F_EXT_RATIO] = 0.6 + amplitude * 0.15 * np.sin(omega * t)

    # Palm normal (oscillates as hand waves)
    pn_phase = omega * t + random.uniform(0, math.pi)
    pn_x = 0.3 * np.sin(pn_phase)
    pn_y = 0.2 * np.cos(pn_phase)
    pn_z = 0.5 + 0.4 * np.sin(pn_phase * 0.7)  # faces toward camera more often
    pn_norm = np.sqrt(pn_x**2 + pn_y**2 + pn_z**2) + 1e-8
    feat[:, F_PN_X] = pn_x / pn_norm
    feat[:, F_PN_Y] = pn_y / pn_norm
    feat[:, F_PN_Z] = pn_z / pn_norm

    # TNLF valid (occasional dropout)
    feat[:, F_VALID] = 1.0
    dropout_mask = np.random.random(seq_len) < 0.05
    feat[dropout_mask, F_VALID] = 0.0

    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, (seq_len, N_FEATURES))
    noise[:, F_VALID] = 0  # don't noise the validity flag
    feat += noise

    return feat.astype(np.float32)


def generate_walking_sequence(
    seq_len: int = DEFAULT_SEQ_LEN,
    fps: float = DEFAULT_FPS,
    side: str = "right",
    noise_std: float = 0.03,
) -> np.ndarray:
    """
    Generate a walking pedestrian arm-swing pattern (negative sample).

    Walking produces low-amplitude, low-frequency arm swing that the
    transformer must learn to distinguish from waving.
    """
    dt = 1.0 / fps
    t = np.arange(seq_len) * dt
    walk_freq = random.uniform(0.6, 1.2)  # typical walking cadence
    omega = 2.0 * math.pi * walk_freq
    amp = random.uniform(0.05, 0.15)

    feat = np.zeros((seq_len, N_FEATURES), dtype=np.float32)

    # Both arms swing in anti-phase during walking
    x_osc = amp * np.sin(omega * t)
    y_osc = amp * 0.2 * np.cos(2 * omega * t)

    feat[:, F_WLX_R] = 0.05 + x_osc
    feat[:, F_WLY_R] = 0.1 + y_osc   # below shoulder (positive y in TNLF)
    feat[:, F_WLX_L] = -0.05 - x_osc # anti-phase
    feat[:, F_WLY_L] = 0.1 - y_osc

    # Velocity
    dx = np.gradient(feat[:, F_WLX_R]) / dt
    dy = np.gradient(feat[:, F_WLY_R]) / dt
    feat[:, F_VEL_MAG] = np.sqrt(dx**2 + dy**2)

    # Arm nearly straight down (low theta1, high theta2)
    feat[:, F_THETA1] = random.uniform(5, 20) + amp * 3 * np.sin(omega * t)
    feat[:, F_THETA2] = random.uniform(140, 170)
    feat[:, F_EXT_RATIO] = 0.95 + amp * 0.02 * np.sin(omega * t)

    # Palm normal pointing down, not toward camera
    n = math.sqrt(2)
    feat[:, F_PN_X] = 0.2 / n
    feat[:, F_PN_Y] = -0.8 / n
    feat[:, F_PN_Z] = 0.2 / n

    feat[:, F_VALID] = 1.0
    dropout_mask = np.random.random(seq_len) < 0.1
    feat[dropout_mask, F_VALID] = 0.0

    noise = np.random.normal(0, noise_std, (seq_len, N_FEATURES))
    noise[:, F_VALID] = 0
    feat += noise

    return feat.astype(np.float32)


def generate_standing_sequence(
    seq_len: int = DEFAULT_SEQ_LEN,
    fps: float = DEFAULT_FPS,
    noise_std: float = 0.02,
) -> np.ndarray:
    """Generate a standing-still pedestrian (negative sample)."""
    feat = np.zeros((seq_len, N_FEATURES), dtype=np.float32)

    # Arms at sides, minimal motion
    feat[:, F_WLX_L] = np.random.normal(-0.1, 0.03, seq_len)
    feat[:, F_WLY_L] = np.random.normal(0.05, 0.03, seq_len)
    feat[:, F_WLX_R] = np.random.normal(0.1, 0.03, seq_len)
    feat[:, F_WLY_R] = np.random.normal(0.05, 0.03, seq_len)
    feat[:, F_VEL_MAG] = np.abs(np.random.normal(0, 0.02, seq_len))

    feat[:, F_THETA1] = random.uniform(5, 15)
    feat[:, F_THETA2] = random.uniform(150, 175)
    feat[:, F_EXT_RATIO] = 0.98

    n = math.sqrt(3)
    feat[:, F_PN_X] = 0.3 / n
    feat[:, F_PN_Y] = -0.7 / n
    feat[:, F_PN_Z] = 0.3 / n
    feat[:, F_VALID] = 1.0

    noise = np.random.normal(0, noise_std, (seq_len, N_FEATURES))
    noise[:, F_VALID] = 0
    feat += noise

    return feat.astype(np.float32)


def generate_phone_use_sequence(
    seq_len: int = DEFAULT_SEQ_LEN,
    fps: float = DEFAULT_FPS,
    noise_std: float = 0.02,
) -> np.ndarray:
    """Generate a person looking at phone / arm bent (negative sample - hand up but not waving)."""
    feat = np.zeros((seq_len, N_FEATURES), dtype=np.float32)

    # Arm raised and bent (like checking phone) — mimics hand_up but stationary
    feat[:, F_WLX_R] = 0.12 + np.random.normal(0, 0.02, seq_len)
    feat[:, F_WLY_R] = -0.25 + np.random.normal(0, 0.02, seq_len)  # raised
    feat[:, F_WLX_L] = np.random.normal(-0.05, 0.03, seq_len)
    feat[:, F_WLY_L] = np.random.normal(0.05, 0.03, seq_len)
    feat[:, F_VEL_MAG] = np.abs(np.random.normal(0, 0.03, seq_len))  # very slow

    feat[:, F_THETA1] = random.uniform(30, 60)   # arm IS raised — tricky negative!
    feat[:, F_THETA2] = random.uniform(60, 100)  # bent elbow
    feat[:, F_EXT_RATIO] = random.uniform(0.4, 0.6)

    n = math.sqrt(3)
    feat[:, F_PN_X] = 0.3 / n
    feat[:, F_PN_Y] = -0.6 / n
    feat[:, F_PN_Z] = 0.1 / n   # palm not facing camera
    feat[:, F_VALID] = 1.0

    noise = np.random.normal(0, noise_std, (seq_len, N_FEATURES))
    noise[:, F_VALID] = 0
    feat += noise

    return feat.astype(np.float32)


def generate_random_gesture_sequence(
    seq_len: int = DEFAULT_SEQ_LEN,
    fps: float = DEFAULT_FPS,
    noise_std: float = 0.04,
) -> np.ndarray:
    """Generate random arm motions (negative sample - noisy background)."""
    feat = np.zeros((seq_len, N_FEATURES), dtype=np.float32)

    # Brownian noise wrist motion
    wx_r = np.cumsum(np.random.normal(0, 0.04, seq_len))
    wy_r = np.cumsum(np.random.normal(0, 0.04, seq_len))
    wx_l = np.cumsum(np.random.normal(0, 0.04, seq_len))
    wy_l = np.cumsum(np.random.normal(0, 0.04, seq_len))

    # Center and constrain
    wx_r -= wx_r.mean()
    wy_r -= wy_r.mean()
    wx_l -= wx_l.mean()
    wy_l -= wy_l.mean()

    feat[:, F_WLX_R] = wx_r * 0.5
    feat[:, F_WLY_R] = wy_r * 0.5
    feat[:, F_WLX_L] = wx_l * 0.5
    feat[:, F_WLY_L] = wy_l * 0.5

    dt = 1.0 / fps
    dx = np.gradient(feat[:, F_WLX_R]) / dt
    dy = np.gradient(feat[:, F_WLY_R]) / dt
    feat[:, F_VEL_MAG] = np.sqrt(dx**2 + dy**2)

    feat[:, F_THETA1] = np.clip(np.cumsum(np.random.normal(0, 3, seq_len)) + 30, 0, 180)
    feat[:, F_THETA2] = np.clip(np.cumsum(np.random.normal(0, 5, seq_len)) + 90, 0, 180)
    feat[:, F_EXT_RATIO] = np.clip(np.random.normal(0.6, 0.2, seq_len), 0.1, 1.0)

    # Random palm normal
    pn = np.random.normal(0, 1, (seq_len, 3))
    pn_norm = np.linalg.norm(pn, axis=1, keepdims=True) + 1e-8
    pn = pn / pn_norm
    feat[:, F_PN_X:F_PN_Z + 1] = pn
    feat[:, F_VALID] = (np.random.random(seq_len) > 0.15).astype(np.float32)

    noise = np.random.normal(0, noise_std, (seq_len, N_FEATURES))
    noise[:, F_VALID] = 0
    feat += noise

    return feat.astype(np.float32)


class SyntheticDataset:
    """
    Generates balanced synthetic TNLF feature sequences for training.

    Positive class: waving gestures (varying frequency, amplitude, side, noise)
    Negative classes: walking, standing, phone-use, random gestures
    """

    NEGATIVE_GENERATORS = [
        generate_walking_sequence,
        generate_standing_sequence,
        generate_phone_use_sequence,
        generate_random_gesture_sequence,
    ]

    def __init__(
        self,
        n_samples: int = 10000,
        seq_len: int = DEFAULT_SEQ_LEN,
        fps: float = DEFAULT_FPS,
        positive_ratio: float = 0.5,
        val_split: float = 0.2,
        seed: int = 42,
    ):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.fps = fps
        self.positive_ratio = positive_ratio
        self.val_split = val_split
        self.seed = seed

    def generate(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train and validation splits.

        Returns:
            ((X_train, y_train), (X_val, y_val))
            X: [N, seq_len, N_FEATURES] float32
            y: [N] int64 (0=negative, 1=positive)
        """
        rng = random.Random(self.seed)
        np.random.seed(self.seed)

        n_pos = int(self.n_samples * self.positive_ratio)
        n_neg = self.n_samples - n_pos

        X_pos = np.zeros((n_pos, self.seq_len, N_FEATURES), dtype=np.float32)
        for i in range(n_pos):
            freq = rng.uniform(0.35, 3.0)
            amp = rng.uniform(0.1, 0.5)
            side = rng.choice(["left", "right"])
            noise = rng.uniform(0.01, 0.06)
            X_pos[i] = generate_waving_sequence(
                self.seq_len, self.fps, freq_hz=freq,
                amplitude=amp, side=side, noise_std=noise,
            )

        X_neg = np.zeros((n_neg, self.seq_len, N_FEATURES), dtype=np.float32)
        for i in range(n_neg):
            gen = rng.choice(self.NEGATIVE_GENERATORS)
            try:
                if gen == generate_walking_sequence:
                    side = rng.choice(["left", "right"])
                    noise = rng.uniform(0.01, 0.05)
                    X_neg[i] = gen(self.seq_len, self.fps, side=side, noise_std=noise)
                elif gen == generate_standing_sequence:
                    X_neg[i] = gen(self.seq_len, self.fps, noise_std=rng.uniform(0.01, 0.04))
                elif gen == generate_phone_use_sequence:
                    X_neg[i] = gen(self.seq_len, self.fps, noise_std=rng.uniform(0.01, 0.04))
                else:
                    X_neg[i] = gen(self.seq_len, self.fps, noise_std=rng.uniform(0.02, 0.06))
            except Exception:
                X_neg[i] = generate_standing_sequence(self.seq_len, self.fps)

        X = np.concatenate([X_pos, X_neg], axis=0)
        y = np.concatenate([np.ones(n_pos, dtype=np.int64), np.zeros(n_neg, dtype=np.int64)])

        # Shuffle
        idx = np.random.permutation(self.n_samples)
        X, y = X[idx], y[idx]

        # Split
        n_val = int(self.n_samples * self.val_split)
        X_val, y_val = X[:n_val], y[:n_val]
        X_train, y_train = X[n_val:], y[n_val:]

        return (X_train, y_train), (X_val, y_val)

    @staticmethod
    def compute_normalization_stats(
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-feature mean and std for normalization.
        Excludes F_VALID (binary flag) from std normalization.

        Returns:
            (mean, std) each of shape [N_FEATURES]
        """
        mean = X.mean(axis=(0, 1))
        std = X.std(axis=(0, 1))
        std[std < 1e-6] = 1.0
        std[F_VALID] = 1.0  # don't scale binary flag
        return mean.astype(np.float32), std.astype(np.float32)


# ============================================================================
# NTU RGB+D Dataset Processor (requires downloaded dataset)
# ============================================================================

# NTU 25-joint skeleton to COCO-17 mapping
# NTU indices -> COCO index (or -1 if no mapping)
NTU_TO_COCO = {
    0: 0,    # base of spine -> nose (approximate)
    1: -1,   # middle of spine -> no COCO equivalent
    2: 5,    # neck -> left shoulder (approximate — neck maps to shoulder midpoint)
    3: 0,    # head -> nose
    4: 11,   # left shoulder -> left hip (check)
    5: 7,    # left elbow -> left elbow
    6: 9,    # left wrist -> left wrist
    7: -1,   # left hand -> no COCO (we use wrist)
    8: 12,   # right shoulder -> right hip
    9: 8,    # right elbow -> right elbow
    10: 10,  # right wrist -> right wrist
    11: -1,  # right hand
    12: 11,  # left hip -> left hip
    13: 13,  # left knee -> left knee
    14: 15,  # left ankle -> left ankle
    15: -1,  # left foot
    16: 12,  # right hip -> right hip
    17: 14,  # right knee -> right knee
    18: 16,  # right ankle -> right ankle
    19: -1,  # right foot
    20: -1,  # spine
    21: -1,  # tip of left hand
    22: -1,  # left thumb
    23: -1,  # tip of right hand
    24: -1,  # right thumb
}


def ntu_skeleton_to_tnlf_features(
    skeleton_3d: np.ndarray,
    fps: float = 30.0,
    target_fps: float = 15.0,
    seq_len: int = DEFAULT_SEQ_LEN,
) -> Optional[np.ndarray]:
    """
    Convert an NTU RGB+D skeleton sequence to TNLF feature sequence.

    Args:
        skeleton_3d: [T, 25, 3] NTU 3D skeleton (x, y, z in mm)
        fps:          Original NTU frame rate (typically 30)
        target_fps:   Desired output frame rate
        seq_len:      Fixed output sequence length

    Returns:
        [seq_len, N_FEATURES] or None if skeleton quality insufficient
    """
    T = skeleton_3d.shape[0]
    if T < 10:
        return None

    # Map to COCO-17 format [T, 17, 3]
    coco_kpts = np.zeros((T, 17, 3), dtype=np.float32)
    for ntu_idx, coco_idx in NTU_TO_COCO.items():
        if coco_idx >= 0 and ntu_idx < skeleton_3d.shape[1]:
            coco_kpts[:, coco_idx, :2] = skeleton_3d[:, ntu_idx, :2]  # x, y
            coco_kpts[:, coco_idx, 2] = 0.8  # synthetic confidence

    # Fill neck as shoulder midpoint
    if skeleton_3d.shape[1] > 2:
        coco_kpts[:, 0, :2] = skeleton_3d[:, 2, :2]  # neck -> nose approximate

    # Downsample to target_fps
    if fps > target_fps:
        stride = max(1, int(fps / target_fps))
        coco_kpts = coco_kpts[::stride]
        T = coco_kpts.shape[0]

    # Compute TNLF features per frame
    features = np.zeros((T, N_FEATURES), dtype=np.float32)
    for t in range(T):
        kpts = coco_kpts[t]

        # Shoulder midpoint
        shoulder_mid = (kpts[5, :2] + kpts[6, :2]) / 2.0
        hip_mid = (kpts[11, :2] + kpts[12, :2]) / 2.0

        e_x = kpts[6, :2] - kpts[5, :2]  # right - left shoulder
        e_x_norm = np.linalg.norm(e_x)
        if e_x_norm < 1e-6:
            features[t, F_VALID] = 0.0
            continue
        e_x = e_x / e_x_norm

        e_y = hip_mid - shoulder_mid  # 与推理 local_frame.py 保持一致：指向图像下方（hip 方向）
        torso_scale = np.linalg.norm(e_y)
        if torso_scale < 10.0:
            e_y = np.array([0.0, e_x_norm * 3.0])
            torso_scale = e_x_norm * 3.0
        e_y = e_y / (np.linalg.norm(e_y) + 1e-8)

        # Left wrist in TNLF
        wl = kpts[9, :2] - shoulder_mid
        features[t, F_WLX_L] = np.dot(wl, e_x) / torso_scale
        features[t, F_WLY_L] = np.dot(wl, e_y) / torso_scale

        # Right wrist in TNLF
        wr = kpts[10, :2] - shoulder_mid
        features[t, F_WLX_R] = np.dot(wr, e_x) / torso_scale
        features[t, F_WLY_R] = np.dot(wr, e_y) / torso_scale

        # 选择活跃手臂：手腕在 TNLF 中 y 更小（更靠上）的一侧（图像 y 越小越靠上，
        # 与之对应的 TNLF y_local 越小代表手腕越靠上）
        active_side = "left" if features[t, F_WLY_L] < features[t, F_WLY_R] else "right"

        # Arm angles for active side
        from .model import compute_arm_angles
        import torch
        kpts_t = torch.tensor(kpts)
        theta1, theta2, ext_ratio = compute_arm_angles(kpts_t, active_side)
        features[t, F_THETA1] = theta1
        features[t, F_THETA2] = theta2
        features[t, F_EXT_RATIO] = ext_ratio

        # Palm normal (use forearm direction of active arm as proxy)
        elbow_idx, wrist_idx = (7, 9) if active_side == "left" else (8, 10)
        forearm = kpts[wrist_idx, :2] - kpts[elbow_idx, :2]
        f_norm = np.linalg.norm(forearm) + 1e-8
        features[t, F_PN_X] = forearm[0] / f_norm
        features[t, F_PN_Y] = forearm[1] / f_norm
        features[t, F_PN_Z] = 0.5  # neutral Z

        features[t, F_VALID] = 1.0

    # Velocity (active arm — choose the side with higher wrist over the entire sequence)
    dt = 1.0 / target_fps
    mean_wly_l = float(np.mean(features[:, F_WLY_L]))
    mean_wly_r = float(np.mean(features[:, F_WLY_R]))
    if mean_wly_l < mean_wly_r:
        dx = np.gradient(features[:, F_WLX_L]) / dt
        dy = np.gradient(features[:, F_WLY_L]) / dt
    else:
        dx = np.gradient(features[:, F_WLX_R]) / dt
        dy = np.gradient(features[:, F_WLY_R]) / dt
    features[:, F_VEL_MAG] = np.sqrt(dx**2 + dy**2)

    # Pad or truncate to seq_len
    if T < seq_len:
        pad = np.tile(features[-1:], (seq_len - T, 1))
        features = np.concatenate([features, pad], axis=0)
    elif T > seq_len:
        # Random crop
        start = np.random.randint(0, T - seq_len + 1)
        features = features[start:start + seq_len]

    return features.astype(np.float32)


def load_synthetic_dataset(
    n_samples: int = 10000,
    seq_len: int = DEFAULT_SEQ_LEN,
    save_dir: Optional[str] = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load or generate synthetic dataset.

    If save_dir is provided, cache the dataset to disk.
    """
    cache_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cache_path = os.path.join(save_dir, f"synthetic_n{n_samples}_t{seq_len}.npz")

    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        return (
            (data["X_train"], data["y_train"]),
            (data["X_val"], data["y_val"]),
        )

    ds = SyntheticDataset(n_samples=n_samples, seq_len=seq_len)
    (X_train, y_train), (X_val, y_val) = ds.generate()

    if cache_path:
        np.savez_compressed(cache_path, X_train=X_train, y_train=y_train,
                            X_val=X_val, y_val=y_val)

    return (X_train, y_train), (X_val, y_val)
