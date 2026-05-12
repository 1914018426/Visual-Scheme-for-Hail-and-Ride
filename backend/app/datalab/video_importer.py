"""
视频文件导入器

将本地视频文件（如 HMDB51/UCF101 数据集）导入为 DataLab 录制素材，
供消融实验离线分析使用。

处理流程：
1. cv2.VideoCapture 读取视频帧
2. YOLO11-Pose 逐帧人体检测
3. 计算 TNLF 特征（wrist_local、velocity、arm angles）
4. 按 DataLab 录制格式保存 keypoints / tnlf / detections / video
"""

import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from app.datalab.models import RecordingSession, ManualLabel
from app.datalab.persistence import DataLabStorage
from app.config import get_config

logger = logging.getLogger(__name__)


class VideoImporter:
    """离线视频导入处理器。"""

    def __init__(self, storage: DataLabStorage) -> None:
        self.storage = storage
        self.config = get_config()
        self._yolo_model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def import_video(
        self,
        video_path: str,
        camera_id: str = "imported",
        label: str = "unlabeled",
        notes: str = "",
    ) -> RecordingSession:
        """
        导入单个视频文件为 DataLab 录制素材。

        Args:
            video_path: 视频文件绝对路径（容器内路径）
            camera_id: 摄像头/来源标识
            label: 人工标签 positive / negative / unlabeled
            notes: 备注

        Returns:
            RecordingSession
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        logger.info(
            "开始导入视频: %s, frames=%d, fps=%.1f",
            video_path, total_frames, fps,
        )

        # 创建录制会话
        session = self.storage.create_recording(
            camera_id=camera_id,
            trigger_mode="manual",
        )
        session.manual_label = ManualLabel(label) if label in ("positive", "negative", "unlabeled") else ManualLabel.UNLABELED
        session.notes = notes

        rec_dir = Path(session.meta_path).parent
        dest_video_path = str(rec_dir / "video.mp4")
        shutil.copy2(str(path), dest_video_path)
        session.video_path = dest_video_path
        self.storage.update_recording(session)

        # 加载 YOLO 模型（懒加载）
        model = self._get_yolo_model()

        # 逐帧处理
        frame_idx = 0
        person_count_peak = 0
        prev_wrist_local: Dict[str, Tuple[float, float]] = {}
        prev_timestamp: Dict[str, float] = {}
        track_id = "person_1"

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            timestamp = time.time()
            h, w = frame.shape[:2]

            # YOLO 检测（只取帧内第一个人）
            persons = self._detect_persons(model, frame)
            if len(persons) > person_count_peak:
                person_count_peak = len(persons)
            if not persons:
                continue

            person = persons[0]
            kpt_array = person["keypoints"]

            # 计算 TNLF
            tnlf_data = self._compute_tnlf(
                kpt_array, prev_wrist_local, prev_timestamp, timestamp, track_id
            )

            # 计算 arm angles
            angles = self._compute_arm_angles(kpt_array)
            if angles:
                tnlf_data.update(angles)

            # 保存关键点
            keypoints_data = {
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "track_id": track_id,
                "keypoints": kpt_array.tolist(),
                "left_hand_landmarks": None,
                "right_hand_landmarks": None,
                "left_palm_normal": None,
                "right_palm_normal": None,
            }
            self.storage.append_keypoints(session.id, keypoints_data)

            # 保存检测输出（导入时无真实手势标签，设为 none）
            detections_data = {
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "person_count": len(persons),
                "gesture": "none",
                "gesture_conf": 0.0,
                "track_id": track_id,
            }
            self.storage.append_detections(session.id, detections_data)

            # 保存 TNLF
            tnlf_record = {
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "left_wrist_local": _tolist(tnlf_data.get("left_wrist_local")),
                "right_wrist_local": _tolist(tnlf_data.get("right_wrist_local")),
                "left_velocity_mag": tnlf_data.get("left_velocity_mag", 0.0),
                "right_velocity_mag": tnlf_data.get("right_velocity_mag", 0.0),
                "left_theta1": tnlf_data.get("left_theta1", 0.0),
                "left_theta2": tnlf_data.get("left_theta2", 0.0),
                "left_ext_ratio": tnlf_data.get("left_ext_ratio", 0.0),
                "right_theta1": tnlf_data.get("right_theta1", 0.0),
                "right_theta2": tnlf_data.get("right_theta2", 0.0),
                "right_ext_ratio": tnlf_data.get("right_ext_ratio", 0.0),
                "left_tnlf_valid": tnlf_data.get("left_tnlf_valid", False),
                "right_tnlf_valid": tnlf_data.get("right_tnlf_valid", False),
            }
            self.storage.append_tnlf(session.id, tnlf_record)

            # 更新上一帧缓存
            for side in ["left", "right"]:
                wl = tnlf_data.get(f"{side}_wrist_local")
                if wl is not None:
                    prev_wrist_local[f"{track_id}_{side}"] = wl
                    prev_timestamp[f"{track_id}_{side}"] = timestamp

            if frame_idx % 100 == 0:
                logger.info("视频导入进度: %s frame=%d/%d", session.id, frame_idx, total_frames or "?")

        cap.release()

        # 完成会话
        session.end_time = time.time()
        session.duration_s = round(session.end_time - session.start_time, 2)
        session.frame_count = frame_idx
        session.person_count = person_count_peak
        session.status = "completed"
        self.storage.update_recording(session)

        logger.info(
            "视频导入完成: id=%s frames=%d duration=%.1fs",
            session.id, frame_idx, session.duration_s or 0,
        )
        return session

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_yolo_model(self):
        """懒加载 YOLO 模型。"""
        if self._yolo_model is not None:
            return self._yolo_model

        from ultralytics import YOLO

        raw = self.config.ai.yolo_model.strip()
        if Path(raw).is_absolute():
            model_path = str(Path(raw).resolve())
        else:
            model_path = str(Path(self.config.ai.model_dir) / Path(raw).name)

        if not Path(model_path).exists():
            # 兜底：让 Ultralytics 自己下载
            model_path = raw

        self._yolo_model = YOLO(model_path, task="pose")
        logger.info("VideoImporter YOLO 模型加载成功: %s", model_path)
        return self._yolo_model

    def _detect_persons(
        self, model, frame: np.ndarray
    ) -> List[Dict[str, Any]]:
        """对单帧运行 YOLO 检测，返回人物列表。"""
        results: List[Dict[str, Any]] = []
        try:
            yolo_results = model(
                frame,
                conf=self.config.ai.conf_threshold,
                max_det=self.config.ai.max_detections,
                verbose=False,
                imgsz=self.config.ai.inference_imgsz,
                half=self.config.ai.inference_half,
            )
            for result in yolo_results:
                if result.boxes is None or result.keypoints is None:
                    continue
                boxes = result.boxes.cpu().numpy()
                keypoints = result.keypoints.cpu().numpy()
                for box, kpts in zip(boxes, keypoints):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    kpt_array = kpts.data if hasattr(kpts, "data") else kpts
                    if isinstance(kpt_array, np.ndarray):
                        if kpt_array.ndim == 3:
                            kpt_array = kpt_array[0]
                    else:
                        kpt_array = np.array(kpt_array)
                        if kpt_array.ndim == 3:
                            kpt_array = kpt_array[0]
                    if kpt_array.ndim == 1:
                        kpt_array = kpt_array.reshape(-1, 3)
                    results.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf,
                        "keypoints": kpt_array,
                    })
        except Exception as e:
            logger.warning("视频导入帧检测失败: %s", e)
        return results

    def _compute_tnlf(
        self,
        keypoints: np.ndarray,
        prev_wrist_local: Dict[str, Tuple[float, float]],
        prev_timestamp: Dict[str, float],
        timestamp: float,
        track_id: str,
    ) -> Dict[str, Any]:
        """计算 TNLF 特征与速度。"""
        from app.ai.local_frame import wrist_to_local_frame_full

        data: Dict[str, Any] = {}
        for side in ["left", "right"]:
            wl, origin, e_x, e_y, torso_scale, valid = wrist_to_local_frame_full(
                keypoints, side=side
            )
            prefix = side
            data[f"{prefix}_wrist_local"] = np.array(wl).tolist() if wl is not None else None
            data[f"{prefix}_tnlf_valid"] = valid

            # 速度
            v_mag = 0.0
            if wl is not None:
                cache_key = f"{track_id}_{side}"
                prev_wl = prev_wrist_local.get(cache_key)
                prev_ts = prev_timestamp.get(cache_key)
                if prev_wl is not None and prev_ts is not None:
                    dt = timestamp - prev_ts
                    if 0 < dt < 1.0:
                        v_mag = float(
                            np.linalg.norm(np.array(wl) - np.array(prev_wl)) / dt
                        )
            data[f"{prefix}_velocity_mag"] = v_mag

        return data

    def _compute_arm_angles(self, keypoints: np.ndarray) -> Optional[Dict[str, float]]:
        """计算左右手臂角度（theta1, theta2, ext_ratio）。"""
        try:
            import torch
            from app.ai.transformer.model import compute_arm_angles

            kpts_t = torch.tensor(keypoints, dtype=torch.float32)
            rt1, rt2, rext = compute_arm_angles(kpts_t, "right")
            lt1, lt2, lext = compute_arm_angles(kpts_t, "left")
            return {
                "left_theta1": float(lt1),
                "left_theta2": float(lt2),
                "left_ext_ratio": float(lext),
                "right_theta1": float(rt1),
                "right_theta2": float(rt2),
                "right_ext_ratio": float(rext),
            }
        except Exception:
            return None


def _tolist(val: Any) -> Any:
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val
