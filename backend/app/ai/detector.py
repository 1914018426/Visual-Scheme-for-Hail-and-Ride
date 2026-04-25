"""
姿态检测器模块

PoseDetector 类提供完整的人体检测和姿态分析流水线：
1. 使用 YOLOv8-pose 进行人体检测和17点姿态估计
2. 使用 MediaPipe Hands 进行手部关键点检测（可选）
3. 骨骼绘制和可视化标注

模型自动下载到 ./models/ 目录。
"""

import os
import logging
import subprocess
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import time

import cv2
import numpy as np

from app.config import get_config, _env_bool

logger = logging.getLogger(__name__)

# 国内镜像拉取权重失败时的最小文件体积（字节），防止下到错误页
_MIN_WEIGHT_BYTES = 100_000


def _download_yolo_weights_cn(dest_path: str, weight_basename: str) -> bool:
    """
    从国内可访问的 Hugging Face 镜像下载 YOLO 权重到 dest_path（hf-mirror / 腾讯云 / 交大镜像），
    最后兜底为 GitHub Release 直连；不使用 GitHub 代理域名。
    大文件使用 curl 落盘（避免整包读入内存）。
    """
    name = os.path.basename(weight_basename.strip())
    if not name.endswith(".pt"):
        return False

    tags = ("v8.4.0", "v8.3.0", "v8.2.0", "v8.1.0", "v8.0.0")
    # 仅国内/常用 HF 镜像（与 huggingface.co 同一路径结构），不使用 GitHub 代理站
    repos = ("Ultralytics/YOLOv8", "Ultralytics/yolov8")
    hf_bases = (
        "https://hf-mirror.com",
        "https://mirrors.cloud.tencent.com/huggingface",
        "https://mirror.sjtu.edu.cn/huggingface.co",
    )
    urls: List[str] = []
    for base in hf_bases:
        for repo in repos:
            urls.append(f"{base}/{repo}/resolve/main/{name}")
    # 官方 GitHub Release 直连（非代理；国内可能较慢，作兜底）
    for tag in tags:
        urls.append(
            f"https://github.com/ultralytics/assets/releases/download/{tag}/{name}"
        )

    d = os.path.dirname(dest_path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = dest_path + ".part"

    def _curl_download(url: str) -> bool:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        logger.info("开始下载权重(curl): %s", url)
        try:
            proc = subprocess.run(
                [
                    "curl",
                    "-fL",
                    "--connect-timeout",
                    "30",
                    "--max-time",
                    "3600",
                    "--retry",
                    "2",
                    "--retry-delay",
                    "8",
                    "-o",
                    tmp,
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=3700,
            )
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "")[:800]
                logger.warning(
                    "curl 下载失败 returncode=%s url=%s err=%s",
                    proc.returncode,
                    url,
                    err,
                )
                return False
            if not os.path.isfile(tmp):
                return False
            sz = os.path.getsize(tmp)
            if sz < _MIN_WEIGHT_BYTES:
                logger.warning("下载文件过小(%d bytes)，丢弃: %s", sz, url)
                return False
            os.replace(tmp, dest_path)
            logger.info("权重已保存: %s (%d bytes)", dest_path, sz)
            return True
        except subprocess.TimeoutExpired:
            logger.warning("curl 下载超时: %s", url)
            return False
        except Exception as e:
            logger.warning("curl 下载异常 %s: %s", url, e)
            return False

    for url in urls:
        if _curl_download(url):
            return True
    return False


@dataclass
class PersonDetection:
    """单个人物检测结果。"""

    bbox: Tuple[int, int, int, int]  # 边界框 (x1, y1, x2, y2)
    confidence: float                # 检测置信度
    keypoints: np.ndarray            # 姿态关键点 (17, 3) [x, y, conf]
    track_id: str = ""               # 跟踪ID
    gesture: str = "none"            # 手势类型
    gesture_conf: float = 0.0        # 手势置信度
    direction: str = "none"          # 对应方向
    left_hand_landmarks: Optional[List[Tuple[float, float, float]]] = None
    right_hand_landmarks: Optional[List[Tuple[float, float, float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于JSON序列化）。"""
        return {
            "bbox": self.bbox,
            "confidence": round(self.confidence, 4),
            "keypoints": self.keypoints.tolist()
            if isinstance(self.keypoints, np.ndarray)
            else self.keypoints,
            "track_id": self.track_id,
            "gesture": self.gesture,
            "gesture_conf": round(self.gesture_conf, 4),
            "direction": self.direction,
        }


@dataclass
class DetectionResult:
    """完整检测结果。"""

    persons: List[PersonDetection] = field(default_factory=list)
    fps: float = 0.0
    inference_time_ms: float = 0.0
    frame_shape: Tuple[int, int] = (0, 0)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "persons": [p.to_dict() for p in self.persons],
            "fps": round(self.fps, 2),
            "inference_time_ms": round(self.inference_time_ms, 2),
            "frame_shape": self.frame_shape,
        }


class PoseDetector:
    """
    姿态检测器

    基于YOLOv8-pose和MediaPipe Hands的完整检测流水线。
    提供人体检测、姿态估计、骨骼绘制和手势分析功能。

    Attributes:
        model: YOLOv8模型实例
        use_mediapipe: 是否启用MediaPipe手部检测
    """

    # COCO姿态关键点连接定义（用于绘制骨骼）
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),        # 头部
        (5, 6),                                   # 肩膀
        (5, 7), (7, 9), (6, 8), (8, 10),        # 手臂
        (11, 12),                                 # 髋部
        (5, 11), (6, 12),                         # 躯干
        (11, 13), (13, 15), (12, 14), (14, 16),  # 腿部
    ]

    # 关键点颜色
    KEYPOINT_COLORS = [
        (0, 0, 255),    # 鼻子 - 红色
        (0, 255, 0), (0, 255, 0),    # 眼睛 - 绿色
        (0, 255, 255), (0, 255, 255),  # 耳朵 - 黄色
        (255, 0, 0), (255, 0, 0),    # 肩膀 - 蓝色
        (255, 255, 0), (255, 255, 0),  # 手肘 - 青色
        (255, 0, 255), (255, 0, 255),  # 手腕 - 紫色
        (128, 0, 128), (128, 0, 128),  # 髋部 - 深紫
        (0, 128, 255), (0, 128, 255),  # 膝盖 - 橙色
        (128, 255, 0), (128, 255, 0),  # 脚踝 - 浅绿
    ]

    def __init__(self) -> None:
        """
        初始化姿态检测器。

        自动加载YOLOv8-pose模型，如模型文件不存在则自动下载。
        可选加载MediaPipe Hands模型用于手部关键点检测。
        """
        self.config = get_config()
        self.model_dir = self.config.ai.model_dir
        self.conf_threshold = self.config.ai.conf_threshold
        self.max_detections = self.config.ai.max_detections
        self.use_mediapipe = self.config.ai.enable_hand_detection

        # 确保模型目录存在
        os.makedirs(self.model_dir, exist_ok=True)

        # 加载YOLOv8-pose模型
        self.model = self._load_yolo_model()

        # 可选加载MediaPipe Hands
        self._mp_hands = None
        self._mp_hands_instance = None
        if self.use_mediapipe:
            self._load_mediapipe_hands()

        # 推理性能统计
        self._inference_times: List[float] = []
        self._stats_window_size = 30
        # 轻量目标跟踪：按摄像头维护 track_id，避免历史手势串人
        self._track_memory: Dict[str, List[Dict[str, Any]]] = {}
        self._track_counter: Dict[str, int] = {}
        # 手势轨迹：track_id -> deque[(x, y), ...]，用于绘制轨迹线
        self._gesture_trails: Dict[str, deque] = {}

        logger.info(
            "PoseDetector 初始化完成: model=%s, conf=%.2f, max_det=%d, "
            "mediapipe=%s",
            self.config.ai.yolo_model,
            self.conf_threshold,
            self.max_detections,
            self.use_mediapipe,
        )

    def _load_yolo_model(self):
        """
        加载YOLOv8-pose模型。

        权重优先落在 MODEL_DIR；缺失时先走国内镜像下载到本地，再交给 Ultralytics，
        避免每次从 GitHub 直连拉取。

        Returns:
            YOLO: YOLOv8模型实例
        """
        try:
            from ultralytics import YOLO

            raw = self.config.ai.yolo_model.strip()
            if os.path.isabs(raw):
                model_path = os.path.abspath(raw)
            else:
                model_path = os.path.abspath(
                    os.path.join(self.model_dir, os.path.basename(raw))
                )

            if not os.path.isfile(model_path) and _env_bool(
                "MODEL_CN_MIRROR", True
            ):
                logger.info(
                    "模型文件不存在，尝试国内镜像下载: %s -> %s",
                    raw,
                    model_path,
                )
                _download_yolo_weights_cn(model_path, raw)

            if os.path.isfile(model_path):
                model_name = model_path
                logger.info("加载本地模型: %s", model_path)
            else:
                model_name = os.path.basename(raw) if raw else raw
                logger.warning(
                    "本地仍无权重文件，将由 Ultralytics 从默认源下载: %s",
                    model_name,
                )

            model = YOLO(model_name, task="pose")
            logger.info("YOLOv8-pose 模型加载成功")
            return model

        except Exception as e:
            logger.error("YOLOv8模型加载失败: %s", str(e))
            raise RuntimeError(f"无法加载YOLOv8模型: {e}") from e

    def _load_mediapipe_hands(self) -> None:
        """加载MediaPipe Hands模型。"""
        try:
            import mediapipe as mp

            self._mp_hands = mp.solutions.hands
            self._mp_hands_instance = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=4,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("MediaPipe Hands 加载成功")
        except Exception as e:
            logger.warning("MediaPipe Hands 加载失败，将禁用手部检测: %s", str(e))
            self.use_mediapipe = False
            self._mp_hands = None
            self._mp_hands_instance = None

    def detect_persons(self, frame: np.ndarray) -> List[PersonDetection]:
        """
        检测图像中的所有人。

        Args:
            frame: 输入图像 (BGR格式)

        Returns:
            PersonDetection对象列表
        """
        if frame is None or frame.size == 0:
            return []

        results: List[PersonDetection] = []

        try:
            # YOLOv8推理
            yolo_results = self.model(
                frame,
                conf=self.conf_threshold,
                max_det=self.max_detections,
                verbose=False,
                imgsz=self.config.ai.inference_imgsz,
                half=self.config.ai.inference_half,
            )

            for result in yolo_results:
                if result.boxes is None or result.keypoints is None:
                    continue

                boxes = result.boxes.cpu().numpy()
                keypoints = result.keypoints.cpu().numpy()

                for i, (box, kpts) in enumerate(zip(boxes, keypoints)):
                    # 边界框
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    # 关键点 (17, 3)
                    kpt_array = kpts.data if hasattr(kpts, 'data') else kpts
                    if isinstance(kpt_array, np.ndarray):
                        if kpt_array.ndim == 3:
                            kpt_array = kpt_array[0]  # 取第一个人
                    else:
                        kpt_array = np.array(kpt_array)
                        if kpt_array.ndim == 3:
                            kpt_array = kpt_array[0]

                    # 确保关键点数组形状为 (17, 3)
                    if kpt_array.ndim == 1:
                        kpt_array = kpt_array.reshape(-1, 3)

                    person = PersonDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        keypoints=kpt_array,
                        track_id=f"person_{i}",
                    )
                    results.append(person)

        except Exception as e:
            logger.error("人体检测失败: %s", str(e))

        return results

    def _detect_hands_for_person(
        self, frame: np.ndarray, person: PersonDetection
    ) -> None:
        """
        对单个人体检测手部关键点，并按左右手分类存入 PersonDetection。

        策略：
        1. 在人体 bbox 上半部分裁剪 ROI（手腕通常在上方）
        2. 运行 MediaPipe Hands
        3. 根据 hand wrist (landmark 0) 与 pose left/right wrist (keypoints 9/10) 的距离分类
        """
        if self._mp_hands_instance is None:
            return

        x1, y1, x2, y2 = person.bbox
        h, w = frame.shape[:2]

        # 扩大 ROI 以覆盖手臂，限制在画面内
        margin_x = int((x2 - x1) * 0.2)
        margin_y = int((y2 - y1) * 0.1)
        rx1 = max(0, x1 - margin_x)
        ry1 = max(0, y1 - margin_y)
        rx2 = min(w, x2 + margin_x)
        ry2 = min(h, y2 + margin_y)

        if rx2 <= rx1 or ry2 <= ry1:
            return

        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return

        try:
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = self._mp_hands_instance.process(roi_rgb)
            if not results.multi_hand_landmarks:
                logger.info("MediaPipe Hands: 未检测到手 (track=%s)", person.track_id)
                return

            # pose 左右手腕位置（像素坐标）
            kpts = person.keypoints
            left_wrist_pose = (
                kpts[9] if len(kpts) > 9 and kpts[9][2] > 0.3 else None
            )
            right_wrist_pose = (
                kpts[10] if len(kpts) > 10 and kpts[10][2] > 0.3 else None
            )

            for hand_landmarks in results.multi_hand_landmarks:
                # 将归一化坐标转换为原始帧绝对坐标
                abs_landmarks: List[Tuple[float, float, float]] = []
                for lm in hand_landmarks.landmark:
                    abs_x = lm.x * (rx2 - rx1) + rx1
                    abs_y = lm.y * (ry2 - ry1) + ry1
                    abs_z = lm.z
                    abs_landmarks.append((abs_x, abs_y, abs_z))

                hand_wrist = abs_landmarks[0]

                # 匹配到最近的手腕
                left_dist = float("inf")
                right_dist = float("inf")
                if left_wrist_pose is not None:
                    left_dist = (
                        (hand_wrist[0] - left_wrist_pose[0]) ** 2
                        + (hand_wrist[1] - left_wrist_pose[1]) ** 2
                    ) ** 0.5
                if right_wrist_pose is not None:
                    right_dist = (
                        (hand_wrist[0] - right_wrist_pose[0]) ** 2
                        + (hand_wrist[1] - right_wrist_pose[1]) ** 2
                    ) ** 0.5

                # 以肩宽一半作为最大匹配距离（约 30-60 像素）
                shoulder_width = 60.0
                if (
                    len(kpts) > 6
                    and kpts[5][2] > 0.3
                    and kpts[6][2] > 0.3
                ):
                    shoulder_width = abs(kpts[6][0] - kpts[5][0])
                max_match_dist = max(40.0, shoulder_width * 0.6)

                if left_dist < right_dist and left_dist <= max_match_dist:
                    person.left_hand_landmarks = abs_landmarks
                elif right_dist <= max_match_dist:
                    person.right_hand_landmarks = abs_landmarks

            hand_count = len(results.multi_hand_landmarks)
            logger.info(
                "MediaPipe Hands: 检测到 %d 只手 (track=%s)",
                hand_count,
                person.track_id,
            )

        except Exception as e:
            logger.warning("MediaPipe Hands 检测失败: %s", str(e))

    def _assign_track_ids(
        self, persons: List[PersonDetection], camera_id: str = ""
    ) -> None:
        """基于 bbox 中心点做轻量跨帧关联，稳定每个人的 track_id。"""
        if not persons:
            return
        cam = camera_id or "default"
        prev_tracks = self._track_memory.get(cam, [])
        now = time.time()
        assigned: set[int] = set()

        for person in persons:
            x1, y1, x2, y2 = person.bbox
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            bw = max(1.0, float(x2 - x1))
            bh = max(1.0, float(y2 - y1))
            max_dist = min(160.0, max(40.0, (bw * bw + bh * bh) ** 0.5 * 0.6))

            best_idx = -1
            best_dist = float("inf")
            for idx, trk in enumerate(prev_tracks):
                if idx in assigned:
                    continue
                px, py = trk["center"]
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if dist < best_dist and dist <= max_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx >= 0:
                trk = prev_tracks[best_idx]
                person.track_id = trk["id"]
                trk["center"] = (cx, cy)
                trk["last_seen"] = now
                assigned.add(best_idx)
            else:
                idx = self._track_counter.get(cam, 0) + 1
                self._track_counter[cam] = idx
                person.track_id = f"{cam}_p{idx}"
                prev_tracks.append(
                    {"id": person.track_id, "center": (cx, cy), "last_seen": now}
                )
                assigned.add(len(prev_tracks) - 1)

        # 仅保留最近出现的人，避免内存增长
        self._track_memory[cam] = [
            trk for trk in prev_tracks if now - float(trk["last_seen"]) <= 1.5
        ][:64]

    def process_frame(
        self, frame: np.ndarray, camera_id: str = ""
    ) -> Tuple[np.ndarray, DetectionResult]:
        """
        完整检测流水线：检测 + 骨骼绘制。

        Args:
            frame: 输入图像 (BGR格式)
            camera_id: 摄像头标识（用于手势识别历史追踪）

        Returns:
            (绘制后的帧, 检测结果)
        """
        if frame is None or frame.size == 0:
            return frame, DetectionResult(frame_shape=(0, 0))

        start_time = time.time()
        h, w = frame.shape[:2]

        # 1. 人体检测
        persons = self.detect_persons(frame)
        self._assign_track_ids(persons, camera_id=camera_id)

        # 2. 手部关键点检测（MediaPipe Hands）
        if self.use_mediapipe:
            for person in persons:
                self._detect_hands_for_person(frame, person)

        # 3. 手势识别（为每个人检测手势）
        from app.ai.gesture import is_hailing_gesture

        frame_ts = time.time()
        for person in persons:
            gesture_type, gesture_conf = is_hailing_gesture(
                person.keypoints,
                person.track_id,
                left_hand_landmarks=person.left_hand_landmarks,
                right_hand_landmarks=person.right_hand_landmarks,
                frame_timestamp=frame_ts,
            )
            person.gesture = gesture_type
            person.gesture_conf = gesture_conf
            if gesture_type != "none":
                logger.info(
                    "手势识别: track=%s gesture=%s conf=%.2f left=%s right=%s",
                    person.track_id,
                    gesture_type,
                    gesture_conf,
                    "Y" if person.left_hand_landmarks else "N",
                    "Y" if person.right_hand_landmarks else "N",
                )
            # 记录 wrist 轨迹（用于绘制轨迹线）
            self._update_gesture_trail(person)

        # 4. 绘制骨骼和手势标记
        annotated_frame = self.draw_skeleton(frame.copy(), persons)

        # 4. 计算推理性能
        inference_time = (time.time() - start_time) * 1000  # ms
        self._inference_times.append(inference_time)
        if len(self._inference_times) > self._stats_window_size:
            self._inference_times.pop(0)

        avg_inference = (
            sum(self._inference_times) / len(self._inference_times)
            if self._inference_times else 0.0
        )
        fps = 1000.0 / avg_inference if avg_inference > 0 else 0.0

        result = DetectionResult(
            persons=persons,
            fps=fps,
            inference_time_ms=inference_time,
            frame_shape=(h, w),
        )

        return annotated_frame, result

    def draw_skeleton(
        self,
        frame: np.ndarray,
        persons: List[PersonDetection],
    ) -> np.ndarray:
        """
        在图像上绘制骨骼和手势标记。

        Args:
            frame: 输入图像
            persons: 人物检测结果列表

        Returns:
            绘制后的图像
        """
        for person in persons:
            x1, y1, x2, y2 = person.bbox

            # 根据手势类型选择边界框颜色
            if person.gesture == "hailing":
                box_color = (0, 0, 255)    # 红色 - 打车
                label = f"HAILING {person.gesture_conf:.2f}"
            elif person.gesture == "greeting":
                box_color = (255, 128, 0)  # 橙色/青色 - 打招呼
                label = f"GREETING {person.gesture_conf:.2f}"
            elif person.gesture == "hand_up":
                box_color = (0, 255, 255)  # 黄色 - 举手
                label = f"HAND_UP {person.gesture_conf:.2f}"
            else:
                box_color = (0, 255, 0)    # 绿色 - 无手势
                label = f"PERSON {person.confidence:.2f}"

            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # 绘制标签背景
            label_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )[0]
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - 8),
                (x1 + label_size[0] + 4, y1),
                box_color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # 绘制骨骼连线
            keypoints = person.keypoints
            if keypoints is not None and len(keypoints) >= 17:
                for connection in self.SKELETON_CONNECTIONS:
                    kp1_idx, kp2_idx = connection
                    if kp1_idx < len(keypoints) and kp2_idx < len(keypoints):
                        kp1 = keypoints[kp1_idx]
                        kp2 = keypoints[kp2_idx]

                        # 检查关键点置信度
                        if kp1[2] > 0.3 and kp2[2] > 0.3:
                            pt1 = (int(kp1[0]), int(kp1[1]))
                            pt2 = (int(kp2[0]), int(kp2[1]))
                            cv2.line(frame, pt1, pt2, (0, 200, 200), 2)

                # 绘制关键点
                for i, kp in enumerate(keypoints):
                    if i >= len(self.KEYPOINT_COLORS):
                        break
                    if len(kp) >= 3 and kp[2] > 0.3:
                        x, y = int(kp[0]), int(kp[1])
                        color = self.KEYPOINT_COLORS[i]
                        cv2.circle(frame, (x, y), 4, color, -1)

                # 绘制 MediaPipe Hands 关键点（21点）
                for side, landmarks in [
                    ("left", person.left_hand_landmarks),
                    ("right", person.right_hand_landmarks),
                ]:
                    if not landmarks or len(landmarks) < 21:
                        continue
                    # 手部连线定义
                    HAND_CONNECTIONS = [
                        (0, 1), (1, 2), (2, 3), (3, 4),   # 拇指
                        (0, 5), (5, 6), (6, 7), (7, 8),   # 食指
                        (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
                        (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
                        (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
                        (5, 9), (9, 13), (13, 17),           # 手掌
                    ]
                    hand_color = (0, 255, 0) if side == "left" else (255, 0, 255)
                    for c1, c2 in HAND_CONNECTIONS:
                        if c1 < len(landmarks) and c2 < len(landmarks):
                            x1h, y1h = int(landmarks[c1][0]), int(landmarks[c1][1])
                            x2h, y2h = int(landmarks[c2][0]), int(landmarks[c2][1])
                            cv2.line(frame, (x1h, y1h), (x2h, y2h), hand_color, 1)
                    for i, lm in enumerate(landmarks):
                        if i >= 21:
                            break
                        x, y = int(lm[0]), int(lm[1])
                        radius = 3 if i in [4, 8, 12, 16, 20] else 2  # 指尖大一点
                        cv2.circle(frame, (x, y), radius, hand_color, -1)

                # 如果是手势，在手腕处绘制特殊标记
                if person.gesture in ("hailing", "greeting", "hand_up"):
                    for wrist_idx in [9, 10]:  # 左右手腕
                        if wrist_idx < len(keypoints):
                            kp = keypoints[wrist_idx]
                            if len(kp) >= 3 and kp[2] > 0.3:
                                wx, wy = int(kp[0]), int(kp[1])
                                # 根据手势类型选择标记颜色
                                if person.gesture == "hailing":
                                    outer_color = (0, 0, 255)      # 红
                                    inner_color = (0, 165, 255)    # 橙
                                elif person.gesture == "greeting":
                                    outer_color = (255, 128, 0)    # 青
                                    inner_color = (0, 255, 255)    # 黄
                                else:
                                    outer_color = (0, 255, 255)    # 黄
                                    inner_color = (0, 0, 255)      # 红
                                cv2.circle(frame, (wx, wy), 15, outer_color, 3)
                                cv2.circle(frame, (wx, wy), 8, inner_color, -1)

                # 绘制手腕轨迹线
                trail = self._gesture_trails.get(person.track_id)
                if trail and len(trail) >= 2:
                    trail_color = {
                        "hailing": (0, 0, 255),
                        "greeting": (255, 128, 0),
                        "hand_up": (0, 255, 255),
                    }.get(person.gesture, (0, 200, 200))
                    pts = np.array(list(trail), np.int32)
                    cv2.polylines(frame, [pts], False, trail_color, 2)

        # 在左上角绘制统计信息
        info_lines = [
            f"Persons: {len(persons)}",
        ]
        active_gesture_count = sum(
            1 for p in persons if p.gesture in ("hailing", "greeting")
        )
        if active_gesture_count > 0:
            info_lines.append(f"GESTURE: {active_gesture_count}")

        y_offset = 25
        for line in info_lines:
            cv2.putText(
                frame,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            y_offset += 25

        return frame

    def _update_gesture_trail(self, person: PersonDetection) -> None:
        """记录手腕轨迹用于绘制轨迹线。"""
        trail = self._gesture_trails.get(person.track_id)
        if trail is None:
            trail = deque(maxlen=20)
            self._gesture_trails[person.track_id] = trail

        # 取左右手腕中置信度更高的一个
        kpts = person.keypoints
        left_wrist = kpts[9] if len(kpts) > 9 and kpts[9][2] > 0.3 else None
        right_wrist = kpts[10] if len(kpts) > 10 and kpts[10][2] > 0.3 else None

        if left_wrist is not None and right_wrist is not None:
            # 取置信度更高的
            wrist = left_wrist if left_wrist[2] > right_wrist[2] else right_wrist
        elif left_wrist is not None:
            wrist = left_wrist
        elif right_wrist is not None:
            wrist = right_wrist
        else:
            return

        trail.append((int(wrist[0]), int(wrist[1])))

        # 清理不存在的 track_id
        active_ids = set()
        # 注意：这里不清理，因为 get_performance_stats 会在别处调用

    def get_performance_stats(self) -> Dict[str, float]:
        """
        获取推理性能统计信息。

        Returns:
            包含平均推理时间、FPS等统计数据的字典
        """
        if not self._inference_times:
            return {"avg_inference_ms": 0.0, "fps": 0.0, "count": 0.0}

        return {
            "avg_inference_ms": round(
                sum(self._inference_times) / len(self._inference_times), 2
            ),
            "fps": round(
                1000.0
                / (sum(self._inference_times) / len(self._inference_times)),
                2,
            ),
            "count": float(len(self._inference_times)),
        }

    def __del__(self) -> None:
        """析构时释放MediaPipe资源。"""
        if self._mp_hands_instance is not None:
            try:
                self._mp_hands_instance.close()
            except Exception:
                pass

    def __repr__(self) -> str:
        return (
            f"PoseDetector(model={self.config.ai.yolo_model}, "
            f"conf={self.conf_threshold}, mediapipe={self.use_mediapipe})"
        )
