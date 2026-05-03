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
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque

import cv2
import numpy as np
import threading


@dataclass
class TrailFrame:
    """单帧轨迹数据：存储 wrist_local 及对应的人体标架快照（用于精确反投影）。"""

    wrist_local: Tuple[float, float]   # EMA 平滑后的局部坐标
    origin: Tuple[float, float]        # EMA 平滑后的肩中点（像素）
    e_x: Tuple[float, float]           # 肩宽方向单位向量
    e_y: Tuple[float, float]           # 躯干方向单位向量
    torso_scale: float                 # EMA 平滑后的躯干尺度（像素）
    ts: float                          # 帧时间戳（秒）

from app.config import get_config, _env_bool

logger = logging.getLogger(__name__)

# 国内镜像拉取权重失败时的最小文件体积（字节），防止下到错误页
_MIN_WEIGHT_BYTES = 100_000


def _download_yolo_weights_cn(dest_path: str, weight_basename: str) -> bool:
    """
    从国内可访问的 Hugging Face 镜像下载 YOLO 权重到 dest_path（hf-mirror / 腾讯云 / 交大镜像），
    最后兜底为 GitHub Release 直连；若环境配置了代理则代理保底。
    大文件使用 curl 落盘（避免整包读入内存）。
    """
    name = os.path.basename(weight_basename.strip())
    if not name.endswith(".pt"):
        return False

    # YOLOv8 / YOLO11 均可能使用的 release tag
    tags = ("v8.4.0", "v8.3.0", "v8.2.0", "v8.1.0", "v8.0.0")
    # 扩展 repo 以覆盖 YOLO11 可能的存放路径
    repos = (
        "Ultralytics/YOLOv8",
        "Ultralytics/yolov8",
        "Ultralytics/yolo11",
        "Ultralytics/YOLO11",
    )
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

    proxy = (
        os.environ.get("ALL_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("HTTP_PROXY")
    )

    def _curl_download(url: str, use_proxy: bool = False) -> bool:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        logger.info("开始下载权重(curl): %s (proxy=%s)", url, use_proxy)
        cmd = [
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
        ]
        if use_proxy and proxy:
            cmd.extend(["--proxy", proxy])
        try:
            proc = subprocess.run(
                cmd,
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

    # 第一轮：国内镜像，不走代理
    for url in urls:
        if _curl_download(url, use_proxy=False):
            return True

    # 第二轮：兜底 GitHub 直连 + 代理保底（若配置了代理）
    if proxy:
        logger.info("国内镜像全部失败，尝试使用代理下载...")
        for url in urls:
            if _curl_download(url, use_proxy=True):
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
    left_palm_normal: Optional[np.ndarray] = None
    right_palm_normal: Optional[np.ndarray] = None

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
    姿态检测器 (YOLO11-Pose + ByteTrack + MediaPipe Hands)

    基于 YOLO11-Pose 和 MediaPipe Hands 的完整检测流水线。
    提供人体检测、姿态估计、多目标跟踪、骨骼绘制和手势分析功能。

    Attributes:
        model: YOLO11-Pose 模型实例
        use_mediapipe: 是否启用 MediaPipe 手部检测
        use_tracking: 是否启用 ByteTrack 多目标跟踪
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

        # 加载 YOLO11-Pose 模型
        self.model = self._load_yolo_model()

        # 可选加载 MediaPipe Hands（作为手部关键点补充）
        self._mp_hands = None
        self._mp_hands_instance = None
        if self.use_mediapipe:
            self._load_mediapipe_hands()

        # ByteTrack 多目标跟踪配置
        self.use_tracking = getattr(self.config.ai, "enable_tracking", True)
        self.tracker_config = os.path.join(
            os.path.dirname(__file__), "bytetrack.yaml"
        )

        # 推理性能统计
        self._inference_times: List[float] = []
        self._stats_window_size = 30
        # 手势轨迹：camera_id -> {track_id_side -> deque[TrailFrame, ...]}
        # 存储每帧的 wrist_local + 人体标架快照，用于精确反投影
        self._gesture_trails: Dict[str, Dict[str, deque]] = {}
        # 轨迹最后更新时间，用于时间基准 GC：camera_id -> {track_id_side -> timestamp}
        self._trail_last_update: Dict[str, Dict[str, float]] = {}
        # MediaPipe 降采样计数器：track_id -> {side -> counter}
        self._mp_skip_counter: Dict[str, Dict[str, int]] = {}
        # 躯干关键点 EMA 平滑（车辆振动补偿）：track_id -> {keypoint_idx_str -> xy_array}
        self._torso_smoother: Dict[str, Dict[str, np.ndarray]] = {}
        # 手势识别器（懒加载，支持 triplelock / transformer / hybrid 模式）
        self._recognizer = None

        logger.info(
            "PoseDetector 初始化完成: model=%s, conf=%.2f, max_det=%d, "
            "half=%s, tracking=%s, mediapipe=%s",
            self.config.ai.yolo_model,
            self.conf_threshold,
            self.max_detections,
            self.config.ai.inference_half,
            self.use_tracking,
            self.use_mediapipe,
        )

    def _load_yolo_model(self):
        """
        加载 YOLO11-Pose 模型。

        权重优先落在 MODEL_DIR；缺失时先走国内镜像下载到本地，再交给 Ultralytics。
        国内镜像全部失败时，若环境配置了代理，将使用代理保底从 GitHub 下载。

        Returns:
            YOLO: YOLO11-Pose 模型实例
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
                if not _download_yolo_weights_cn(model_path, raw):
                    logger.error(
                        "所有下载源均失败。请手动下载 %s 并放置到 %s，"
                        "或配置 HTTP_PROXY/HTTPS_PROXY 环境变量后重试。",
                        raw,
                        model_path,
                    )
                    raise RuntimeError(
                        f"无法下载模型 {raw}。请检查网络或配置代理。"
                    )

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
            logger.info("YOLO11-Pose 模型加载成功: %s", model_name)
            return model

        except Exception as e:
            logger.error("YOLO11-Pose 模型加载失败: %s", str(e))
            raise RuntimeError(f"无法加载YOLO11-Pose模型: {e}") from e

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
            # MediaPipe Hands.process() 非线程安全，多路并发会串扰内部图计算状态
            self._mp_lock = threading.Lock()
            logger.info("MediaPipe Hands 加载成功")
        except Exception as e:
            logger.warning("MediaPipe Hands 加载失败，将禁用手部检测: %s", str(e))
            self.use_mediapipe = False
            self._mp_hands = None
            self._mp_hands_instance = None
            self._mp_lock = None

    def detect_persons(self, frame: np.ndarray) -> List[PersonDetection]:
        """
        检测图像中的所有人。

        当 enable_tracking=True 时，使用 ByteTrack 进行多目标跟踪，
        输出稳定的 track_id，替代原有轻量 bbox 中心点关联算法。

        Args:
            frame: 输入图像 (BGR格式)

        Returns:
            PersonDetection对象列表
        """
        if frame is None or frame.size == 0:
            return []

        results: List[PersonDetection] = []

        try:
            if self.use_tracking:
                # ByteTrack 跟踪推理（推荐）
                yolo_results = self.model.track(
                    frame,
                    conf=self.conf_threshold,
                    max_det=self.max_detections,
                    verbose=False,
                    imgsz=self.config.ai.inference_imgsz,
                    half=self.config.ai.inference_half,
                    tracker=self.tracker_config,
                    persist=True,
                )
            else:
                # 纯检测模式（无跟踪）
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
                            kpt_array = kpt_array[0]
                    else:
                        kpt_array = np.array(kpt_array)
                        if kpt_array.ndim == 3:
                            kpt_array = kpt_array[0]

                    # 确保关键点数组形状为 (17, 3)
                    if kpt_array.ndim == 1:
                        kpt_array = kpt_array.reshape(-1, 3)

                    # ByteTrack track_id（稳定跨帧）
                    track_id = f"person_{i}"
                    if self.use_tracking and box.id is not None:
                        tid = int(box.id[0]) if hasattr(box.id, '__len__') else int(box.id)
                        track_id = f"person_{tid}"

                    person = PersonDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        keypoints=kpt_array,
                        track_id=track_id,
                    )
                    results.append(person)

        except Exception as e:
            logger.error("人体检测失败: %s", str(e))

        return results

    def _detect_palm_for_person(
        self, frame: np.ndarray, person: PersonDetection
    ) -> None:
        """
        对单个人体检测手掌法向量，按左右手分别检测存入 PersonDetection。

        策略：
        1. MediaPipe 降采样：手臂未抬起时每3帧调用，抬起后每帧调用
        2. 只返回手掌平面法向量 n（3维单位向量），禁止向上层暴露21点 landmark
        3. landmarks 仅保留在 PersonDetection 中用于本模块可视化
        """
        if self._mp_hands_instance is None:
            return

        kpts = person.keypoints
        h, w = frame.shape[:2]

        # 肩宽用于 ROI
        left_shoulder = kpts[5] if len(kpts) > 5 and kpts[5][2] > 0.3 else None
        right_shoulder = kpts[6] if len(kpts) > 6 and kpts[6][2] > 0.3 else None
        shoulder_width = 80.0
        if left_shoulder is not None and right_shoulder is not None:
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        roi_size = int(max(160, shoulder_width * 1.5))

        detected_any = False

        for side, w_idx in [("left", 9), ("right", 10)]:
            if len(kpts) <= w_idx or kpts[w_idx][2] < 0.3:
                continue

            # ---- 降采样决策 ----
            track_counters = self._mp_skip_counter.setdefault(person.track_id, {})
            counter = track_counters.get(side, 0)

            # 判断手臂是否抬起（优先使用帧级 TNLF 缓存）
            cache_key = f"{person.track_id}_{side}"
            cached = getattr(self, '_tnlf_frame_cache', {}).get(cache_key)
            if cached is not None:
                wl, _, _, _, _, valid = cached
            else:
                from app.ai.local_frame import wrist_to_local_frame
                wl, _, valid = wrist_to_local_frame(kpts, side=side)
            is_arm_raised = valid and wl is not None and wl[1] < -0.2  # 手腕在肩中点上方 0.2 躯干长度

            interval = 1 if is_arm_raised else 3
            counter += 1
            if counter < interval:
                track_counters[side] = counter
                continue
            track_counters[side] = 0

            wrist = kpts[w_idx]
            wx, wy = int(wrist[0]), int(wrist[1])

            rx1 = max(0, wx - roi_size // 2)
            ry1 = max(0, wy - roi_size // 2)
            rx2 = min(w, wx + roi_size // 2)
            ry2 = min(h, wy + roi_size // 2)

            if rx2 <= rx1 or ry2 <= ry1:
                continue

            roi = frame[ry1:ry2, rx1:rx2]
            if roi.size == 0:
                continue

            try:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                # MediaPipe 实例非线程安全，用锁保护 process()
                with self._mp_lock:
                    results = self._mp_hands_instance.process(roi_rgb)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]

                    # 转换回全图绝对坐标（仅用于可视化）
                    abs_landmarks: List[Tuple[float, float, float]] = []
                    for lm in hand_landmarks.landmark:
                        abs_x = lm.x * (rx2 - rx1) + rx1
                        abs_y = lm.y * (ry2 - ry1) + ry1
                        abs_z = lm.z
                        abs_landmarks.append((abs_x, abs_y, abs_z))

                    if side == "left":
                        person.left_hand_landmarks = abs_landmarks
                    else:
                        person.right_hand_landmarks = abs_landmarks

                    # 计算法向量（向上层暴露的唯一信息）
                    from app.ai.slerp import compute_palm_normal
                    n, ok = compute_palm_normal(abs_landmarks)
                    if ok and n is not None:
                        if side == "left":
                            person.left_palm_normal = n
                        else:
                            person.right_palm_normal = n
                        detected_any = True

            except Exception as e:
                logger.warning("MediaPipe %s hand detection failed: %s", side, str(e))

        if detected_any:
            logger.info(
                "MediaPipe Hands: track=%s left=%s right=%s",
                person.track_id,
                "Y" if person.left_palm_normal is not None else "N",
                "Y" if person.right_palm_normal is not None else "N",
            )

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

        # Phase 1: GPU 推理（YOLO + ByteTrack）
        persons = self.detect_persons(frame)
        # Phase 2: CPU 后处理（MediaPipe + gesture + 绘制 + 统计）
        return self.process_persons(frame, persons, camera_id=camera_id)

    def process_persons(
        self, frame: np.ndarray, persons: List[PersonDetection], camera_id: str = ""
    ) -> Tuple[np.ndarray, DetectionResult]:
        """
        CPU 后处理阶段：手部检测 + 手势识别 + 轨迹 + 绘制 + 性能统计。

        与 process_frame 分离以允许 GPU 锁仅覆盖 detect_persons，
        CPU 密集的部分在锁外并行执行。

        Args:
            frame: 输入图像 (BGR格式)
            persons: detect_persons 返回的人物检测列表
            camera_id: 摄像头标识

        Returns:
            (绘制后的帧, 检测结果)
        """
        if frame is None or frame.size == 0:
            return frame, DetectionResult(frame_shape=(0, 0))

        start_time = time.time()
        h, w = frame.shape[:2]

        # 预计算 TNLF 结果，避免 _detect_palm_for_person 和 _update_gesture_trail 重复计算
        from app.ai.local_frame import wrist_to_local_frame_full
        self._tnlf_frame_cache: Dict[str, tuple] = {}
        for person in persons:
            for side in ["left", "right"]:
                key = f"{person.track_id}_{side}"
                result = wrist_to_local_frame_full(person.keypoints, side=side)
                self._tnlf_frame_cache[key] = result

        # 1. 手部关键点检测（MediaPipe Hands，降采样）
        if self.use_mediapipe:
            for person in persons:
                self._detect_palm_for_person(frame, person)

        # 2. 手势识别（为每个人检测手势，含车辆振动补偿）
        from app.ai.gesture import _choose_recognizer, GestureType

        if self._recognizer is None:
            self._recognizer = _choose_recognizer()

        recognizer = self._recognizer

        # 车辆振动补偿：对躯干关键点做 EMA 平滑，消除平台抖动对 TNLF 的影响
        TORSO_EMA_ALPHA = 0.35
        TORSO_INDICES = [5, 6, 11, 12]  # L/R shoulder, L/R hip
        frame_ts = time.time()
        active_track_ids = {p.track_id for p in persons}
        # GC torso smoothers for stale tracks
        stale_torso = [tid for tid in self._torso_smoother if tid not in active_track_ids]
        for tid in stale_torso:
            del self._torso_smoother[tid]

        for person in persons:
            tid = person.track_id
            # 初始化或更新躯干 EMA
            if tid not in self._torso_smoother:
                self._torso_smoother[tid] = {
                    str(idx): person.keypoints[idx][:2].copy()
                    for idx in TORSO_INDICES
                }
            else:
                prev = self._torso_smoother[tid]
                for idx in TORSO_INDICES:
                    if person.keypoints[idx][2] > 0.3:
                        prev[str(idx)] = (
                            TORSO_EMA_ALPHA * person.keypoints[idx][:2]
                            + (1.0 - TORSO_EMA_ALPHA) * prev[str(idx)]
                        )

            # 构造平滑后的 keypoints 副本供 gesture 识别使用
            sm_kpts = person.keypoints.copy()
            if tid in self._torso_smoother:
                for idx in TORSO_INDICES:
                    sm_kpts[idx, :2] = self._torso_smoother[tid][str(idx)]

            # 从 TNLF 缓存提取特征（供 Transformer 引擎使用）
            left_wl = right_wl = None
            left_valid = right_valid = False
            left_vmag = right_vmag = 0.0
            left_t1 = left_t2 = left_ext = 0.0
            right_t1 = right_t2 = right_ext = 0.0

            for side in ["left", "right"]:
                cache_key = f"{tid}_{side}"
                cached = self._tnlf_frame_cache.get(cache_key)
                if cached is not None:
                    wl, _, _, _, _, valid = cached
                else:
                    wl, _, valid = None, None, False

                if side == "left":
                    left_wl = np.array(wl) if wl is not None else None
                    left_valid = valid
                else:
                    right_wl = np.array(wl) if wl is not None else None
                    right_valid = valid

            # 计算速度（基于 TNLF 缓存中的历史 wrist_local）
            for side, wl_curr in [("left", left_wl), ("right", right_wl)]:
                if wl_curr is None:
                    continue
                trail_key = f"{tid}_{side}"
                cam_trails = self._gesture_trails.get(camera_id, {})
                trail = cam_trails.get(trail_key)
                if trail and len(trail) >= 2:
                    prev_frame = trail[-1]
                    dt = frame_ts - getattr(prev_frame, 'ts', frame_ts - 0.067)
                    if dt > 0 and dt < 1.0:
                        prev_wl = np.array(prev_frame.wrist_local)
                        v_mag = float(np.linalg.norm(wl_curr - prev_wl) / dt)
                    else:
                        v_mag = 0.0
                else:
                    v_mag = 0.0
                if side == "left":
                    left_vmag = v_mag
                else:
                    right_vmag = v_mag

            # 计算手臂角度
            from app.ai.transformer.model import compute_arm_angles
            import torch as _torch
            kpts_t = _torch.tensor(sm_kpts)
            try:
                right_t1, right_t2, right_ext = compute_arm_angles(kpts_t, "right")
                left_t1, left_t2, left_ext = compute_arm_angles(kpts_t, "left")
            except Exception:
                pass

            # 根据识别器类型选择调用方式
            from app.ai.gesture import TransformerGestureRecognizer, HybridGestureRecognizer, SimpleGestureRecognizer, SimpleTransformerHybridRecognizer
            if isinstance(recognizer, (TransformerGestureRecognizer, HybridGestureRecognizer, SimpleGestureRecognizer, SimpleTransformerHybridRecognizer)):
                result = recognizer.recognize(
                    sm_kpts, person.track_id,
                    left_palm_normal=person.left_palm_normal,
                    right_palm_normal=person.right_palm_normal,
                    frame_timestamp=frame_ts,
                    active_track_ids=active_track_ids,
                    left_wrist_local=left_wl,
                    right_wrist_local=right_wl,
                    left_tnlf_valid=left_valid,
                    right_tnlf_valid=right_valid,
                    left_velocity_mag=left_vmag,
                    right_velocity_mag=right_vmag,
                    left_theta1=left_t1, left_theta2=left_t2, left_ext_ratio=left_ext,
                    right_theta1=right_t1, right_theta2=right_t2, right_ext_ratio=right_ext,
                )
                gesture_type = result.gesture_type.value if isinstance(result.gesture_type, GestureType) else str(result.gesture_type)
                gesture_conf = result.confidence
            else:
                # TripleLock 引擎使用 is_hailing_gesture 包装函数
                from app.ai.gesture import is_hailing_gesture
                gesture_type, gesture_conf = is_hailing_gesture(
                    sm_kpts,
                    person.track_id,
                    left_palm_normal=person.left_palm_normal,
                    right_palm_normal=person.right_palm_normal,
                    frame_timestamp=frame_ts,
                    active_track_ids=active_track_ids,
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
            self._update_gesture_trail(person, camera_id, frame_ts)

        # 3. 绘制骨骼和手势标记
        annotated_frame = self.draw_skeleton(frame.copy(), persons, camera_id)

        # 4. 清理不活跃的轨迹和降采样计数器
        active_ids = {p.track_id for p in persons}
        cam_trails = self._gesture_trails.get(camera_id)
        if cam_trails:
            stale = [
                key for key in cam_trails
                if key.rsplit("_", 1)[0] not in active_ids
            ]
            for key in stale:
                del cam_trails[key]
                # 同步清理 trail 时间戳
                if camera_id in self._trail_last_update:
                    self._trail_last_update[camera_id].pop(key, None)
        # 清理降采样计数器（嵌套结构）
        stale_ids = [tid for tid in self._mp_skip_counter if tid not in active_ids]
        for tid in stale_ids:
            del self._mp_skip_counter[tid]

        # 时间基准 GC：清理所有摄像头中超过 TRAIL_MAX_AGE 的轨迹
        TRAIL_MAX_AGE = 10.0
        now = time.time()
        for cam_id in list(self._gesture_trails.keys()):
            last_updates = self._trail_last_update.get(cam_id, {})
            stale_keys = [
                k for k in self._gesture_trails[cam_id]
                if now - last_updates.get(k, 0) > TRAIL_MAX_AGE
            ]
            for k in stale_keys:
                del self._gesture_trails[cam_id][k]
                last_updates.pop(k, None)
            if not self._gesture_trails[cam_id]:
                del self._gesture_trails[cam_id]
                self._trail_last_update.pop(cam_id, None)

        # 5. 计算推理性能
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

        # 清理帧级 TNLF 缓存
        if hasattr(self, '_tnlf_frame_cache'):
            self._tnlf_frame_cache.clear()

        return annotated_frame, result

    def draw_skeleton(
        self,
        frame: np.ndarray,
        persons: List[PersonDetection],
        camera_id: str = "",
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
            if person.gesture == "waving":
                box_color = (0, 0, 255)    # 红色 - 招手
                label = f"Waving {person.gesture_conf:.2f}"
            elif person.gesture == "hailing":
                box_color = (0, 0, 255)    # 红色 - 打车（兼容旧值）
                label = f"Waving {person.gesture_conf:.2f}"
            elif person.gesture == "greeting":
                box_color = (255, 128, 0)  # 橙色/青色 - 打招呼（兼容旧值）
                label = f"Waving {person.gesture_conf:.2f}"
            elif person.gesture == "hand_up":
                box_color = (128, 128, 128)  # 灰色 - 举手
                label = f"HandUp {person.gesture_conf:.2f}"
            else:
                box_color = (0, 255, 0)    # 绿色 - 无手势
                label = f"Person {person.confidence:.2f}"

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
                if person.gesture in ("waving", "hailing", "greeting", "hand_up"):
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

                # 绘制手腕轨迹线（左右手各一条）
                from app.ai.local_frame import local_to_pixel_with_frame

                cam_trails = self._gesture_trails.get(camera_id, {})
                for side in ["left", "right"]:
                    trail_key = f"{person.track_id}_{side}"
                    trail = cam_trails.get(trail_key)
                    if trail and len(trail) >= 2:
                        trail_color = {
                            "waving": (0, 0, 255),
                            "hailing": (0, 0, 255),
                            "greeting": (0, 0, 255),
                            "hand_up": (0, 255, 255),
                        }.get(person.gesture, (0, 200, 200))
                        # 使用最新帧的统一标架反投影所有历史点，消除车辆移动导致的轨迹漂移
                        ref = trail[-1]
                        origin_arr = np.array(ref.origin)
                        ex_arr = np.array(ref.e_x)
                        ey_arr = np.array(ref.e_y)
                        pixel_pts = []
                        for tf in trail:
                            px = local_to_pixel_with_frame(
                                tf.wrist_local,
                                origin_arr,
                                ex_arr,
                                ey_arr,
                                ref.torso_scale,
                            )
                            if px is not None:
                                pixel_pts.append(px)
                        # 连续性保护：相邻点距离过大时断线
                        if len(pixel_pts) >= 2:
                            segments = []
                            current_seg = [pixel_pts[0]]
                            for i in range(1, len(pixel_pts)):
                                dx = pixel_pts[i][0] - pixel_pts[i-1][0]
                                dy = pixel_pts[i][1] - pixel_pts[i-1][1]
                                if (dx*dx + dy*dy) ** 0.5 > 3.0 * ref.torso_scale:
                                    # 断线
                                    if len(current_seg) >= 2:
                                        segments.append(current_seg)
                                    current_seg = [pixel_pts[i]]
                                else:
                                    current_seg.append(pixel_pts[i])
                            if len(current_seg) >= 2:
                                segments.append(current_seg)
                            for seg in segments:
                                pts = np.array(seg, np.int32)
                                cv2.polylines(frame, [pts], False, trail_color, 4)
                                # 加粗轨迹：在轨迹点周围绘制小圆点增强可见性
                                for p in seg:
                                    cv2.circle(frame, (int(p[0]), int(p[1])), 3, trail_color, -1)

        return frame

    def _update_gesture_trail(self, person: PersonDetection, camera_id: str = "", frame_ts: float = 0.0) -> None:
        """记录手腕轨迹（人体局部坐标 wrist_local + 标架快照），用于绘制轨迹线。

        对 origin / torso_scale / wrist_local 做 EMA 平滑，消除 YOLO 关键点
        帧间抖动导致的轨迹跳动。
        """
        from app.ai.local_frame import wrist_to_local_frame_full

        EMA_ALPHA = 0.5  # 折中：快速响应 vs 平滑噪声

        if camera_id not in self._gesture_trails:
            self._gesture_trails[camera_id] = {}
        cam_trails = self._gesture_trails[camera_id]

        for side in ["left", "right"]:
            trail_key = f"{person.track_id}_{side}"
            trail = cam_trails.get(trail_key)
            if trail is None:
                trail = deque(maxlen=15)
                cam_trails[trail_key] = trail

            # 优先使用帧级 TNLF 缓存
            cache_key = f"{person.track_id}_{side}"
            cached = getattr(self, '_tnlf_frame_cache', {}).get(cache_key)
            if cached is not None:
                wl, origin, e_x, e_y, torso_scale, valid = cached
            else:
                wl, origin, e_x, e_y, torso_scale, valid = wrist_to_local_frame_full(
                    person.keypoints, side=side
                )
            if not valid or wl is None or origin is None:
                continue

            # ---- EMA 平滑 ----
            EMA_ALPHA = 0.25  # 更强的平滑，抑制 YOLO 关键点检测抖动
            if trail:
                prev = trail[-1]
                # 对 origin 平滑（消除肩中点抖动）
                ox = EMA_ALPHA * origin[0] + (1 - EMA_ALPHA) * prev.origin[0]
                oy = EMA_ALPHA * origin[1] + (1 - EMA_ALPHA) * prev.origin[1]
                origin = (ox, oy)
                # 对 torso_scale 平滑（消除躯干尺度抖动）
                torso_scale = EMA_ALPHA * torso_scale + (1 - EMA_ALPHA) * prev.torso_scale
                # 对 wrist_local 平滑（消除局部坐标抖动）
                wx = EMA_ALPHA * wl[0] + (1 - EMA_ALPHA) * prev.wrist_local[0]
                wy = EMA_ALPHA * wl[1] + (1 - EMA_ALPHA) * prev.wrist_local[1]
                wl = (wx, wy)

            # 运动距离过滤（基于平滑后的 wrist_local，阈值 1.5 torso_units）
            # 车辆颠簸时 YOLO 关键点可能大幅跳动，放宽阈值避免轨迹被清空
            if trail:
                last_wl = trail[-1].wrist_local
                dist = ((wl[0] - last_wl[0]) ** 2 + (wl[1] - last_wl[1]) ** 2) ** 0.5
                if dist > 1.5:
                    trail.clear()

            frame = TrailFrame(
                wrist_local=wl,
                origin=(float(origin[0]), float(origin[1])),
                e_x=(float(e_x[0]), float(e_x[1])),
                e_y=(float(e_y[0]), float(e_y[1])),
                torso_scale=float(torso_scale),
                ts=frame_ts,
            )
            trail.append(frame)
            # 记录更新时间用于时间基准 GC
            if camera_id not in self._trail_last_update:
                self._trail_last_update[camera_id] = {}
            self._trail_last_update[camera_id][trail_key] = time.time()

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
            f"conf={self.conf_threshold}, half={self.config.ai.inference_half}, "
            f"tracking={self.use_tracking}, mediapipe={self.use_mediapipe})"
        )
