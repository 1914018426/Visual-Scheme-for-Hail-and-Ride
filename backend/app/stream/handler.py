"""
视频流处理器模块

StreamHandler 类负责从单一视频源持续读取帧，
使用独立线程进行采集，通过线程安全队列提供帧缓冲。
支持RTSP/RTMP/HTTP/本地文件/摄像头设备等多种协议，
具备自动重连机制确保流稳定性。
"""

import os
import time
import logging
import threading
from queue import Queue, Empty
from typing import Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from app.config import get_config

logger = logging.getLogger(__name__)


class StreamStatus(str, Enum):
    """视频流状态枚举。"""

    STOPPED = "stopped"      # 已停止
    CONNECTING = "connecting"  # 连接中
    CONNECTED = "connected"   # 已连接，正常读取
    ERROR = "error"         # 发生错误
    RECONNECTING = "reconnecting"  # 重连中


@dataclass
class StreamInfo:
    """视频流信息数据结构。"""

    source: str = ""                          # 视频源地址
    status: StreamStatus = StreamStatus.STOPPED  # 当前状态
    fps: float = 0.0                          # 实际帧率
    frame_count: int = 0                      # 已读取帧数
    width: int = 0                            # 帧宽度
    height: int = 0                           # 帧高度
    reconnect_count: int = 0                  # 重连次数
    last_error: str = ""                      # 最后错误信息
    start_time: float = 0.0                   # 启动时间戳


class StreamHandler:
    """
    视频流处理器

    独立线程读取视频帧，支持多种视频源协议，
    具备自动重连和帧缓冲功能。

    Attributes:
        source: 视频源地址（URL或设备ID）
        camera_id: 摄像头唯一标识
        info: 流信息对象
    """

    def __init__(self, source: str, camera_id: str = "") -> None:
        """
        初始化视频流处理器。

        Args:
            source: 视频源地址，支持:
                - RTSP: rtsp://192.168.1.100:554/stream
                - RTMP: rtmp://localhost/live/stream
                - HTTP: http://example.com/video.mp4
                - 本地文件: /path/to/video.mp4 或 ./video.avi
                - 摄像头设备ID: 0, 1, 2 (数字字符串)
            camera_id: 摄像头唯一标识，为空时自动从source生成
        """
        self.source = source
        self.camera_id = camera_id or source
        self.config = get_config()

        # 流信息
        self.info = StreamInfo(source=source)

        # 帧缓冲队列：低延迟模式下用更小队列，便于丢弃旧帧、追最新画面
        _qmax = self.config.stream.buffer_size
        if self.config.stream.low_latency_capture:
            _qmax = min(2, max(1, _qmax))
        self._frame_queue: Queue[np.ndarray] = Queue(maxsize=_qmax)

        # OpenCV视频捕获对象
        self._cap: Optional[cv2.VideoCapture] = None

        # 线程控制事件
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None

        # 帧率控制
        self._target_interval = 1.0 / max(self.config.stream.fps, 1)
        self._last_frame_time = 0.0

        logger.debug(
            "StreamHandler 初始化: camera_id=%s, source=%s", camera_id, source
        )

    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        """
        根据视频源协议创建对应的VideoCapture对象。

        Returns:
            成功返回VideoCapture对象，失败返回None
        """
        cap: Optional[cv2.VideoCapture] = None

        try:
            # 判断视频源类型
            if self.source.isdigit():
                # 摄像头设备（数字ID）
                device_id = int(self.source)
                cap = cv2.VideoCapture(device_id)
                logger.info("打开摄像头设备: %d", device_id)

            elif self.source.startswith(("rtsp://", "rtsps://")):
                # RTSP流：使用FFmpeg后端 + TCP传输 + 极小缓冲 以消除延迟
                # 必须在 VideoCapture 创建前设置环境变量
                os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                    "rtsp_transport;tcp|buffer_size;1024|max_delay;100000")
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                logger.info("打开RTSP流: %s (tcp/low-latency)", self.source)

            elif self.source.startswith(("rtmp://", "rtmps://")):
                # RTMP流：FFmpeg 默认缓冲 2-10 秒，必须显式关闭
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = \
                    "buffer_size;1024|max_delay;100000|fflags;nobuffer|flags;low_delay"
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                logger.info("打开RTMP流: %s (low-latency)", self.source)

            elif self.source.startswith(("http://", "https://")):
                # HTTP流：使用FFmpeg后端
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = \
                    "buffer_size;1024|max_delay;100000|fflags;nobuffer|flags;low_delay"
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                logger.info("打开HTTP流: %s (low-latency)", self.source)

            else:
                # 本地文件
                cap = cv2.VideoCapture(self.source)
                logger.info("打开本地文件: %s", self.source)

            # 验证是否成功打开
            if cap is not None and cap.isOpened():
                return cap
            else:
                if cap is not None:
                    cap.release()
                logger.error("无法打开视频源: %s", self.source)
                return None

        except Exception as e:
            logger.error("创建VideoCapture失败: %s - %s", self.source, str(e))
            if cap is not None:
                cap.release()
            return None

    def _configure_capture(self) -> None:
        """配置视频捕获参数（分辨率等）。"""
        if self._cap is None or not self._cap.isOpened():
            return

        # 设置分辨率（摄像头设备有效）
        if self.source.isdigit():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.stream.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.stream.height)
            self._cap.set(
                cv2.CAP_PROP_FPS, self.config.stream.fps
            )

        # 读取实际参数
        self.info.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.info.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.info.fps = self._cap.get(cv2.CAP_PROP_FPS) or self.config.stream.fps

    def _capture_loop(self) -> None:
        """
        帧采集线程的主循环。

        持续从视频源读取帧，放入缓冲队列。
        支持帧率控制和自动重连。
        """
        retry_count = 0
        max_retries = self.config.stream.reconnect_max_retries
        retry_interval = self.config.stream.reconnect_interval

        while not self._stop_event.is_set():
            # 创建或重新创建视频捕获
            self.info.status = StreamStatus.CONNECTING
            self._cap = self._create_capture()

            if self._cap is None:
                # 连接失败，进入重连逻辑
                retry_count += 1
                self.info.reconnect_count = retry_count
                self.info.status = StreamStatus.RECONNECTING
                self.info.last_error = f"连接失败，第{retry_count}次重连"

                if retry_count > max_retries:
                    logger.error(
                        "视频源 %s 重连次数超过上限(%d)，停止采集",
                        self.camera_id, max_retries,
                    )
                    self.info.status = StreamStatus.ERROR
                    break

                logger.warning(
                    "视频源 %s 连接失败，%d秒后第%d/%d次重连",
                    self.camera_id, retry_interval, retry_count, max_retries,
                )
                # 等待重连间隔（需检查停止事件）
                self._stop_event.wait(retry_interval)
                continue

            # 连接成功，配置参数
            self._configure_capture()
            self.info.status = StreamStatus.CONNECTED
            self.info.start_time = time.time()
            self.info.last_error = ""  # 清空历史错误
            retry_count = 0  # 重置重连计数
            logger.info(
                "视频源 %s 已连接: %dx%d @ %.1ffps",
                self.camera_id, self.info.width, self.info.height, self.info.fps,
            )

            # 帧读取循环
            is_network = not self.source.isdigit() and self.config.stream.low_latency_capture
            while not self._stop_event.is_set():
                # 防御性检查：stop() 可能已释放 _cap
                if self._cap is None:
                    break

                if is_network:
                    # 低延迟网络流：快速连续 read() 排空 FFmpeg 内部缓冲，
                    # 只保留最后一帧送入队列。FFmpeg 低延迟选项 (nobuffer/low_delay)
                    # 确保内部缓冲不超过 1-2 帧，read() 几乎不会阻塞。
                    latest_frame = None
                    latest_ret = False
                    drain_count = 0
                    while True:
                        ret, frame = self._cap.read()
                        if not ret or frame is None:
                            if drain_count == 0:
                                latest_ret = ret
                            break
                        latest_ret = ret
                        latest_frame = frame
                        drain_count += 1
                        # 安全上限：最多排空 20 帧，避免无限循环
                        if drain_count >= 20:
                            break
                    if not latest_ret or latest_frame is None:
                        break  # stream ended
                    ret, frame = latest_ret, latest_frame
                else:
                    ret, frame = self._cap.read()

                if not ret or frame is None:
                    # 读取失败，可能是流中断
                    logger.warning("视频源 %s 帧读取失败，准备重连", self.camera_id)
                    self.info.last_error = "帧读取失败"
                    break

                # 帧率控制：本地摄像头始终节流
                if not is_network:
                    current_time = time.time()
                    elapsed = current_time - self._last_frame_time
                    if elapsed < self._target_interval:
                        time.sleep(self._target_interval - elapsed)
                    self._last_frame_time = time.time()
                else:
                    self._last_frame_time = time.time()

                # 调整分辨率（如果需要）
                if (
                    frame.shape[1] != self.config.stream.width
                    or frame.shape[0] != self.config.stream.height
                ):
                    frame = cv2.resize(
                        frame,
                        (self.config.stream.width, self.config.stream.height),
                    )

                # 放入缓冲队列（队列满时丢弃最旧帧）
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except Empty:
                        pass
                self._frame_queue.put(frame)
                self.info.frame_count += 1

            # 退出读取循环，释放当前捕获
            if self._cap is not None:
                self._cap.release()
                self._cap = None

            # 如果不是主动停止，则标记为错误状态
            if not self._stop_event.is_set():
                self.info.status = StreamStatus.ERROR

        # 线程退出
        self.info.status = StreamStatus.STOPPED
        logger.info("视频源 %s 采集线程已退出", self.camera_id)

    def start(self) -> bool:
        """
        启动帧采集线程。

        Returns:
            True: 启动成功
            False: 启动失败（已在运行或source为空）
        """
        if not self.source:
            logger.error("视频源地址为空，无法启动")
            return False

        if self._capture_thread is not None and self._capture_thread.is_alive():
            logger.warning("视频源 %s 采集线程已在运行", self.camera_id)
            return False

        self._stop_event.clear()
        self.info.frame_count = 0
        self.info.reconnect_count = 0

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"Capture-{self.camera_id}",
            daemon=True,
        )
        self._capture_thread.start()

        logger.info("视频源 %s 采集线程已启动", self.camera_id)
        return True

    def stop(self) -> None:
        """停止帧采集线程并释放资源。"""
        logger.info("正在停止视频源 %s 采集线程...", self.camera_id)
        self._stop_event.set()

        # 关键修复：先释放 VideoCapture，强制阻塞在 read() 上的调用返回错误，
        # 避免 join(timeout=5.0) 超时后留下僵尸线程
        cap_to_release = self._cap
        self._cap = None
        if cap_to_release is not None:
            try:
                cap_to_release.release()
            except Exception:
                pass

        # 清空帧队列
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Empty:
                break

        # 等待线程结束
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5.0)
            if self._capture_thread.is_alive():
                logger.warning("视频源 %s 采集线程未能正常退出", self.camera_id)

        self.info.status = StreamStatus.STOPPED
        logger.info("视频源 %s 已停止", self.camera_id)

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        从缓冲队列获取一帧图像。

        Args:
            timeout: 等待超时时间（秒）

        Returns:
            成功返回numpy.ndarray图像(BGR格式)，超时返回None
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except Empty:
            return None

    def get_frame_nowait(self) -> Optional[np.ndarray]:
        """
        非阻塞方式获取最新帧（不等待）。

        Returns:
            成功返回numpy.ndarray图像，队列为空返回None
        """
        # 取队列中最新的一帧（丢弃旧帧）
        latest: Optional[np.ndarray] = None
        while not self._frame_queue.empty():
            try:
                latest = self._frame_queue.get_nowait()
            except Empty:
                break
        return latest

    def is_running(self) -> bool:
        """
        检查采集线程是否在运行。

        Returns:
            True: 线程正在运行
            False: 线程已停止
        """
        return (
            self._capture_thread is not None
            and self._capture_thread.is_alive()
            and not self._stop_event.is_set()
        )

    def is_connected(self) -> bool:
        """
        检查视频流是否已连接。

        Returns:
            True: 已连接且有帧可读
            False: 未连接
        """
        return self.info.status == StreamStatus.CONNECTED

    @property
    def queue_size(self) -> int:
        """返回当前缓冲队列中的帧数。"""
        return self._frame_queue.qsize()

    def __repr__(self) -> str:
        return (
            f"StreamHandler(camera_id={self.camera_id}, "
            f"source={self.source}, status={self.info.status.value})"
        )
