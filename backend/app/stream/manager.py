"""
视频流管理器模块

StreamManager 类负责管理多路视频流（最多4路），
提供统一的添加、删除、查询和控制接口。
是上层API与StreamHandler之间的中间管理层。
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import numpy as np

from app.stream.handler import StreamHandler, StreamStatus
from app.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class CameraStatus:
    """摄像头状态信息。"""

    camera_id: str = ""
    source: str = ""
    status: str = "stopped"
    fps: float = 0.0
    frame_count: int = 0
    width: int = 0
    height: int = 0
    reconnect_count: int = 0
    queue_size: int = 0
    last_error: str = ""


class StreamManager:
    """
    视频流管理器

    管理多路视频流，支持最多4路摄像头同时接入。
    提供流的添加、删除、查询和批量控制功能。

    Attributes:
        max_cameras: 最大支持的摄像头数量
        handlers: 存储所有StreamHandler的字典
    """

    def __init__(self, max_cameras: int = 4) -> None:
        """
        初始化视频流管理器。

        Args:
            max_cameras: 最大支持的摄像头数量，默认4路
        """
        self.config = get_config()
        self.max_cameras = max_cameras
        # 存储所有视频流处理器: {camera_id: StreamHandler}
        self._handlers: Dict[str, StreamHandler] = {}

        logger.info("StreamManager 初始化完成，最大支持 %d 路摄像头", max_cameras)

    def add_stream(self, camera_id: str, source: str) -> bool:
        """
        添加一路视频流。

        Args:
            camera_id: 摄像头唯一标识（如: front, back, left, right）
            source: 视频源地址（URL或设备ID）

        Returns:
            True: 添加成功
            False: 添加失败（ID已存在或超出最大数量限制）

        Raises:
            ValueError: camera_id或source为空
        """
        if not camera_id or not source:
            raise ValueError("camera_id 和 source 不能为空")

        if camera_id in self._handlers:
            logger.warning("摄像头 %s 已存在，请先删除后重新添加", camera_id)
            return False

        if len(self._handlers) >= self.max_cameras:
            logger.error(
                "已达到最大摄像头数量限制(%d)，无法添加 %s",
                self.max_cameras, camera_id,
            )
            return False

        handler = StreamHandler(source=source, camera_id=camera_id)
        self._handlers[camera_id] = handler
        logger.info("摄像头 %s (源: %s) 已添加", camera_id, source)
        return True

    def remove_stream(self, camera_id: str) -> bool:
        """
        删除一路视频流。

        Args:
            camera_id: 摄像头唯一标识

        Returns:
            True: 删除成功
            False: 摄像头不存在
        """
        handler = self._handlers.pop(camera_id, None)
        if handler is None:
            logger.warning("摄像头 %s 不存在，无法删除", camera_id)
            return False

        handler.stop()
        logger.info("摄像头 %s 已删除", camera_id)
        return True

    def get_frame(
        self, camera_id: str, timeout: float = 1.0
    ) -> Optional[np.ndarray]:
        """
        获取指定摄像头的最新帧。

        Args:
            camera_id: 摄像头唯一标识
            timeout: 获取超时时间（秒）

        Returns:
            成功返回numpy.ndarray图像，失败返回None
        """
        handler = self._handlers.get(camera_id)
        if handler is None:
            logger.debug("摄像头 %s 不存在", camera_id)
            return None
        return handler.get_frame(timeout=timeout)

    def get_frame_nowait(self, camera_id: str) -> Optional[np.ndarray]:
        """
        非阻塞获取指定摄像头的最新帧。

        Args:
            camera_id: 摄像头唯一标识

        Returns:
            成功返回numpy.ndarray图像，失败返回None
        """
        handler = self._handlers.get(camera_id)
        if handler is None:
            return None
        return handler.get_frame_nowait()

    def start_stream(self, camera_id: str) -> bool:
        """
        启动指定摄像头的视频流。

        Args:
            camera_id: 摄像头唯一标识

        Returns:
            True: 启动成功
            False: 启动失败或摄像头不存在
        """
        handler = self._handlers.get(camera_id)
        if handler is None:
            logger.warning("摄像头 %s 不存在，无法启动", camera_id)
            return False
        return handler.start()

    def stop_stream(self, camera_id: str) -> bool:
        """
        停止指定摄像头的视频流。

        Args:
            camera_id: 摄像头唯一标识

        Returns:
            True: 停止成功
            False: 摄像头不存在
        """
        handler = self._handlers.get(camera_id)
        if handler is None:
            logger.warning("摄像头 %s 不存在，无法停止", camera_id)
            return False
        handler.stop()
        return True

    def start_all(self) -> Dict[str, bool]:
        """
        启动所有已添加的视频流。

        Returns:
            字典，key为camera_id，value为启动结果
        """
        results: Dict[str, bool] = {}
        for camera_id, handler in self._handlers.items():
            logger.info("正在启动摄像头 %s ...", camera_id)
            results[camera_id] = handler.start()
        return results

    def stop_all(self) -> None:
        """停止所有视频流并清理资源。"""
        logger.info("正在停止所有摄像头...")
        for camera_id, handler in self._handlers.items():
            logger.info("停止摄像头 %s", camera_id)
            handler.stop()
        logger.info("所有摄像头已停止")

    def list_cameras(self) -> List[CameraStatus]:
        """
        获取所有摄像头的状态列表。

        Returns:
            CameraStatus对象列表
        """
        statuses: List[CameraStatus] = []
        for camera_id, handler in self._handlers.items():
            info = handler.info
            status = CameraStatus(
                camera_id=camera_id,
                source=handler.source,
                status=info.status.value,
                fps=info.fps,
                frame_count=info.frame_count,
                width=info.width,
                height=info.height,
                reconnect_count=info.reconnect_count,
                queue_size=handler.queue_size,
                last_error=info.last_error,
            )
            statuses.append(status)
        return statuses

    def get_camera_status(self, camera_id: str) -> Optional[CameraStatus]:
        """
        获取指定摄像头的状态。

        Args:
            camera_id: 摄像头唯一标识

        Returns:
            CameraStatus对象，不存在返回None
        """
        handler = self._handlers.get(camera_id)
        if handler is None:
            return None
        info = handler.info
        return CameraStatus(
            camera_id=camera_id,
            source=handler.source,
            status=info.status.value,
            fps=info.fps,
            frame_count=info.frame_count,
            width=info.width,
            height=info.height,
            reconnect_count=info.reconnect_count,
            queue_size=handler.queue_size,
            last_error=info.last_error,
        )

    def get_handler(self, camera_id: str) -> Optional[StreamHandler]:
        """
        获取指定摄像头的StreamHandler实例。

        Args:
            camera_id: 摄像头唯一标识

        Returns:
            StreamHandler对象，不存在返回None
        """
        return self._handlers.get(camera_id)

    def has_camera(self, camera_id: str) -> bool:
        """
        检查指定摄像头是否存在。

        Args:
            camera_id: 摄像头唯一标识

        Returns:
            True: 存在
            False: 不存在
        """
        return camera_id in self._handlers

    @property
    def camera_count(self) -> int:
        """当前已添加的摄像头数量。"""
        return len(self._handlers)

    @property
    def camera_ids(self) -> List[str]:
        """获取所有摄像头ID列表。"""
        return list(self._handlers.keys())

    def __len__(self) -> int:
        return len(self._handlers)

    def __contains__(self, camera_id: str) -> bool:
        return camera_id in self._handlers

    def __repr__(self) -> str:
        return (
            f"StreamManager(cameras={self.camera_count}/"
            f"{self.max_cameras}, ids={self.camera_ids})"
        )
