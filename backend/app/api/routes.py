"""
REST API路由模块

提供HTTP RESTful API端点，包括：
- 健康检查
- 摄像头管理（增删查）
- 系统统计信息
"""

import logging
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.stream.manager import StreamManager

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/api")
api_router = router

# 全局流管理器引用（在main.py中初始化时注入）
_stream_manager: Optional[StreamManager] = None

# 摄像头显示标签存储（独立于流管理器，用于前端自定义命名）
_camera_labels: Dict[str, str] = {}


def set_stream_manager(manager: StreamManager) -> None:
    """
    设置全局流管理器实例。

    Args:
        manager: StreamManager实例
    """
    global _stream_manager
    _stream_manager = manager
    logger.info("REST API 流管理器已注入")


def get_stream_manager() -> StreamManager:
    """
    获取流管理器实例。

    Returns:
        StreamManager实例

    Raises:
        HTTPException: 管理器未初始化时返回503错误
    """
    if _stream_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="流管理器未初始化",
        )
    return _stream_manager


# ========== Pydantic数据模型 ==========


class CameraConfig(BaseModel):
    """摄像头配置请求模型。"""

    camera_id: str = Field(
        ..., min_length=1, max_length=50, description="摄像头唯一标识"
    )
    source: str = Field(
        ..., min_length=1, max_length=500, description="视频源地址"
    )
    label: Optional[str] = Field(
        default=None, max_length=100, description="显示名称"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "camera_id": "front",
                "source": "rtsp://192.168.1.100:554/stream",
                "label": "前视摄像头",
            }
        }


class CameraResponse(BaseModel):
    """摄像头响应模型。"""

    camera_id: str
    source: str
    label: str = ""
    status: str
    fps: float = 0.0
    frame_count: int = 0
    width: int = 0
    height: int = 0
    queue_size: int = 0
    last_error: str = ""


class CameraListResponse(BaseModel):
    """摄像头列表响应模型。"""

    cameras: List[CameraResponse]
    total: int
    max_cameras: int


class StatsResponse(BaseModel):
    """统计信息响应模型。"""

    uptime_seconds: float
    camera_count: int
    max_cameras: int
    camera_statuses: List[Dict[str, Any]]
    system_info: Dict[str, Any]


class DirectionResponse(BaseModel):
    """方向决策响应模型。"""

    direction: str
    confidence: float
    source_camera: str
    all_detections: List[Dict[str, Any]]


class MessageResponse(BaseModel):
    """通用消息响应模型。"""

    message: str


# ========== API端点 ==========


@router.get("/health", response_model=Dict[str, Any], tags=["系统"])
async def health_check() -> Dict[str, Any]:
    """
    健康检查端点。

    返回服务运行状态和基本信息。
    """
    import time

    # 计算运行时间
    start_time = getattr(health_check, "_start_time", None)
    if start_time is None:
        health_check._start_time = time.time()
        uptime = 0.0
    else:
        uptime = time.time() - start_time

    manager = _stream_manager
    camera_count = manager.camera_count if manager else 0

    return {
        "status": "healthy",
        "service": "hailuo-car-backend",
        "version": "1.0.0",
        "uptime_seconds": round(uptime, 2),
        "camera_count": camera_count,
        "timestamp": time.time(),
    }


@router.get("/cameras", response_model=CameraListResponse, tags=["摄像头"])
async def list_cameras() -> CameraListResponse:
    """
    获取所有摄像头列表及状态。

    返回当前所有已添加摄像头的详细状态信息。
    """
    manager = get_stream_manager()
    statuses = manager.list_cameras()

    cameras = [
        CameraResponse(
            camera_id=s.camera_id,
            source=s.source,
            label=_camera_labels.get(s.camera_id, s.camera_id),
            status=s.status,
            fps=s.fps,
            frame_count=s.frame_count,
            width=s.width,
            height=s.height,
            queue_size=s.queue_size,
            last_error=s.last_error,
        )
        for s in statuses
    ]

    return CameraListResponse(
        cameras=cameras,
        total=len(cameras),
        max_cameras=manager.max_cameras,
    )


@router.post(
    "/cameras",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["摄像头"],
)
async def add_camera(config: CameraConfig) -> MessageResponse:
    """
    添加一路摄像头。

    Args:
        config: 摄像头配置（ID和视频源地址）

    Returns:
        操作结果消息

    Raises:
        HTTPException: 添加失败时返回相应错误
    """
    manager = get_stream_manager()

    # 检查是否已达到最大数量
    if manager.camera_count >= manager.max_cameras:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"已达到最大摄像头数量限制({manager.max_cameras})",
        )

    # 检查ID是否已存在
    if manager.has_camera(config.camera_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"摄像头 '{config.camera_id}' 已存在",
        )

    # 添加摄像头
    success = manager.add_stream(config.camera_id, config.source)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"添加摄像头 '{config.camera_id}' 失败",
        )

    # 启动视频流
    manager.start_stream(config.camera_id)

    # 保存显示标签
    if config.label is not None:
        _camera_labels[config.camera_id] = config.label

    logger.info("摄像头已添加: %s -> %s", config.camera_id, config.source)
    return MessageResponse(
        message=f"摄像头 '{config.camera_id}' 添加成功"
    )


@router.delete("/cameras/{camera_id}", response_model=MessageResponse, tags=["摄像头"])
async def remove_camera(camera_id: str) -> MessageResponse:
    """
    删除一路摄像头。

    Args:
        camera_id: 摄像头唯一标识

    Returns:
        操作结果消息

    Raises:
        HTTPException: 摄像头不存在时返回404
    """
    manager = get_stream_manager()

    if not manager.has_camera(camera_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"摄像头 '{camera_id}' 不存在",
        )

    manager.remove_stream(camera_id)

    # 清理显示标签
    _camera_labels.pop(camera_id, None)

    logger.info("摄像头已删除: %s", camera_id)
    return MessageResponse(message=f"摄像头 '{camera_id}' 删除成功")


@router.get(
    "/cameras/{camera_id}/status",
    response_model=CameraResponse,
    tags=["摄像头"],
)
async def get_camera_status(camera_id: str) -> CameraResponse:
    """
    获取指定摄像头的状态。

    Args:
        camera_id: 摄像头唯一标识

    Returns:
        摄像头状态信息

    Raises:
        HTTPException: 摄像头不存在时返回404
    """
    manager = get_stream_manager()

    status = manager.get_camera_status(camera_id)
    if status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"摄像头 '{camera_id}' 不存在",
        )

    return CameraResponse(
        camera_id=status.camera_id,
        source=status.source,
        status=status.status,
        fps=status.fps,
        frame_count=status.frame_count,
        width=status.width,
        height=status.height,
        queue_size=status.queue_size,
        last_error=status.last_error,
    )


@router.post(
    "/cameras/{camera_id}/start",
    response_model=MessageResponse,
    tags=["摄像头"],
)
async def start_camera(camera_id: str) -> MessageResponse:
    """
    启动指定摄像头的视频流。

    Args:
        camera_id: 摄像头唯一标识

    Returns:
        操作结果消息
    """
    manager = get_stream_manager()

    if not manager.has_camera(camera_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"摄像头 '{camera_id}' 不存在",
        )

    success = manager.start_stream(camera_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"启动摄像头 '{camera_id}' 失败",
        )

    return MessageResponse(message=f"摄像头 '{camera_id}' 已启动")


@router.post(
    "/cameras/{camera_id}/stop",
    response_model=MessageResponse,
    tags=["摄像头"],
)
async def stop_camera(camera_id: str) -> MessageResponse:
    """
    停止指定摄像头的视频流。

    Args:
        camera_id: 摄像头唯一标识

    Returns:
        操作结果消息
    """
    manager = get_stream_manager()

    if not manager.has_camera(camera_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"摄像头 '{camera_id}' 不存在",
        )

    manager.stop_stream(camera_id)
    return MessageResponse(message=f"摄像头 '{camera_id}' 已停止")


@router.get("/stats", response_model=Dict[str, Any], tags=["系统"])
async def get_stats() -> Dict[str, Any]:
    """
    获取系统统计信息。

    返回摄像头状态、系统资源使用等统计信息。
    """
    import time
    import platform
    import psutil

    manager = get_stream_manager()

    # 摄像头状态汇总
    camera_statuses = []
    for s in manager.list_cameras():
        camera_statuses.append(
            {
                "camera_id": s.camera_id,
                "source": s.source,
                "status": s.status,
                "fps": s.fps,
                "frame_count": s.frame_count,
                "resolution": f"{s.width}x{s.height}",
                "queue_size": s.queue_size,
            }
        )

    # 系统信息
    system_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_used_gb": round(
            psutil.virtual_memory().used / (1024 ** 3), 2
        ),
        "memory_total_gb": round(
            psutil.virtual_memory().total / (1024 ** 3), 2
        ),
    }

    return {
        "camera_count": manager.camera_count,
        "max_cameras": manager.max_cameras,
        "camera_statuses": camera_statuses,
        "system_info": system_info,
        "timestamp": time.time(),
    }
