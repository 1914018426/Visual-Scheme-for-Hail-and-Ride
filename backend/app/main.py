"""
Hailuo Car Backend - 智能招手打车检测系统主入口

提供多路视频流接入、实时姿态检测、手势识别和WebSocket视频推送功能。
基于FastAPI框架，支持CORS跨域访问。

启动方式:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os

# 未显式设置时使用 Hugging Face 国内镜像（部分库拉取资源时会读 HF_ENDPOINT）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_config
from app.stream.manager import StreamManager
from app.ai.detector import PoseDetector
from app.api import routes, ws

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 全局服务实例
_stream_manager: StreamManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    应用生命周期管理。

    启动时: 初始化视频流管理器、AI检测器，启动默认摄像头
    关闭时: 停止所有视频流，释放资源

    AI 检测器在后台线程加载，避免阻塞 HTTP（健康检查）与长时间权重下载
    导致连接不可用；WebSocket 推流在模型就绪后开始工作。
    """
    global _stream_manager

    logger.info("=" * 50)
    logger.info("Hailuo Car Backend 正在启动...")
    logger.info("=" * 50)

    config = get_config()

    # 初始化视频流管理器
    logger.info("正在初始化视频流管理器...")
    _stream_manager = StreamManager(max_cameras=config.camera.max_cameras)

    # 注入依赖到API模块（检测器稍后异步就绪）
    routes.set_stream_manager(_stream_manager)
    ws.set_stream_manager(_stream_manager)
    ws.set_detector(None)

    async def _load_detector_bg() -> None:
        logger.info(
            "正在后台加载 AI 检测器（首次下载/加载权重可能较慢，"
            "不影响 /api/health）..."
        )
        try:
            detector = await asyncio.to_thread(PoseDetector)
            ws.set_detector(detector)
            logger.info("AI检测器初始化完成")
        except Exception as e:
            logger.error("AI检测器初始化失败: %s", str(e))
            logger.warning("将继续运行，但AI检测与推流推理不可用")
            ws.set_detector(None)

    det_task = asyncio.create_task(_load_detector_bg())

    # 添加默认摄像头
    logger.info("正在配置默认摄像头...")
    default_cameras = config.camera.default_cameras
    for camera_id, source in default_cameras.items():
        if source:
            try:
                _stream_manager.add_stream(camera_id, source)
                logger.info("默认摄像头已添加: %s -> %s", camera_id, source)
            except Exception as e:
                logger.warning("添加默认摄像头失败 [%s]: %s", camera_id, str(e))

    # 启动所有视频流
    logger.info("正在启动所有视频流...")
    results = _stream_manager.start_all()
    for camera_id, success in results.items():
        status = "成功" if success else "失败"
        logger.info("摄像头 %s 启动%s", camera_id, status)

    logger.info("=" * 50)
    logger.info("Hailuo Car Backend 启动完成!")
    logger.info("API文档: http://%s:%d/docs", config.server.host, config.server.port)
    logger.info("=" * 50)

    yield

    # 应用关闭时清理
    logger.info("=" * 50)
    logger.info("Hailuo Car Backend 正在关闭...")
    logger.info("=" * 50)

    if det_task.done():
        exc = det_task.exception()
        if exc is not None:
            logger.debug("检测器后台任务异常: %s", exc)
    else:
        det_task.cancel()
        with suppress(asyncio.CancelledError):
            await det_task

    if _stream_manager is not None:
        logger.info("正在停止所有视频流...")
        _stream_manager.stop_all()

    logger.info("资源已清理，服务已停止")


def create_app() -> FastAPI:
    """
    创建FastAPI应用实例。

    Returns:
        FastAPI: 配置完成的应用实例
    """
    config = get_config()

    # 创建FastAPI实例
    app = FastAPI(
        title="Hailuo Car Backend",
        description="智能招手打车检测系统 - 多路视频流实时分析与方向决策",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # 配置CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册REST API路由
    app.include_router(routes.api_router)

    # 注册WebSocket路由
    app.include_router(ws.router)

    # 根路径重定向到API文档
    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        """根路径返回服务信息。"""
        return {
            "service": "Hailuo Car Backend",
            "version": "1.0.0",
            "docs": "/docs",
            "api": "/api",
            "websocket": "/ws/video",
        }

    return app


# 创建应用实例（uvicorn入口使用）
app = create_app()


# 直接运行入口
if __name__ == "__main__":
    import uvicorn

    config = get_config()
    uvicorn.run(
        "app.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.debug,
        log_level=config.server.log_level.lower(),
    )
