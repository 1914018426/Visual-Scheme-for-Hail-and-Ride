"""
API模块

提供RESTful API和WebSocket通信接口。
包括摄像头管理、健康检查、实时视频流推送等功能。
"""

from app.api.routes import router as api_router
from app.api.ws import router as ws_router

__all__ = ["api_router", "ws_router"]
