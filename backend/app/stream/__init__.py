"""
视频流管理模块

提供多路视频流的捕获、缓冲和管理功能。
支持RTSP/RTMP/HTTP/本地文件等多种视频源。
"""

from app.stream.handler import StreamHandler
from app.stream.manager import StreamManager

__all__ = ["StreamHandler", "StreamManager"]
