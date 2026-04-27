"""
实时日志 WebSocket 模块

通过 /ws/logs 端点将后端 Python 日志实时推送到前端。
使用 asyncio Queue 做生产者-消费者缓冲，避免日志风暴阻塞。
"""

import asyncio
import json
import logging
import time
from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter()

# 日志缓冲队列（最大 2000 条，避免内存爆炸）
_log_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=2000)
# 活跃的 WebSocket 连接集合
_active_log_ws: Set[WebSocket] = set()


class LogWebSocketHandler(logging.Handler):
    """
    自定义 logging.Handler，将日志记录推入 asyncio Queue。
    在 main.py 中会被添加到 root logger。
    """

    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            payload = {
                "timestamp": time.time(),
                "level": record.levelname,
                "logger": record.name,
                "message": msg,
            }
            # 使用 call_soon_threadsafe 在事件循环中安全地放入队列
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(
                    lambda: _put_nowait(payload)
                )
            except RuntimeError:
                # 无运行中的事件循环（如启动阶段），直接丢弃
                pass
        except Exception:
            self.handleError(record)


def _put_nowait(payload: dict) -> None:
    """非阻塞放入队列，队列满时丢弃最旧条目。"""
    if _log_queue.full():
        try:
            _log_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    try:
        _log_queue.put_nowait(payload)
    except asyncio.QueueFull:
        pass


async def _broadcast_log(payload: dict) -> None:
    """向所有活跃日志 WebSocket 连接广播单条日志。"""
    dead = set()
    for ws in _active_log_ws:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.add(ws)
    for ws in dead:
        _active_log_ws.discard(ws)


async def _log_broadcaster_task() -> None:
    """后台任务：从队列中消费日志并广播。"""
    while True:
        try:
            payload = await asyncio.wait_for(_log_queue.get(), timeout=1.0)
            if _active_log_ws:
                await _broadcast_log(payload)
        except asyncio.TimeoutError:
            continue
        except Exception:
            await asyncio.sleep(1)


@router.websocket("/ws/logs")
async def logs_websocket(websocket: WebSocket) -> None:
    """
    实时日志 WebSocket 端点。

    连接后立即开始接收后端日志消息，格式：
        {
            "timestamp": 1703000000.000,
            "level": "INFO",
            "logger": "app.ai.detector",
            "message": "..."
        }
    """
    await websocket.accept()
    _active_log_ws.add(websocket)
    logger.info("日志 WebSocket 客户端已连接")

    try:
        while True:
            # 保持连接活跃，客户端可发送 ping
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("action") == "ping":
                    await websocket.send_json({"action": "pong"})
            except Exception:
                pass
    except WebSocketDisconnect:
        logger.info("日志 WebSocket 客户端已断开")
    finally:
        _active_log_ws.discard(websocket)


# 启动广播后台任务（由 main.py lifespan 调用）
_log_broadcaster_started = False


def start_log_broadcaster() -> None:
    """在主事件循环中启动日志广播后台任务。"""
    global _log_broadcaster_started
    if _log_broadcaster_started:
        return
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_log_broadcaster_task())
        _log_broadcaster_started = True
        logger.info("日志广播后台任务已启动")
    except RuntimeError:
        pass
