"""
AI推理引擎模块

提供人体姿态检测、手势识别和方向决策功能。
基于YOLOv8姿态检测和MediaPipe手部关键点检测。
"""

from app.ai.detector import PoseDetector
from app.ai.gesture import GestureRecognizer, is_hailing_gesture

__all__ = [
    "PoseDetector",
    "GestureRecognizer",
    "is_hailing_gesture",
]
