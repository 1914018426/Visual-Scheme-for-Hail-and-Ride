"""
方向决策模块

根据多路摄像头的检测结果，决策车辆行驶方向。
将摄像头位置映射到行驶方向，取最高置信度的非none检测结果。
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Direction(str, Enum):
    """行驶方向枚举。

    注意：值必须与前端的 Direction 类型保持一致
    （forward/backward/left/right/none）。
    """

    NONE = "none"
    FRONT = "forward"     # 前行
    BACK = "backward"     # 后退
    LEFT = "left"         # 左转
    RIGHT = "right"       # 右转
    STOP = "stop"         # 停止


# 摄像头位置到行驶方向的映射
# 键为摄像头ID，值为对应的方向
CAMERA_TO_DIRECTION: Dict[str, Direction] = {
    "front": Direction.FRONT,
    "back": Direction.BACK,
    "left": Direction.LEFT,
    "right": Direction.RIGHT,
}


@dataclass
class DirectionResult:
    """方向决策结果。"""

    direction: Direction = Direction.NONE  # 决策方向
    confidence: float = 0.0                # 置信度
    source_camera: str = ""                # 来源摄像头
    all_detections: List[Dict] = None      # 所有检测结果

    def __post_init__(self):
        if self.all_detections is None:
            self.all_detections = []

    def to_dict(self) -> Dict:
        """转换为字典格式（用于JSON序列化）。"""
        return {
            "direction": self.direction.value,
            "confidence": round(self.confidence, 4),
            "source_camera": self.source_camera,
            "all_detections": self.all_detections,
        }


class DirectionDecider:
    """
    方向决策器

    收集多路摄像头的检测结果，根据置信度做出最终方向决策。
    策略：取最高置信度的非none检测结果。
    """

    def __init__(self) -> None:
        """初始化方向决策器。"""
        self._camera_results: Dict[str, Tuple[str, float]] = {}

    def update_camera_result(
        self, camera_id: str, gesture_type: str, confidence: float
    ) -> None:
        """
        更新指定摄像头的检测结果。

        Args:
            camera_id: 摄像头唯一标识
            gesture_type: 手势类型 ('none', 'hand_up', 'wave')
            confidence: 置信度 (0-1)
        """
        self._camera_results[camera_id] = (gesture_type, confidence)

    def decide(self) -> DirectionResult:
        """
        根据所有摄像头的检测结果做出方向决策。

        策略:
        1. 过滤掉 'none' 类型的检测结果
        2. 在剩余结果中取置信度最高的
        3. 将该摄像头对应的方向作为最终决策

        Returns:
            DirectionResult: 方向决策结果
        """
        all_detections: List[Dict] = []
        best_camera = ""
        best_confidence = 0.0
        best_gesture = ""

        # 遍历所有摄像头结果
        for camera_id, (gesture_type, confidence) in self._camera_results.items():
            detection = {
                "camera": camera_id,
                "gesture": gesture_type,
                "confidence": round(confidence, 4),
                "direction": CAMERA_TO_DIRECTION.get(
                    camera_id, Direction.NONE
                ).value,
            }
            all_detections.append(detection)

            # 跳过无手势检测结果
            if gesture_type == "none" or confidence <= 0:
                continue

            # 取最高置信度的有效检测
            if confidence > best_confidence:
                best_confidence = confidence
                best_camera = camera_id
                best_gesture = gesture_type

        # 构建结果
        if best_camera and best_camera in CAMERA_TO_DIRECTION:
            direction = CAMERA_TO_DIRECTION[best_camera]
            return DirectionResult(
                direction=direction,
                confidence=best_confidence,
                source_camera=best_camera,
                all_detections=all_detections,
            )

        # 没有有效检测
        return DirectionResult(
            direction=Direction.NONE,
            confidence=0.0,
            source_camera="",
            all_detections=all_detections,
        )

    def reset(self) -> None:
        """重置所有摄像头结果。"""
        self._camera_results.clear()

    def get_camera_count(self) -> int:
        """获取已更新结果的摄像头数量。"""
        return len(self._camera_results)


# 全局方向决策器实例
_decider: Optional[DirectionDecider] = None


def get_decider() -> DirectionDecider:
    """
    获取全局方向决策器实例（单例模式）。

    Returns:
        DirectionDecider: 方向决策器实例
    """
    global _decider
    if _decider is None:
        _decider = DirectionDecider()
    return _decider


def decide_direction(
    camera_results: Dict[str, Tuple[str, float]]
) -> DirectionResult:
    """
    便捷函数：根据多摄像头结果做出方向决策。

    Args:
        camera_results: {camera_id: (gesture_type, confidence)}

    Returns:
        DirectionResult: 方向决策结果
    """
    decider = get_decider()
    decider.reset()
    for camera_id, (gesture, conf) in camera_results.items():
        decider.update_camera_result(camera_id, gesture, conf)
    return decider.decide()
