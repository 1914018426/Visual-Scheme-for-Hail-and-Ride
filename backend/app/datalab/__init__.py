"""
DataLab 数据实验室模块

提供招手素材录制、消融实验运行与统计分析功能。
"""

from app.datalab.models import (
    RecordingSession,
    AblationExperiment,
    EngineStats,
    AnalysisReport,
)
from app.datalab.persistence import DataLabStorage
from app.datalab.recorder import GestureRecorder
from app.datalab.ablation import AblationRunner
from app.datalab.analyzer import AblationAnalyzer

__all__ = [
    "RecordingSession",
    "AblationExperiment",
    "EngineStats",
    "AnalysisReport",
    "DataLabStorage",
    "GestureRecorder",
    "AblationRunner",
    "AblationAnalyzer",
]
