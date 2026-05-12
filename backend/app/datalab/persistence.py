"""
DataLab 持久化层

基于文件系统的录制数据与实验结果存储。
目录结构：
    data/datalab/
        recordings/
            YYYY-MM-DD/
                {session_id}/
                    meta.json
                    keypoints.jsonl
                    tnlf_features.jsonl
                    detections.jsonl
                    video.mp4 (optional)
        experiments/
            {exp_id}/
                experiment.json
                frame_results.jsonl
                engine_stats.json
                report.json
"""

import os
import json
import csv
import logging
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Iterator

from app.datalab.models import (
    RecordingSession,
    AblationExperiment,
    AnalysisReport,
    MultiEngineFrameResult,
)
from app.config import get_config

logger = logging.getLogger(__name__)


class DataLabStorage:
    """DataLab 文件持久化管理器。"""

    def __init__(self, base_dir: Optional[str] = None) -> None:
        config = get_config()
        self.base_dir = Path(base_dir or "data/datalab")
        self.recordings_dir = self.base_dir / "recordings"
        self.experiments_dir = self.base_dir / "experiments"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """确保目录结构存在。"""
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def _recording_dir(self, session_id: str, start_time: float) -> Path:
        """获取录制会话目录。"""
        date_str = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d")
        return self.recordings_dir / date_str / session_id

    def _experiment_dir(self, exp_id: str) -> Path:
        """获取实验目录。"""
        return self.experiments_dir / exp_id

    # ------------------------------------------------------------------
    # RecordingSession CRUD
    # ------------------------------------------------------------------

    def create_recording(
        self,
        camera_id: str,
        trigger_mode: str,
    ) -> RecordingSession:
        """创建新的录制会话目录和元数据。"""
        session_id = str(uuid.uuid4())[:8]
        start_time = datetime.now().timestamp()
        rec_dir = self._recording_dir(session_id, start_time)
        rec_dir.mkdir(parents=True, exist_ok=True)

        session = RecordingSession(
            id=session_id,
            camera_id=camera_id,
            trigger_mode=trigger_mode,
            start_time=start_time,
            meta_path=str(rec_dir / "meta.json"),
            keypoints_path=str(rec_dir / "keypoints.jsonl"),
            tnlf_path=str(rec_dir / "tnlf_features.jsonl"),
            detections_path=str(rec_dir / "detections.jsonl"),
        )
        self._write_json(session.meta_path, session.model_dump())
        # 创建空 jsonl 文件
        for path in (session.keypoints_path, session.tnlf_path, session.detections_path):
            if path:
                Path(path).touch()
        return session

    def update_recording(self, session: RecordingSession) -> None:
        """更新录制会话元数据。"""
        self._write_json(session.meta_path, session.model_dump())

    def get_recording(self, session_id: str) -> Optional[RecordingSession]:
        """根据 ID 查找录制会话。"""
        # 扫描所有日期目录
        for date_dir in self.recordings_dir.iterdir():
            if not date_dir.is_dir():
                continue
            meta_path = date_dir / session_id / "meta.json"
            if meta_path.exists():
                data = self._read_json(str(meta_path))
                if data:
                    return RecordingSession(**data)
        return None

    def list_recordings(self) -> List[RecordingSession]:
        """列出所有录制会话（按时间倒序）。"""
        sessions: List[RecordingSession] = []
        for date_dir in sorted(self.recordings_dir.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue
            for session_dir in date_dir.iterdir():
                meta_path = session_dir / "meta.json"
                if meta_path.exists():
                    data = self._read_json(str(meta_path))
                    if data:
                        sessions.append(RecordingSession(**data))
        sessions.sort(key=lambda s: s.start_time, reverse=True)
        return sessions

    def delete_recording(self, session_id: str) -> bool:
        """删除录制会话及其所有文件。"""
        import shutil

        session = self.get_recording(session_id)
        if session is None:
            return False
        rec_dir = Path(session.meta_path).parent
        if rec_dir.exists():
            shutil.rmtree(rec_dir)
            logger.info("录制已删除: %s", session_id)
            return True
        return False

    def append_keypoints(self, session_id: str, frame_data: Dict[str, Any]) -> None:
        """追加关键点到 jsonl。"""
        session = self.get_recording(session_id)
        if session and session.keypoints_path:
            self._append_jsonl(session.keypoints_path, frame_data)

    def append_tnlf(self, session_id: str, frame_data: Dict[str, Any]) -> None:
        """追加 TNLF 特征到 jsonl。"""
        session = self.get_recording(session_id)
        if session and session.tnlf_path:
            self._append_jsonl(session.tnlf_path, frame_data)

    def append_detections(self, session_id: str, frame_data: Dict[str, Any]) -> None:
        """追加检测输出到 jsonl。"""
        session = self.get_recording(session_id)
        if session and session.detections_path:
            self._append_jsonl(session.detections_path, frame_data)

    def iter_keypoints(self, session_id: str) -> Iterator[Dict[str, Any]]:
        """迭代读取关键点序列。"""
        session = self.get_recording(session_id)
        if session and session.keypoints_path and Path(session.keypoints_path).exists():
            yield from self._iter_jsonl(session.keypoints_path)

    def iter_tnlf(self, session_id: str) -> Iterator[Dict[str, Any]]:
        """迭代读取 TNLF 特征序列。"""
        session = self.get_recording(session_id)
        if session and session.tnlf_path and Path(session.tnlf_path).exists():
            yield from self._iter_jsonl(session.tnlf_path)

    def iter_detections(self, session_id: str) -> Iterator[Dict[str, Any]]:
        """迭代读取检测输出序列。"""
        session = self.get_recording(session_id)
        if session and session.detections_path and Path(session.detections_path).exists():
            yield from self._iter_jsonl(session.detections_path)

    # ------------------------------------------------------------------
    # Video recording
    # ------------------------------------------------------------------

    def get_video_path(self, session_id: str) -> Optional[str]:
        """获取录制视频路径（若存在）。"""
        session = self.get_recording(session_id)
        if session and session.video_path and Path(session.video_path).exists():
            return session.video_path
        # 尝试推断路径
        for date_dir in self.recordings_dir.iterdir():
            if not date_dir.is_dir():
                continue
            vid = date_dir / session_id / "video.mp4"
            if vid.exists():
                return str(vid)
        return None

    # ------------------------------------------------------------------
    # AblationExperiment CRUD
    # ------------------------------------------------------------------

    def create_experiment(self, recording_id: str, engine_names: List[str]) -> AblationExperiment:
        """创建新的消融实验。"""
        exp_id = str(uuid.uuid4())[:8]
        exp_dir = self._experiment_dir(exp_id)
        exp_dir.mkdir(parents=True, exist_ok=True)

        exp = AblationExperiment(
            id=exp_id,
            recording_id=recording_id,
            engine_names=engine_names,
            created_at=datetime.now().timestamp(),
            frame_results_path=str(exp_dir / "frame_results.jsonl"),
            engine_stats_path=str(exp_dir / "engine_stats.json"),
            report_path=str(exp_dir / "report.json"),
        )
        self._write_json(str(exp_dir / "experiment.json"), exp.model_dump())
        # 初始化空 JSONL
        if exp.frame_results_path:
            Path(exp.frame_results_path).touch()
        return exp

    def update_experiment(self, exp: AblationExperiment) -> None:
        """更新实验状态。"""
        exp_dir = self._experiment_dir(exp.id)
        self._write_json(str(exp_dir / "experiment.json"), exp.model_dump())

    def get_experiment(self, exp_id: str) -> Optional[AblationExperiment]:
        """获取实验。"""
        exp_dir = self._experiment_dir(exp_id)
        meta_path = exp_dir / "experiment.json"
        if meta_path.exists():
            data = self._read_json(str(meta_path))
            if data:
                return AblationExperiment(**data)
        return None

    def list_experiments(self) -> List[AblationExperiment]:
        """列出所有实验（按时间倒序），过滤掉子实验。"""
        exps: List[AblationExperiment] = []
        for exp_dir in self.experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            meta_path = exp_dir / "experiment.json"
            if meta_path.exists():
                data = self._read_json(str(meta_path))
                if data:
                    exp = AblationExperiment(**data)
                    # 隐藏子实验（有 parent_id 的），只展示父实验
                    if not exp.parent_id:
                        exps.append(exp)
        exps.sort(key=lambda e: e.created_at, reverse=True)
        return exps

    def delete_experiment(self, exp_id: str) -> bool:
        """删除实验，级联删除子实验。"""
        import shutil

        exp = self.get_experiment(exp_id)
        if exp is None:
            return False

        # 级联删除子实验
        for child_id in exp.sub_experiment_ids:
            child_dir = self._experiment_dir(child_id)
            if child_dir.exists():
                shutil.rmtree(child_dir)
                logger.info("子实验已删除: %s", child_id)

        exp_dir = self._experiment_dir(exp_id)
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
            logger.info("实验已删除: %s", exp_id)
            return True
        return False

    def append_frame_result(self, exp_id: str, row: Dict[str, Any]) -> None:
        """追加逐帧结果到 JSONL。"""
        exp = self.get_experiment(exp_id)
        if exp and exp.frame_results_path:
            self._append_jsonl(exp.frame_results_path, row)

    def read_frame_results(self, exp_id: str) -> List[Dict[str, Any]]:
        """读取全部逐帧结果。"""
        exp = self.get_experiment(exp_id)
        if not exp or not exp.frame_results_path:
            return []
        path = Path(exp.frame_results_path)
        if not path.exists():
            return []
        return list(self._iter_jsonl(str(path)))

    def save_engine_stats(self, exp_id: str, stats: List[Dict[str, Any]]) -> None:
        """保存引擎统计。"""
        exp = self.get_experiment(exp_id)
        if exp and exp.engine_stats_path:
            self._write_json(exp.engine_stats_path, {"stats": stats})

    def save_report(self, exp_id: str, report: AnalysisReport) -> None:
        """保存分析报告。"""
        exp = self.get_experiment(exp_id)
        if exp and exp.report_path:
            self._write_json(exp.report_path, report.model_dump())

    def get_report(self, exp_id: str) -> Optional[AnalysisReport]:
        """读取分析报告。"""
        exp = self.get_experiment(exp_id)
        if not exp or not exp.report_path:
            return None
        data = self._read_json(exp.report_path)
        if data:
            return AnalysisReport(**data)
        return None

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_experiment_csv(self, exp_id: str, output_path: str) -> str:
        """导出实验结果 CSV 到指定路径。"""
        rows = self.read_frame_results(exp_id)
        if not rows:
            raise ValueError(f"实验 {exp_id} 没有数据")
        # 收集所有可能出现的列名，保证 CSV 表头完整
        fieldnames_set: set[str] = set()
        for r in rows:
            fieldnames_set.update(r.keys())
        # 固定基础列顺序，其余按字母序
        base_order = ["frame_idx", "timestamp", "threshold"]
        fieldnames = [k for k in base_order if k in fieldnames_set]
        fieldnames += sorted(k for k in fieldnames_set if k not in base_order)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return output_path

    def export_experiment_json(self, exp_id: str, output_path: str) -> str:
        """导出实验结果 JSON 到指定路径。"""
        exp = self.get_experiment(exp_id)
        report = self.get_report(exp_id)
        data = {
            "experiment": exp.model_dump() if exp else None,
            "frame_results": self.read_frame_results(exp_id),
            "report": report.model_dump() if report else None,
        }
        self._write_json(output_path, data)
        return output_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_json(path: str, data: Any) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def _read_json(path: str) -> Optional[Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None

    @staticmethod
    def _append_jsonl(path: str, data: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

