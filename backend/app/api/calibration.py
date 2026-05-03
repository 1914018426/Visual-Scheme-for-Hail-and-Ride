"""
标定数据收集与特征分析模块

提供标定样本的 REST API 接收、自动特征分析、日志记录功能，
为后续模型微调提供参考数据与统计依据。
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/calibration")

# ========== 内存存储（生产环境建议替换为数据库） ==========

_calibration_store: Dict[str, "StoredSample"] = {}
_store_lock = False  # 简单 flag；生产环境改用 asyncio.Lock


# ========== Pydantic 请求模型 ==========


class CalibrationFrame(BaseModel):
    """单帧标定数据，与前端 KeypointFrame 结构一致。"""

    timestamp: float
    camera_id: str
    track_id: str
    bbox: Tuple[float, float, float, float]
    gesture: str
    gesture_conf: float
    keypoints: Optional[List[List[float]]] = None


class CalibrationSample(BaseModel):
    """单个标定样本。"""

    id: str
    label: str  # "waving" | "not_waving"
    start_time: float
    end_time: float
    frames: List[CalibrationFrame]


class CalibrationUpload(BaseModel):
    """标定数据上传请求体。"""

    version: str = "1.0.0"
    exported_at: str = ""
    total_samples: int = 0
    samples: List[CalibrationSample]


# ========== 存储模型 ==========


@dataclass
class StoredSample:
    """入库的标定样本（带入库时间戳）。"""

    id: str
    label: str
    start_time: float
    end_time: float
    frames: List[Dict[str, Any]]
    received_at: float = 0.0
    analysis: Optional[Dict[str, Any]] = None


# ========== 特征分析引擎 ==========


@dataclass
class KeypointStats:
    """单个关键点的统计信息。"""

    name: str
    x_mean: float = 0.0
    x_std: float = 0.0
    y_mean: float = 0.0
    y_std: float = 0.0
    conf_mean: float = 0.0
    conf_std: float = 0.0
    visibility_rate: float = 0.0  # conf > 0.3 的比例


KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# 骨架连接的语义分组（用于空间特征分析）
UPPER_BODY_INDICES = list(range(5, 11))   # shoulder → wrist
LOWER_BODY_INDICES = list(range(11, 17))  # hip → ankle
FACE_INDICES = list(range(0, 5))           # nose → ear


def _normalize_keypoints(
    kpts: np.ndarray, bbox: Tuple[float, float, float, float]
) -> Optional[np.ndarray]:
    """将关键点坐标按 bbox 归一化到 [0, 1] 区间。"""
    x1, y1, x2, y2 = bbox
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    if kpts.shape[0] < 17:
        return None
    normalized = kpts.copy().astype(np.float64)
    normalized[:, 0] = (normalized[:, 0] - x1) / bw
    normalized[:, 1] = (normalized[:, 1] - y1) / bh
    return normalized


def _compute_arm_angles(kpts: np.ndarray) -> Dict[str, float]:
    """计算手臂关键角度（用于分析招手动作的典型姿态）。"""
    result: Dict[str, float] = {}
    if kpts.shape[0] < 13:
        return result

    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """三点夹角（度）。"""
        v1 = a - b
        v2 = c - b
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm < 1e-6:
            return 0.0
        cos = np.clip(np.dot(v1, v2) / norm, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos)))

    # 左臂角 (shoulder-elbow-wrist)
    if all(kpts[i, 2] > 0.3 for i in [5, 7, 9]):
        result["left_arm_angle"] = _angle(kpts[5], kpts[7], kpts[9])
    # 右臂角
    if all(kpts[i, 2] > 0.3 for i in [6, 8, 10]):
        result["right_arm_angle"] = _angle(kpts[6], kpts[8], kpts[10])
    # 左肩抬升角 (shoulder → wrist 与水平夹角)
    if kpts[5, 2] > 0.3 and kpts[9, 2] > 0.3:
        dx = kpts[9, 0] - kpts[5, 0]
        dy = kpts[5, 1] - kpts[9, 1]  # y 向下为正，取反得抬升
        result["left_arm_elevation"] = float(np.degrees(np.arctan2(max(dy, 0), abs(dx) + 1e-6)))
    if kpts[6, 2] > 0.3 and kpts[10, 2] > 0.3:
        dx = kpts[10, 0] - kpts[6, 0]
        dy = kpts[6, 1] - kpts[10, 1]
        result["right_arm_elevation"] = float(np.degrees(np.arctan2(max(dy, 0), abs(dx) + 1e-6)))

    return result


def _compute_hand_wrist_distance(
    kpts: np.ndarray,
) -> Optional[float]:
    """计算两手手腕间距（归一化后），招手时通常较大。"""
    if kpts.shape[0] < 11:
        return None
    if kpts[9, 2] < 0.3 or kpts[10, 2] < 0.3:
        return None
    dx = kpts[9, 0] - kpts[10, 0]
    dy = kpts[9, 1] - kpts[10, 1]
    return float(np.sqrt(dx**2 + dy**2))


def _compute_wrist_shoulder_ratio(
    kpts: np.ndarray,
) -> Optional[float]:
    """计算手腕相对肩膀高度比 — 招手时手腕常在肩膀上方。"""
    if kpts.shape[0] < 11:
        return None
    left_ok = kpts[5, 2] > 0.3 and kpts[9, 2] > 0.3
    right_ok = kpts[6, 2] > 0.3 and kpts[10, 2] > 0.3
    if not left_ok and not right_ok:
        return None
    ratios = []
    if left_ok:
        ratios.append((kpts[5, 1] - kpts[9, 1]) / max(kpts[5, 1], 1.0))
    if right_ok:
        ratios.append((kpts[6, 1] - kpts[10, 1]) / max(kpts[6, 1], 1.0))
    return float(np.mean(ratios))


def analyze_samples(samples: List[StoredSample]) -> Dict[str, Any]:
    """
    对标定样本执行全量特征分析。

    分析维度：
    1. 样本基础统计（数量、分布、时长）
    2. 关键点空间分布（各部位可见率、位置均值/方差）
    3. 姿态特征（手臂角度、抬升角、手腕间距）
    4. 时序特征（帧数分布、时长分布）
    5. 置信度分布
    6. 摄像头分布
    """
    if not samples:
        return {"status": "empty", "message": "无标定样本可分析"}

    total = len(samples)
    waving = [s for s in samples if s.label == "waving"]
    not_waving = [s for s in samples if s.label == "not_waving"]

    # ---- 1. 基础统计 ----
    durations = [s.end_time - s.start_time for s in samples]
    frame_counts = [len(s.frames) for s in samples]
    labels = [s.label for s in samples]

    # ---- 2. 全帧关键点收集 ----
    all_kpts_normalized: Dict[str, List[np.ndarray]] = {"waving": [], "not_waving": []}
    all_arm_angles: Dict[str, List[Dict]] = {"waving": [], "not_waving": []}
    all_wrist_dist: Dict[str, List[float]] = {"waving": [], "not_waving": []}
    all_wrist_shoulder: Dict[str, List[float]] = {"waving": [], "not_waving": []}
    all_confs: Dict[str, List[float]] = {"waving": [], "not_waving": []}

    for s in samples:
        label = s.label
        for f in s.frames:
            kpts_raw = f.get("keypoints")
            bbox = f.get("bbox")
            if not kpts_raw or not bbox or len(kpts_raw) < 17:
                continue

            kpts_arr = np.array(kpts_raw, dtype=np.float64)
            if kpts_arr.shape[1] < 3:
                continue

            # 收集归一化关键点
            normed = _normalize_keypoints(kpts_arr, tuple(bbox))
            if normed is not None:
                all_kpts_normalized[label].append(normed)

            # 收集角度特征
            angles = _compute_arm_angles(kpts_arr)
            if angles:
                all_arm_angles[label].append(angles)

            # 手腕间距
            wd = _compute_hand_wrist_distance(kpts_arr)
            if wd is not None:
                all_wrist_dist[label].append(wd)

            # 手腕-肩膀高度比
            ws = _compute_wrist_shoulder_ratio(kpts_arr)
            if ws is not None:
                all_wrist_shoulder[label].append(ws)

            # 手势置信度
            gesture_conf = f.get("gesture_conf", 0)
            all_confs[label].append(gesture_conf)

    # ---- 3. 逐关键点统计 ----
    kpt_stats: Dict[str, Dict[str, KeypointStats]] = {}
    for label_name in ("waving", "not_waving"):
        kpts_list = all_kpts_normalized.get(label_name, [])
        if not kpts_list:
            kpt_stats[label_name] = {}
            continue

        stacked = np.stack(kpts_list, axis=0)  # [N, 17, 3]
        per_kpt: Dict[str, KeypointStats] = {}
        for i, name in enumerate(KEYPOINT_NAMES):
            col = stacked[:, i, :]
            visible = np.mean(col[:, 2] > 0.3) * 100
            per_kpt[name] = KeypointStats(
                name=name,
                x_mean=float(np.mean(col[:, 0])),
                x_std=float(np.std(col[:, 0])),
                y_mean=float(np.mean(col[:, 1])),
                y_std=float(np.std(col[:, 1])),
                conf_mean=float(np.mean(col[:, 2])),
                conf_std=float(np.std(col[:, 2])),
                visibility_rate=round(visible, 1),
            )
        kpt_stats[label_name] = per_kpt

    # ---- 4. 角度特征汇总 ----
    def _avg_angles(angle_list: List[Dict]) -> Dict[str, float]:
        if not angle_list:
            return {}
        keys = angle_list[0].keys()
        result = {}
        for k in keys:
            vals = [d[k] for d in angle_list if k in d]
            if vals:
                result[f"{k}_mean"] = round(float(np.mean(vals)), 1)
                result[f"{k}_std"] = round(float(np.std(vals)), 1)
        return result

    # ---- 5. 摄像头统计 ----
    camera_counts: Dict[str, int] = {}
    for s in samples:
        for f in s.frames:
            cam = f.get("camera_id", "unknown")
            camera_counts[cam] = camera_counts.get(cam, 0) + 1

    # ---- 6. 组装分析结果 ----
    analysis: Dict[str, Any] = {
        "status": "ok",
        "total_samples": total,
        "waving_count": len(waving),
        "not_waving_count": len(not_waving),
        "waving_ratio": round(len(waving) / total, 3) if total > 0 else 0,
        "total_frames": sum(frame_counts),
        "duration_stats": {
            "total_seconds": round(sum(durations), 2),
            "mean_seconds": round(float(np.mean(durations)), 2) if durations else 0,
            "std_seconds": round(float(np.std(durations)), 2) if durations else 0,
            "min_seconds": round(float(min(durations)), 2) if durations else 0,
            "max_seconds": round(float(max(durations)), 2) if durations else 0,
        },
        "frame_count_stats": {
            "mean": round(float(np.mean(frame_counts)), 1) if frame_counts else 0,
            "std": round(float(np.std(frame_counts)), 1) if frame_counts else 0,
            "min": int(min(frame_counts)) if frame_counts else 0,
            "max": int(max(frame_counts)) if frame_counts else 0,
        },
        "gesture_confidence": {
            "waving": _describe_conf_dist(all_confs.get("waving", [])),
            "not_waving": _describe_conf_dist(all_confs.get("not_waving", [])),
        },
        "keypoint_visibility": {
            label: {
                k: v.visibility_rate for k, v in stats.items()
            }
            for label, stats in kpt_stats.items()
        },
        "arm_angles": {
            label: _avg_angles(all_arm_angles.get(label, []))
            for label in ("waving", "not_waving")
        },
        "hand_wrist_distance": {
            label: _describe_dist(all_wrist_dist.get(label, []))
            for label in ("waving", "not_waving")
        },
        "wrist_shoulder_height_ratio": {
            label: _describe_dist(all_wrist_shoulder.get(label, []))
            for label in ("waving", "not_waving")
        },
        "camera_frame_distribution": dict(
            sorted(camera_counts.items(), key=lambda x: -x[1])
        ),
        "unique_cameras": len(camera_counts),
        "unique_track_ids": len(set(
            f.get("track_id", "") for s in samples for f in s.frames
        )),
    }

    return analysis


def _describe_conf_dist(values: List[float]) -> Dict[str, float]:
    """描述置信度分布。"""
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0, "count": 0}
    return {
        "mean": round(float(np.mean(values)), 3),
        "std": round(float(np.std(values)), 3),
        "min": round(float(min(values)), 3),
        "max": round(float(max(values)), 3),
        "median": round(float(np.median(values)), 3),
        "count": len(values),
    }


def _describe_dist(values: List[float]) -> Dict[str, float]:
    """描述数值分布。"""
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "count": 0}
    return {
        "mean": round(float(np.mean(values)), 3),
        "std": round(float(np.std(values)), 3),
        "min": round(float(min(values)), 3),
        "max": round(float(max(values)), 3),
        "count": len(values),
    }


def _keypoint_stats_to_dict(stats: Dict[str, KeypointStats]) -> Dict[str, Dict[str, float]]:
    """KeypointStats 转可序列化字典。"""
    return {
        name: {
            "x_mean": round(s.x_mean, 3),
            "x_std": round(s.x_std, 3),
            "y_mean": round(s.y_mean, 3),
            "y_std": round(s.y_std, 3),
            "conf_mean": round(s.conf_mean, 3),
            "conf_std": round(s.conf_std, 3),
            "visibility_rate": s.visibility_rate,
        }
        for name, s in stats.items()
    }


# ========== 日志格式化 ==========


def log_analysis(analysis: Dict[str, Any]) -> None:
    """将分析结果以结构化格式写入日志，便于微调时查阅。"""
    if analysis.get("status") != "ok":
        logger.info("[标定分析] 无有效样本，跳过分析")
        return

    n = analysis["total_samples"]
    nw = analysis["waving_count"]
    nn = analysis["not_waving_count"]
    logger.info(
        "╔══════════════════════════════════════════════╗\n"
        "║         标定数据特征分析报告                 ║\n"
        "╚══════════════════════════════════════════════╝\n"
        "【样本概况】总量=%d | 招手=%d | 非招手=%d | 招手占比=%.1f%%\n"
        "【帧数统计】总帧数=%d | 均值=%.1f | 标准差=%.1f | 范围=[%d, %d]\n"
        "【时长统计】总计=%.1fs | 均值=%.2fs | 范围=[%.2fs, %.2fs]\n"
        "【摄像头数】%d 路 | 去重track_id数=%d",
        n, nw, nn, analysis["waving_ratio"] * 100,
        analysis["total_frames"],
        analysis["frame_count_stats"]["mean"],
        analysis["frame_count_stats"]["std"],
        analysis["frame_count_stats"]["min"],
        analysis["frame_count_stats"]["max"],
        analysis["duration_stats"]["total_seconds"],
        analysis["duration_stats"]["mean_seconds"],
        analysis["duration_stats"]["min_seconds"],
        analysis["duration_stats"]["max_seconds"],
        analysis["unique_cameras"],
        analysis["unique_track_ids"],
    )

    # 置信度分布
    for label_name in ("waving", "not_waving"):
        cd = analysis["gesture_confidence"].get(label_name, {})
        logger.info(
            "【%s 置信度】mean=%.3f | median=%.3f | std=%.3f | range=[%.3f, %.3f] | 样本帧数=%d",
            label_name,
            cd.get("mean", 0), cd.get("median", 0),
            cd.get("std", 0), cd.get("min", 0), cd.get("max", 0),
            cd.get("count", 0),
        )

    # 手臂角度
    for label_name in ("waving", "not_waving"):
        angles = analysis["arm_angles"].get(label_name, {})
        if angles:
            logger.info(
                "【%s 手臂角度】%s",
                label_name,
                " | ".join(f"{k}={v}" for k, v in angles.items()),
            )

    # 手腕高度比
    for label_name in ("waving", "not_waving"):
        ws = analysis["wrist_shoulder_height_ratio"].get(label_name, {})
        if ws and ws.get("count", 0) > 0:
            logger.info(
                "【%s 手腕/肩膀高度比】mean=%.3f | std=%.3f | range=[%.3f, %.3f]",
                label_name,
                ws.get("mean", 0), ws.get("std", 0),
                ws.get("min", 0), ws.get("max", 0),
            )

    # 高可见率关键点 Top-5（对于招手动作区分最有价值的部位）
    for label_name in ("waving", "not_waving"):
        vis = analysis["keypoint_visibility"].get(label_name, {})
        top5 = sorted(vis.items(), key=lambda x: -x[1])[:5]
        logger.info(
            "【%s 高可见关键点 Top5】%s",
            label_name,
            " | ".join(f"{n}={v}%" for n, v in top5),
        )

    # 摄像头分布
    cam_dist = analysis.get("camera_frame_distribution", {})
    if cam_dist:
        total_cam_frames = sum(cam_dist.values())
        cam_summary = " | ".join(
            f"{cam}={count}({count/total_cam_frames*100:.0f}%)"
            for cam, count in cam_dist.items()
        )
        logger.info("【摄像头帧分布】%s", cam_summary)

    logger.info(
        "╔══════════════════════════════════════════════╗\n"
        "║       标定分析完成 — 数据可用于模型微调      ║\n"
        "╚══════════════════════════════════════════════╝"
    )


# ========== API 端点 ==========


@router.post("/samples", summary="上传标定样本")
async def upload_calibration(upload: CalibrationUpload) -> Dict[str, Any]:
    """
    接收前端标定样本，执行自动特征分析并入库。

    - 验证数据完整性
    - 转为 StoredSample 存储
    - 执行特征分析并记录详细日志
    - 返回存储状态与分析摘要
    """
    samples = upload.samples
    if not samples:
        raise HTTPException(status_code=400, detail="标定样本列表为空")

    n_expected = upload.total_samples
    n_actual = len(samples)
    if n_expected > 0 and n_actual != n_expected:
        logger.warning(
            "标定上传样本数不匹配: 声明=%d, 实际=%d", n_expected, n_actual
        )

    # 入库
    now = time.time()
    stored: List[StoredSample] = []
    for s in samples:
        stored_sample = StoredSample(
            id=s.id or str(uuid.uuid4()),
            label=s.label,
            start_time=s.start_time,
            end_time=s.end_time,
            frames=[f.model_dump() for f in s.frames],
            received_at=now,
        )
        _calibration_store[stored_sample.id] = stored_sample
        stored.append(stored_sample)

    # 执行特征分析
    analysis = analyze_samples(stored)
    log_analysis(analysis)

    # 将分析结果挂回存储
    for s in stored:
        s.analysis = analysis

    summary = {
        "status": "ok",
        "stored_count": len(stored),
        "analysis_summary": {
            "total_samples": analysis["total_samples"],
            "waving_count": analysis["waving_count"],
            "not_waving_count": analysis["not_waving_count"],
            "total_frames": analysis["total_frames"],
            "unique_cameras": analysis["unique_cameras"],
            "total_duration_seconds": analysis["duration_stats"]["total_seconds"],
            "gesture_confidence_waving_mean": analysis["gesture_confidence"]["waving"].get("mean", 0),
            "gesture_confidence_not_waving_mean": analysis["gesture_confidence"]["not_waving"].get("mean", 0),
        },
    }

    logger.info(
        "标定数据上传成功: %d 个样本 (%d 招手 / %d 非招手), "
        "来自 %d 路摄像头, 共 %d 帧",
        summary["analysis_summary"]["total_samples"],
        summary["analysis_summary"]["waving_count"],
        summary["analysis_summary"]["not_waving_count"],
        summary["analysis_summary"]["unique_cameras"],
        summary["analysis_summary"]["total_frames"],
    )

    return summary


@router.get("/samples", summary="获取已存储的标定样本列表")
async def list_samples() -> Dict[str, Any]:
    """返回所有已存储的标定样本摘要列表。"""
    items = []
    for sid, s in _calibration_store.items():
        items.append({
            "id": sid,
            "label": s.label,
            "frame_count": len(s.frames),
            "start_time": s.start_time,
            "end_time": s.end_time,
            "duration": round(s.end_time - s.start_time, 2),
            "received_at": s.received_at,
        })

    return {
        "total": len(items),
        "samples": items,
    }


@router.get("/analysis", summary="获取最新标定数据分析报告")
async def get_analysis() -> Dict[str, Any]:
    """返回对所有已存储标定样本的完整分析报告。"""
    if not _calibration_store:
        return {"status": "empty", "message": "无标定数据，请先上传样本"}

    samples = list(_calibration_store.values())
    analysis = analyze_samples(samples)
    return analysis


@router.delete("/samples", summary="清空所有标定数据")
async def clear_samples() -> Dict[str, Any]:
    """清空内存中的标定数据。"""
    count = len(_calibration_store)
    _calibration_store.clear()
    logger.info("标定数据已清空: 共 %d 个样本", count)
    return {"status": "ok", "cleared_count": count}
