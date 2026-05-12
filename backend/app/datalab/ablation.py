"""
消融实验运行器 — 全面增强版

支持 4 种实验类型：
  1. engine_comparison   — 5 主引擎横向对比
  2. component_ablation  — STH 组件逐一消融
  3. threshold_sweep     — 阈值扫描生成 PR/ROC 曲线
  4. scenario_analysis   — 按速度/距离/左右手分场景统计

组件消融变体通过子类化 + 方法覆盖实现，不修改 gesture.py 原引擎代码。
"""

import asyncio
import logging
import math
import time
from collections import deque
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from app.datalab.models import (
    AblationExperiment,
    ExperimentStatus,
    ExperimentType,
    MultiEngineFrameResult,
    EngineFrameResult,
)
from app.datalab.persistence import DataLabStorage
from app.config import get_config

logger = logging.getLogger(__name__)


# =============================================================================
# STH 组件消融包装器（子类化，不修改原引擎）
# =============================================================================

class _STHNoTransformBypass:
    """STH 去掉 Transformer 不同意时的降级 fallback（0.9× Simple conf）。

    当前 STH 为 Simple-primary：Simple 确认 waving 后，若 Transformer
    也确认则取 max(conf)；若 Transformer 不同意且 Simple conf > 0.45，
    则降级输出 0.9× Simple conf。本包装器移除该降级分支，强制要求
    Transformer 与 Simple 同时确认才输出 waving。
    """

    def __init__(self, base) -> None:
        self._base = base

    def recognize(self, *args, **kwargs):
        from app.ai.gesture import GestureType, GestureResult

        if self._base._has_transformer:
            tf_result = self._base.transformer.recognize(*args, **kwargs)
        else:
            tf_result = None
        s_result = self._base.simple.recognize(*args, **kwargs)

        # Simple-primary：Simple 必须先确认
        if s_result.gesture_type != GestureType.WAVING:
            return GestureResult(gesture_type=GestureType.NONE, confidence=0.0)

        if tf_result and tf_result.gesture_type == GestureType.WAVING:
            return GestureResult(
                gesture_type=GestureType.WAVING,
                confidence=max(s_result.confidence, tf_result.confidence),
            )

        # 去掉 fallback：Transformer 不同意则直接拒绝
        return GestureResult(gesture_type=GestureType.NONE, confidence=0.0)

    def reset(self):
        self._base.reset()

    @property
    def simple(self):
        return self._base.simple

    @property
    def transformer(self):
        return self._base.transformer


class _STHStrictAnd:
    """STH 严格 AND：Simple 与 Transformer 必须同时确认，无任何 fallback。"""

    def __init__(self, base) -> None:
        self._base = base

    def recognize(self, *args, **kwargs):
        from app.ai.gesture import GestureType, GestureResult

        if self._base._has_transformer:
            tf_result = self._base.transformer.recognize(*args, **kwargs)
        else:
            tf_result = None
        s_result = self._base.simple.recognize(*args, **kwargs)

        if s_result.gesture_type != GestureType.WAVING:
            return GestureResult(gesture_type=GestureType.NONE, confidence=0.0)

        if tf_result and tf_result.gesture_type == GestureType.WAVING:
            return GestureResult(
                gesture_type=GestureType.WAVING,
                confidence=max(s_result.confidence, tf_result.confidence),
            )

        return GestureResult(gesture_type=GestureType.NONE, confidence=0.0)

    def reset(self):
        self._base.reset()


class _STHTransformerOnly:
    """STH 仅保留 Transformer，无 Simple 后验过滤。"""

    def __init__(self, base) -> None:
        self._base = base

    def recognize(self, *args, **kwargs):
        # 直接返回 Transformer 结果
        return self._base.transformer.recognize(*args, **kwargs)

    def reset(self):
        self._base.reset()


# =============================================================================
# Simple / TripleLock 组件消融包装器
# =============================================================================

class _SimpleNoPeriodicity:
    """Simple 去掉周期性检测（仅保留姿态门）。"""

    def __init__(self, base) -> None:
        self._base = base
        from app.ai.gesture import SimpleGestureEngine

        class EngineNoPeriodicity(SimpleGestureEngine):
            def _detect_periodic(self, k: str) -> Tuple[bool, float]:
                # 强制通过周期性检测，返回中等置信度
                return True, 0.55

        self._base.engine = EngineNoPeriodicity(
            nose_conf_threshold=base.engine.nose_conf_threshold,
            eye_conf_threshold=base.engine.eye_conf_threshold,
            period_window_seconds=base.engine.period_window,
            fps=base.engine.fps,
            min_freq_hz=base.engine.min_freq_hz,
            max_freq_hz=base.engine.max_freq_hz,
            min_cycles=base.engine.min_cycles,
            min_confirm_frames=base.engine.min_confirm_frames,
            hold_frames=base.engine.hold_frames,
            ema_alpha=base.engine.ema_alpha,
        )

    def recognize(self, *args, **kwargs):
        return self._base.recognize(*args, **kwargs)

    def reset(self):
        self._base.reset()


class _SimpleNoPoseGate:
    """Simple 去掉姿态门（鼻子可见 + 手腕高于手肘），仅保留周期性。"""

    def __init__(self, base) -> None:
        self._base = base
        from app.ai.gesture import SimpleGestureEngine

        class EngineNoPoseGate(SimpleGestureEngine):
            def process_frame(self, keypoints, side, track_id, wrist_local, timestamp):
                # 跳过鼻子和手腕高度检查，直接做周期性（使用 wrist-elbow 相对向量）
                k = self._key(track_id, side)
                maxlen = int(self.period_window * self.fps)

                if side == "left":
                    elbow_idx, wrist_idx = 7, 9
                else:
                    elbow_idx, wrist_idx = 8, 10

                if float(keypoints[elbow_idx, 2]) < 0.3 or float(keypoints[wrist_idx, 2]) < 0.3:
                    self._reset(k)
                    return "none", 0.0

                # 使用 wrist-elbow 相对向量（与当前 SimpleGestureEngine 一致）
                shoulder_l = keypoints[5][:2]
                shoulder_r = keypoints[6][:2]
                shoulder_width = float(np.linalg.norm(shoulder_r - shoulder_l))
                if shoulder_width < 1.0:
                    self._reset(k)
                    return "none", 0.0

                wrist_elbow_vec = (keypoints[wrist_idx][:2] - keypoints[elbow_idx][:2]) / shoulder_width
                rel_vec = (float(wrist_elbow_vec[0]), float(wrist_elbow_vec[1]))

                # 5帧滑动平均去抖（与当前 SimpleGestureEngine 一致）
                if k not in self._smooth_win:
                    self._smooth_win[k] = deque(maxlen=5)
                self._smooth_win[k].append(rel_vec)
                smoothed = (
                    sum(p[0] for p in self._smooth_win[k]) / len(self._smooth_win[k]),
                    sum(p[1] for p in self._smooth_win[k]) / len(self._smooth_win[k]),
                )

                if k not in self._wrist_history:
                    self._wrist_history[k] = deque(maxlen=maxlen)
                self._wrist_history[k].append(smoothed)

                is_periodic, raw_conf = self._detect_periodic(k)

                if is_periodic:
                    self._confirm_count[k] = self._confirm_count.get(k, 0) + 1
                else:
                    self._confirm_count[k] = max(0, self._confirm_count.get(k, 0) - 1)

                if k not in self._ema_conf:
                    self._ema_conf[k] = raw_conf
                else:
                    self._ema_conf[k] = (
                        self.ema_alpha * raw_conf
                        + (1.0 - self.ema_alpha) * self._ema_conf[k]
                    )
                smoothed_conf = self._ema_conf[k]

                if self._confirm_count.get(k, 0) >= self.min_confirm_frames:
                    self._hold_count[k] = self.hold_frames
                    gesture, conf = "waving", smoothed_conf
                    self._last_result[k] = (gesture, conf)
                    return gesture, conf

                if self._hold_count.get(k, 0) > 0:
                    self._hold_count[k] -= 1
                    if k in self._last_result:
                        decay = self._hold_count[k] / max(self.hold_frames, 1)
                        return self._last_result[k][0], self._last_result[k][1] * decay

                return "none", smoothed_conf

        self._base.engine = EngineNoPoseGate(
            nose_conf_threshold=base.engine.nose_conf_threshold,
            eye_conf_threshold=base.engine.eye_conf_threshold,
            period_window_seconds=base.engine.period_window,
            fps=base.engine.fps,
            min_freq_hz=base.engine.min_freq_hz,
            max_freq_hz=base.engine.max_freq_hz,
            min_cycles=base.engine.min_cycles,
            min_confirm_frames=base.engine.min_confirm_frames,
            hold_frames=base.engine.hold_frames,
            ema_alpha=base.engine.ema_alpha,
            min_amplitude_tu=base.engine.min_amplitude_tu,
        )

    def recognize(self, *args, **kwargs):
        return self._base.recognize(*args, **kwargs)

    def reset(self):
        self._base.reset()


class _TripleLockNoOrientation:
    """TripleLock 去掉朝向锁。"""

    def __init__(self, base) -> None:
        self._base = base

    def recognize(self, *args, **kwargs):
        from app.ai.gesture import GestureType, GestureResult

        keypoints = kwargs.get("keypoints") if kwargs.get("keypoints") is not None else args[0] if args else None
        track_id = kwargs.get("track_id", "default")
        left_pn = kwargs.get("left_palm_normal")
        right_pn = kwargs.get("right_palm_normal")
        timestamp = kwargs.get("frame_timestamp")
        active_ids = kwargs.get("active_track_ids")

        if active_ids is not None:
            self._base.engine.gc_states(active_ids)

        now = timestamp if timestamp is not None else time.time()

        c = self._base.config.ai
        from app.ai.facing import facing_gate

        f_human, is_hard_rejected, f_human_multiplier = facing_gate(
            keypoints,
            hard_threshold=c.gesture_facing_hard_threshold,
            soft_threshold=c.gesture_facing_soft_threshold,
        )
        if is_hard_rejected:
            return GestureResult(gesture_type=GestureType.NONE, confidence=0.0)

        best_result = None
        for side in ["right", "left"]:
            raw_normal = left_pn if side == "left" else right_pn
            palm_normal = None
            if raw_normal is not None:
                state = self._base.engine._get_state(track_id, side)
                palm_normal = state.normal_smoother.update(raw_normal)

            # Monkey-patch orientation lock angle to always pass
            original_angle = self._base.engine.orientation_lock_angle
            self._base.engine.orientation_lock_angle = 180.0
            try:
                gesture, confidence = self._base.engine.process_frame(
                    keypoints=keypoints,
                    side=side,
                    track_id=track_id,
                    palm_normal=palm_normal,
                    timestamp=now,
                    f_human=f_human,
                    f_human_multiplier=f_human_multiplier,
                )
            finally:
                self._base.engine.orientation_lock_angle = original_angle

            result = GestureResult(
                gesture_type=GestureType.WAVING if gesture == "waving" else GestureType.NONE,
                confidence=confidence,
            )
            if best_result is None or result.confidence > best_result.confidence:
                best_result = result

        return best_result if best_result else GestureResult(
            gesture_type=GestureType.NONE, confidence=0.0
        )

    def reset(self):
        self._base.reset()


# =============================================================================
# 主运行器
# =============================================================================

class AblationRunner:
    """消融实验运行器（全面增强版，支持队列）。"""

    def __init__(self, storage: DataLabStorage) -> None:
        self.storage = storage
        self.config = get_config()
        self._current_exp: Optional[AblationExperiment] = None
        self._cancelled: bool = False
        self._lock = asyncio.Lock()
        self._queue: List[Tuple[AblationExperiment, List[str], ExperimentType, Optional[List[float]]]] = []
        self._queue_task: Optional[asyncio.Task] = None

    async def run_experiment(
        self,
        recording_id: str,
        experiment_type: str = "engine_comparison",
        engine_names: Optional[List[str]] = None,
        threshold_range: Optional[List[float]] = None,
        parent_id: Optional[str] = None,
    ) -> AblationExperiment:
        """
        运行消融实验（入队执行）。

        Args:
            recording_id: 录制会话 ID
            experiment_type: engine_comparison / component_ablation / threshold_sweep / scenario_analysis
            engine_names: 要测试的引擎列表（engine_comparison / scenario_analysis 用）
            threshold_range: 阈值列表（threshold_sweep 用），默认 [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            parent_id: 父实验 ID（全量实验的子实验用）
        """
        exp_type = ExperimentType(experiment_type)

        if exp_type == ExperimentType.COMPONENT_ABLATION:
            engines = [
                "sth_full",
                "sth_no_transformer_bypass",
                "sth_transformer_only",
                "simple_full",
                "simple_no_periodicity",
                "simple_no_pose_gate",
                "triplelock_full",
                "triplelock_no_orientation",
            ]
        elif exp_type == ExperimentType.THRESHOLD_SWEEP:
            engines = engine_names or ["simple_transformer", "transformer"]
        else:
            default_engines = [
                "simple",
                "transformer",
                "triplelock",
                "transformer_triplelock",
                "simple_transformer",
            ]
            engines = engine_names or default_engines

        exp = self.storage.create_experiment(recording_id, engines)
        exp.experiment_type = exp_type
        if parent_id:
            exp.parent_id = parent_id
        self.storage.update_experiment(exp)

        async with self._lock:
            self._queue.append((exp, engines, exp_type, threshold_range))
            if self._queue_task is None or self._queue_task.done():
                self._queue_task = asyncio.create_task(self._process_queue())

        return exp

    async def run_full_suite(
        self,
        positive_recording_ids: List[str],
        negative_recording_ids: List[str],
    ) -> AblationExperiment:
        """一键全量实验：对正样本和负样本分别运行实验，最后合并分析。

        必须至少提供一个正样本（包含 waving）和一个负样本（不包含 waving），
        以便分别评估召回率与误检率。
        """
        if not positive_recording_ids:
            raise ValueError("全量实验至少需要 1 个正样本录制")
        if not negative_recording_ids:
            raise ValueError("全量实验至少需要 1 个负样本录制")

        # 验证录制存在
        for rid in positive_recording_ids + negative_recording_ids:
            rec = self.storage.get_recording(rid)
            if rec is None:
                raise ValueError(f"录制不存在: {rid}")

        # 创建父实验（recording_id 使用第一个正样本作为代表）
        parent = self.storage.create_experiment(positive_recording_ids[0], [])
        parent.experiment_type = ExperimentType.FULL_SUITE
        parent.status = ExperimentStatus.RUNNING
        parent.positive_recording_ids = positive_recording_ids
        parent.negative_recording_ids = negative_recording_ids
        self.storage.update_experiment(parent)

        children: List[AblationExperiment] = []

        # ---- 正样本子实验 ----
        # 1. 引擎横向对比
        pos_ec = await self.run_experiment(
            positive_recording_ids[0], "engine_comparison", parent_id=parent.id
        )
        pos_ec.positive_recording_ids = positive_recording_ids
        self.storage.update_experiment(pos_ec)
        children.append(pos_ec)

        # 2. 组件消融
        pos_ca = await self.run_experiment(
            positive_recording_ids[0], "component_ablation", parent_id=parent.id
        )
        pos_ca.positive_recording_ids = positive_recording_ids
        self.storage.update_experiment(pos_ca)
        children.append(pos_ca)

        # 3. 阈值扫描
        pos_ts = await self.run_experiment(
            positive_recording_ids[0],
            "threshold_sweep",
            engine_names=["simple_transformer", "transformer"],
            parent_id=parent.id,
        )
        pos_ts.positive_recording_ids = positive_recording_ids
        self.storage.update_experiment(pos_ts)
        children.append(pos_ts)

        # 4. 场景分析
        pos_sa = await self.run_experiment(
            positive_recording_ids[0], "scenario_analysis", parent_id=parent.id
        )
        pos_sa.positive_recording_ids = positive_recording_ids
        self.storage.update_experiment(pos_sa)
        children.append(pos_sa)

        # ---- 负样本子实验 ----
        # 仅运行引擎横向对比（核心误检评估）
        neg_ec = await self.run_experiment(
            negative_recording_ids[0], "engine_comparison", parent_id=parent.id
        )
        neg_ec.negative_recording_ids = negative_recording_ids
        self.storage.update_experiment(neg_ec)
        children.append(neg_ec)

        parent.sub_experiment_ids = [c.id for c in children]
        self.storage.update_experiment(parent)

        # 启动父实验 watcher
        asyncio.create_task(self._watch_parent(parent.id, parent.sub_experiment_ids))

        return parent

    async def _watch_parent(self, parent_id: str, child_ids: List[str]) -> None:
        """轮询子实验状态，更新父实验进度和最终状态。"""
        while True:
            await asyncio.sleep(5.0)
            parent = self.storage.get_experiment(parent_id)
            if not parent or parent.status != ExperimentStatus.RUNNING:
                break

            children = [self.storage.get_experiment(cid) for cid in child_ids]
            valid_children = [c for c in children if c]
            if not valid_children:
                break

            # 进度 = 已完成子实验 / 总数
            completed = sum(
                1 for c in valid_children if c.status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED)
            )
            parent.progress = round(completed / len(valid_children), 4)

            # 任一失败则父实验失败
            if any(c.status == ExperimentStatus.FAILED for c in valid_children):
                parent.status = ExperimentStatus.FAILED
                parent.error_message = "部分子实验执行失败"
                self.storage.update_experiment(parent)
                break

            # 全部完成
            if completed == len(valid_children):
                parent.status = ExperimentStatus.COMPLETED
                parent.completed_at = time.time()
                parent.progress = 1.0
                self.storage.update_experiment(parent)
                logger.info("全量实验完成: %s", parent_id)
                # 生成合并报告
                try:
                    from app.datalab.analyzer import AblationAnalyzer
                    analyzer = AblationAnalyzer(self.storage)
                    analyzer.analyze_full_suite(parent_id)
                except Exception as e:
                    logger.error("全量实验合并报告生成失败: %s", e, exc_info=True)
                break

            self.storage.update_experiment(parent)

    async def _process_queue(self) -> None:
        """队列处理器：串行执行排队实验。"""
        while True:
            async with self._lock:
                if not self._queue:
                    self._current_exp = None
                    break
                exp, engines, exp_type, threshold_range = self._queue.pop(0)
                self._current_exp = exp
                self._cancelled = False

            await self._run_background(exp, engines, exp_type, threshold_range)

    async def _run_background(
        self,
        exp: AblationExperiment,
        engine_names: List[str],
        exp_type: ExperimentType,
        threshold_range: Optional[List[float]] = None,
    ) -> None:
        """后台运行实验主体。支持单录制或多录制拼接（全量实验正负样本）。"""
        try:
            exp.status = ExperimentStatus.RUNNING
            self.storage.update_experiment(exp)

            # 决定使用哪些录制 ID（支持全量实验的多录制拼接）
            recording_ids: List[str] = []
            is_negative = False
            if exp.positive_recording_ids:
                recording_ids = exp.positive_recording_ids
            elif exp.negative_recording_ids:
                recording_ids = exp.negative_recording_ids
                is_negative = True
            else:
                recording_ids = [exp.recording_id]

            keypoints_frames, tnlf_frames, detections_frames = self._load_combined_frames(
                recording_ids, is_negative=is_negative
            )

            total_frames = len(keypoints_frames)
            if total_frames == 0:
                raise ValueError("录制数据为空")

            exp.total_frames = total_frames
            self.storage.update_experiment(exp)

            if exp_type == ExperimentType.THRESHOLD_SWEEP:
                await self._run_threshold_sweep(exp, keypoints_frames, tnlf_frames, engine_names, threshold_range, detections_frames)
            elif exp_type == ExperimentType.COMPONENT_ABLATION:
                await self._run_component_ablation(exp, keypoints_frames, tnlf_frames, engine_names, detections_frames)
            elif exp_type == ExperimentType.SCENARIO_ANALYSIS:
                await self._run_scenario_analysis(exp, keypoints_frames, tnlf_frames, engine_names, detections_frames)
            else:
                await self._run_standard_comparison(exp, keypoints_frames, tnlf_frames, engine_names, detections_frames)

            exp.status = ExperimentStatus.COMPLETED
            exp.completed_at = time.time()
            exp.progress = 1.0
            self.storage.update_experiment(exp)
            logger.info("消融实验完成: %s (type=%s)", exp.id, exp_type.value)

            # 若存在父实验，立即触发一次父状态更新
            if getattr(exp, 'parent_id', None):
                await self._check_parent_status(exp.parent_id)

        except Exception as e:
            logger.error("消融实验失败: %s", e, exc_info=True)
            exp.status = ExperimentStatus.FAILED
            exp.error_message = str(e)
            self.storage.update_experiment(exp)
            if getattr(exp, 'parent_id', None):
                await self._check_parent_status(exp.parent_id)

    async def _check_parent_status(self, parent_id: str) -> None:
        """检查父实验状态（由子实验完成后立即触发）。"""
        parent = self.storage.get_experiment(parent_id)
        if not parent or parent.status != ExperimentStatus.RUNNING:
            return
        child_ids = parent.sub_experiment_ids
        if not child_ids:
            return
        children = [self.storage.get_experiment(cid) for cid in child_ids]
        valid_children = [c for c in children if c]
        if not valid_children:
            return
        completed = sum(
            1 for c in valid_children if c.status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED)
        )
        parent.progress = round(completed / len(valid_children), 4)
        if any(c.status == ExperimentStatus.FAILED for c in valid_children):
            parent.status = ExperimentStatus.FAILED
            parent.error_message = "部分子实验执行失败"
        elif completed == len(valid_children):
            parent.status = ExperimentStatus.COMPLETED
            parent.completed_at = time.time()
            parent.progress = 1.0
            try:
                from app.datalab.analyzer import AblationAnalyzer
                analyzer = AblationAnalyzer(self.storage)
                analyzer.analyze_full_suite(parent_id)
            except Exception as e:
                logger.error("全量实验合并报告生成失败: %s", e, exc_info=True)
        self.storage.update_experiment(parent)

    async def cancel(self) -> None:
        """取消当前实验并清空队列。"""
        self._cancelled = True
        async with self._lock:
            self._queue.clear()

    # ------------------------------------------------------------------
    # 标准引擎对比
    # ------------------------------------------------------------------

    async def _run_standard_comparison(
        self, exp: AblationExperiment,
        keypoints_frames: List[Dict],
        tnlf_frames: List[Dict],
        engine_names: List[str],
        detections_frames: List[Dict],
    ) -> None:
        engines = self._instantiate_engines(engine_names)
        logger.info("标准引擎对比: exp=%s engines=%s frames=%d", exp.id, engine_names, len(keypoints_frames))
        await self._infer_all_frames(exp, keypoints_frames, tnlf_frames, engines, detections_frames)

    # ------------------------------------------------------------------
    # 组件消融
    # ------------------------------------------------------------------

    async def _run_component_ablation(
        self, exp: AblationExperiment,
        keypoints_frames: List[Dict],
        tnlf_frames: List[Dict],
        engine_names: List[str],
        detections_frames: List[Dict],
    ) -> None:
        engines = self._instantiate_ablation_engines(engine_names)
        logger.info("组件消融: exp=%s variants=%s frames=%d", exp.id, list(engines.keys()), len(keypoints_frames))
        await self._infer_all_frames(exp, keypoints_frames, tnlf_frames, engines, detections_frames)

    # ------------------------------------------------------------------
    # 阈值扫描
    # ------------------------------------------------------------------

    async def _run_threshold_sweep(
        self, exp: AblationExperiment,
        keypoints_frames: List[Dict],
        tnlf_frames: List[Dict],
        engine_names: List[str],
        threshold_range: Optional[List[float]],
        detections_frames: List[Dict],
    ) -> None:
        # 更细粒度阈值扫描（0.05~0.95，步长0.05），确保曲线覆盖完整区间
        if threshold_range:
            thresholds = threshold_range
        else:
            thresholds = [round(x, 2) for x in np.arange(0.05, 1.0, 0.05).tolist()]
        total = len(keypoints_frames)
        processed = 0

        for engine_name in engine_names:
            for thr in thresholds:
                if self._cancelled:
                    return

                engines = {engine_name: self._instantiate_engine_with_threshold(engine_name, thr)}

                for idx, kp_frame in enumerate(keypoints_frames):
                    if self._cancelled:
                        return
                    row = self._infer_single_frame(kp_frame, tnlf_frames, idx, engines, detections_frames)
                    row["threshold"] = thr
                    self.storage.append_frame_result(exp.id, row)

                processed += total
                exp.current_frame = min(processed, total * len(engine_names) * len(thresholds))
                exp.progress = round(processed / (total * len(engine_names) * len(thresholds)), 4)
                self.storage.update_experiment(exp)
                await asyncio.sleep(0)

                # 释放引擎状态
                engines[engine_name].reset()

    # ------------------------------------------------------------------
    # 场景分析
    # ------------------------------------------------------------------

    async def _run_scenario_analysis(
        self, exp: AblationExperiment,
        keypoints_frames: List[Dict],
        tnlf_frames: List[Dict],
        engine_names: List[str],
        detections_frames: List[Dict],
    ) -> None:
        engines = self._instantiate_engines(engine_names)
        logger.info("场景分析: exp=%s engines=%s frames=%d", exp.id, engine_names, len(keypoints_frames))

        for idx, kp_frame in enumerate(keypoints_frames):
            if self._cancelled:
                return

            tnlf = tnlf_frames[idx] if idx < len(tnlf_frames) else {}
            row = self._infer_single_frame(kp_frame, tnlf_frames, idx, engines, detections_frames)

            # 计算场景标签
            v_left = float(tnlf.get("left_velocity_mag", 0.0))
            v_right = float(tnlf.get("right_velocity_mag", 0.0))
            v_max = max(v_left, v_right)

            if v_max < 0.03:
                row["scenario_velocity"] = "static"
            elif v_max < 0.1:
                row["scenario_velocity"] = "slow"
            else:
                row["scenario_velocity"] = "fast"

            left_valid = bool(tnlf.get("left_tnlf_valid", False))
            right_valid = bool(tnlf.get("right_tnlf_valid", False))
            if left_valid and right_valid:
                row["scenario_hand"] = "both"
            elif left_valid:
                row["scenario_hand"] = "left"
            elif right_valid:
                row["scenario_hand"] = "right"
            else:
                row["scenario_hand"] = "none"

            keypoints = np.array(kp_frame.get("keypoints", []))
            if keypoints.ndim == 1:
                keypoints = keypoints.reshape(-1, 3)
            if len(keypoints) >= 12:
                # 肩宽作为距离代理
                shoulder_dist = float(np.linalg.norm(keypoints[5, :2] - keypoints[6, :2]))
                if shoulder_dist < 80:
                    row["scenario_distance"] = "far"
                elif shoulder_dist < 150:
                    row["scenario_distance"] = "mid"
                else:
                    row["scenario_distance"] = "near"
            else:
                row["scenario_distance"] = "unknown"

            self.storage.append_frame_result(exp.id, row)

            exp.current_frame = idx + 1
            exp.progress = round((idx + 1) / len(keypoints_frames), 4)
            if idx % 30 == 0:
                self.storage.update_experiment(exp)
            if idx % 5 == 0:
                await asyncio.sleep(0)

    # ------------------------------------------------------------------
    # 通用推理
    # ------------------------------------------------------------------

    async def _infer_all_frames(
        self, exp: AblationExperiment,
        keypoints_frames: List[Dict],
        tnlf_frames: List[Dict],
        engines: Dict[str, Any],
        detections_frames: List[Dict],
    ) -> None:
        for idx, kp_frame in enumerate(keypoints_frames):
            if self._cancelled:
                return

            row = self._infer_single_frame(kp_frame, tnlf_frames, idx, engines, detections_frames)
            self.storage.append_frame_result(exp.id, row)

            exp.current_frame = idx + 1
            exp.progress = round((idx + 1) / len(keypoints_frames), 4)
            if idx % 30 == 0:
                self.storage.update_experiment(exp)
            if idx % 5 == 0:
                await asyncio.sleep(0)

    def _infer_single_frame(
        self, kp_frame: Dict, tnlf_frames: List[Dict], idx: int, engines: Dict[str, Any],
        detections_frames: List[Dict],
    ) -> Dict[str, Any]:
        frame_idx = kp_frame.get("frame_idx", idx)
        timestamp = kp_frame.get("timestamp", 0.0)
        keypoints = np.array(kp_frame.get("keypoints", []))
        if keypoints.ndim == 1:
            keypoints = keypoints.reshape(-1, 3)

        tnlf = tnlf_frames[idx] if idx < len(tnlf_frames) else {}
        left_wl = _to_array(tnlf.get("left_wrist_local"))
        right_wl = _to_array(tnlf.get("right_wrist_local"))
        left_pn = _to_array(kp_frame.get("left_palm_normal"))
        right_pn = _to_array(kp_frame.get("right_palm_normal"))

        kwargs = {
            "keypoints": keypoints,
            "track_id": kp_frame.get("track_id", "person_1"),
            "left_palm_normal": left_pn,
            "right_palm_normal": right_pn,
            "frame_timestamp": timestamp,
            "active_track_ids": {kp_frame.get("track_id", "person_1")},
            "left_wrist_local": left_wl,
            "right_wrist_local": right_wl,
            "left_tnlf_valid": tnlf.get("left_tnlf_valid", False),
            "right_tnlf_valid": tnlf.get("right_tnlf_valid", False),
            "left_velocity_mag": tnlf.get("left_velocity_mag", 0.0),
            "right_velocity_mag": tnlf.get("right_velocity_mag", 0.0),
            "left_theta1": tnlf.get("left_theta1", 0.0),
            "left_theta2": tnlf.get("left_theta2", 0.0),
            "left_ext_ratio": tnlf.get("left_ext_ratio", 0.0),
            "right_theta1": tnlf.get("right_theta1", 0.0),
            "right_theta2": tnlf.get("right_theta2", 0.0),
            "right_ext_ratio": tnlf.get("right_ext_ratio", 0.0),
        }

        row: Dict[str, Any] = {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
        }
        for name, engine in engines.items():
            t0 = time.perf_counter()
            try:
                result = engine.recognize(**kwargs)
                gesture = (
                    result.gesture_type.value
                    if hasattr(result.gesture_type, "value")
                    else str(result.gesture_type)
                )
                conf = result.confidence
            except Exception as e:
                logger.warning("引擎 %s 推理失败 frame=%d: %s", name, frame_idx, e)
                gesture = "error"
                conf = 0.0
            latency = (time.perf_counter() - t0) * 1000
            row[f"{name}_gesture"] = gesture
            row[f"{name}_confidence"] = round(max(0.0, min(1.0, conf)), 4)
            row[f"{name}_latency_ms"] = round(latency, 3)

        row["velocity_left"] = round(kwargs["left_velocity_mag"], 4)
        row["velocity_right"] = round(kwargs["right_velocity_mag"], 4)

        # 携带 ground truth（来自录制时的生产引擎检测结果）
        det = detections_frames[idx] if idx < len(detections_frames) else {}
        row["gt_gesture"] = det.get("gesture", "none")
        row["gt_conf"] = det.get("gesture_conf", 0.0)
        row["recording_id"] = det.get("_recording_id", "")
        return row

    # ------------------------------------------------------------------
    # 多录制数据拼接（全量实验正负样本支持）
    # ------------------------------------------------------------------

    def _load_combined_frames(
        self, recording_ids: List[str], is_negative: bool = False
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """加载并拼接多个录制的 keypoints、tnlf、detections 数据。

        对于负样本，强制将所有 detections 的 gesture 设为 'none'，
        因为负样本的 ground truth 就是无 waving。
        """
        keypoints_frames: List[Dict] = []
        tnlf_frames: List[Dict] = []
        detections_frames: List[Dict] = []

        for rid in recording_ids:
            kp_list = list(self.storage.iter_keypoints(rid))
            tnlf_list = list(self.storage.iter_tnlf(rid))
            det_list = list(self.storage.iter_detections(rid))

            # 统一长度（以 keypoints 为准）
            n = len(kp_list)
            if n == 0:
                logger.warning("录制 %s 无 keypoints 数据，跳过", rid)
                continue

            for i in range(n):
                kp = kp_list[i]
                tnlf = tnlf_list[i] if i < len(tnlf_list) else {}
                det = det_list[i] if i < len(det_list) else {}

                # 标记来源录制 ID，便于分析时区分
                det = dict(det)
                det["_recording_id"] = rid

                if is_negative:
                    # 负样本 ground truth 强制为 none
                    det["gesture"] = "none"
                    det["gesture_conf"] = 0.0

                keypoints_frames.append(kp)
                tnlf_frames.append(tnlf)
                detections_frames.append(det)

        return keypoints_frames, tnlf_frames, detections_frames

    # ------------------------------------------------------------------
    # 引擎工厂
    # ------------------------------------------------------------------

    def _instantiate_engines(self, engine_names: List[str]) -> Dict[str, Any]:
        """实例化标准引擎。"""
        engines: Dict[str, Any] = {}
        for name in engine_names:
            engines[name] = self._instantiate_engine(name)
        return engines

    def _instantiate_engine(self, name: str) -> Any:
        from app.ai.gesture import (
            SimpleGestureRecognizer,
            TransformerGestureRecognizer,
            GestureRecognizer,
            HybridGestureRecognizer,
            SimpleTransformerHybridRecognizer,
        )

        model_path = self.config.ai.transformer_model_path
        threshold = self.config.ai.transformer_confidence_threshold

        if name == "simple":
            return SimpleGestureRecognizer()
        elif name == "transformer":
            return TransformerGestureRecognizer(
                model_path=model_path, confidence_threshold=threshold,
            )
        elif name == "triplelock":
            recognizer = GestureRecognizer()
            # 消融数据 TNLF 振幅较小，降低运动锁阈值以保证公平比较
            recognizer.engine.motion_amp_min = 0.02
            return recognizer
        elif name == "transformer_triplelock":
            hybrid = HybridGestureRecognizer(
                transformer_model_path=model_path,
                transformer_threshold=threshold,
            )
            hybrid.triplelock.engine.motion_amp_min = 0.02
            return hybrid
        elif name == "simple_transformer":
            return SimpleTransformerHybridRecognizer(
                transformer_model_path=model_path,
                transformer_threshold=threshold,
            )
        else:
            raise ValueError(f"未知引擎: {name}")

    def _instantiate_engine_with_threshold(self, name: str, threshold: float) -> Any:
        from app.ai.gesture import (
            TransformerGestureRecognizer,
            SimpleTransformerHybridRecognizer,
        )
        model_path = self.config.ai.transformer_model_path

        if name == "transformer":
            return TransformerGestureRecognizer(
                model_path=model_path, confidence_threshold=threshold,
            )
        elif name == "simple_transformer":
            return SimpleTransformerHybridRecognizer(
                transformer_model_path=model_path,
                transformer_threshold=threshold,
            )
        else:
            return self._instantiate_engine(name)

    def _instantiate_ablation_engines(self, engine_names: List[str]) -> Dict[str, Any]:
        """实例化组件消融变体引擎。"""
        engines: Dict[str, Any] = {}
        model_path = self.config.ai.transformer_model_path
        threshold = self.config.ai.transformer_confidence_threshold

        from app.ai.gesture import (
            SimpleGestureRecognizer,
            TransformerGestureRecognizer,
            GestureRecognizer,
            SimpleTransformerHybridRecognizer,
        )

        for name in engine_names:
            if name == "sth_full":
                engines[name] = SimpleTransformerHybridRecognizer(
                    transformer_model_path=model_path,
                    transformer_threshold=threshold,
                )
            elif name == "sth_no_transformer_bypass":
                base = SimpleTransformerHybridRecognizer(
                    transformer_model_path=model_path,
                    transformer_threshold=threshold,
                )
                engines[name] = _STHNoTransformBypass(base)
            elif name == "sth_strict_and":
                base = SimpleTransformerHybridRecognizer(
                    transformer_model_path=model_path,
                    transformer_threshold=threshold,
                )
                engines[name] = _STHStrictAnd(base)
            elif name == "sth_transformer_only":
                base = SimpleTransformerHybridRecognizer(
                    transformer_model_path=model_path,
                    transformer_threshold=threshold,
                )
                engines[name] = _STHTransformerOnly(base)
            elif name == "simple_full":
                engines[name] = SimpleGestureRecognizer()
            elif name == "simple_no_periodicity":
                base = SimpleGestureRecognizer()
                engines[name] = _SimpleNoPeriodicity(base)
            elif name == "simple_no_pose_gate":
                base = SimpleGestureRecognizer()
                engines[name] = _SimpleNoPoseGate(base)
            elif name == "triplelock_full":
                recognizer = GestureRecognizer()
                recognizer.engine.motion_amp_min = 0.02
                engines[name] = recognizer
            elif name == "triplelock_no_orientation":
                base = GestureRecognizer()
                base.engine.motion_amp_min = 0.02
                engines[name] = _TripleLockNoOrientation(base)
            else:
                logger.warning("未知消融变体: %s", name)
        return engines

    def get_progress(self) -> Optional[AblationExperiment]:
        """获取当前实验进度。"""
        return self._current_exp


def _to_array(val: Any) -> Optional[np.ndarray]:
    """将 list/tuple 转为 numpy array。"""
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return val
    try:
        return np.array(val, dtype=np.float32)
    except (ValueError, TypeError):
        return None
