#!/usr/bin/env python3
"""
YOLO11-Pose TensorRT 导出脚本

用法:
    python scripts/export_tensorrt.py --model yolo11x-pose.pt --imgsz 896 --half

环境要求:
    - CUDA >= 11.8
    - TensorRT >= 8.6
    - ultralytics >= 8.3.0

说明:
    - FP16 (half=True) 在精度损失极小的前提下，推理速度提升 40~60%
    - 导出后的 .engine 文件可直接被 PoseDetector 加载（Ultralytics 自动识别）
    - 若国内下载模型失败，请先设置 HF_ENDPOINT=https://hf-mirror.com
"""

import argparse
import os
import sys

# 允许从项目根目录导入 backend 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ai.detector import _download_yolo_weights_cn


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 YOLO11-Pose 为 TensorRT 引擎")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11x-pose.pt",
        help="输入模型文件名或路径 (默认: yolo11x-pose.pt)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=896,
        help="推理输入尺寸 (默认: 896，需为 32 倍数)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=True,
        help="启用 FP16 半精度 (默认开启)",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="禁用 FP16，使用 FP32",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="目标 GPU 设备号 (默认: 0)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models",
        help="模型存放目录 (默认: ./models)",
    )
    parser.add_argument(
        "--workspace",
        type=float,
        default=4.0,
        help="TensorRT workspace (GB, 默认: 4.0)",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=True,
        help="ONNX simplify (默认开启)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        default=False,
        help="启用动态 batch size (默认关闭，固定 batch=1)",
    )
    args = parser.parse_args()

    half = args.half and not args.no_half
    model_path = os.path.join(args.model_dir, os.path.basename(args.model))

    os.makedirs(args.model_dir, exist_ok=True)

    # 若本地无权重，尝试国内镜像下载
    if not os.path.isfile(model_path):
        print(f"本地未找到模型，尝试从国内镜像下载: {args.model}")
        ok = _download_yolo_weights_cn(model_path, args.model)
        if not ok:
            print("下载失败，请手动放置权重文件到:", model_path)
            sys.exit(1)

    print(f"正在加载模型: {model_path}")
    try:
        from ultralytics import YOLO

        model = YOLO(model_path, task="pose")
    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)

    print(f"开始导出 TensorRT 引擎...")
    print(f"  输入尺寸: {args.imgsz}x{args.imgsz}")
    print(f"  FP16: {half}")
    print(f"  设备: cuda:{args.device}")
    print(f"  Workspace: {args.workspace} GB")
    print(f"  Dynamic batch: {args.dynamic}")

    try:
        export_path = model.export(
            format="engine",
            imgsz=args.imgsz,
            half=half,
            device=args.device,
            workspace=args.workspace,
            simplify=args.simplify,
            dynamic=args.dynamic,
        )
        print(f"导出成功: {export_path}")
        print(
            "使用方法: 在 docker-compose.yml 中将 YOLO_MODEL 设置为上述 .engine 文件路径"
        )
    except Exception as e:
        print(f"导出失败: {e}")
        print("常见原因:")
        print("  1. TensorRT 未正确安装 (pip install tensorrt)")
        print("  2. CUDA 版本与 TensorRT 不匹配")
        print("  3. 显存不足 (尝试减小 workspace 或 imgsz)")
        sys.exit(1)


if __name__ == "__main__":
    main()
