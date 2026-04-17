# Visual Scheme for Hail-and-Ride | 扬招视觉方案

[English](#english) · [中文](#中文)

---

## English

### Overview

End-to-end **person detection (YOLOv8)** plus **hand gesture recognition (MediaPipe Hands)** for retail / autonomous vehicle scenarios: detect humans in the frame and classify **waving** (including greeting-style and hailing-style waves) as a single “wave” target.

- **Stack**: FastAPI, Ultralytics YOLOv8n, MediaPipe Hands, OpenCV  
- **Runtime**: Docker (CUDA base image); **GPU optional** — if CUDA is unavailable, inference falls back to **CPU** automatically.

### Features

- HTTP **`POST /detect`** — upload an image, return per-person bbox and gesture (`is_waving`, `confidence`, `wave_style`: `greeting` / `hail` / `mixed`).  
- **WebSocket `/ws`** — binary JPEG frames, JSON response with targets.  
- **`GET /health`** — service and GPU availability.

### Quick start (Docker)

```bash
docker compose up -d --build
```

API: `http://localhost:8010` (mapped from container port 8000).

The image downloads `yolov8n.pt` during build; local `./models` can override via volume.

### Configuration

- **Gesture thresholds** are in `src/gesture_recognizer.py` (greeting / hail ranges, reversal counts, confidence weights).  
- **YOLO device**: `src/yolo_detector.py` uses CUDA when available, else CPU.

### API sketch

| Method | Path | Description |
|--------|------|-------------|
| POST | `/detect` | `multipart/form-data` file field — JPEG/PNG image |
| WS | `/ws` | Raw image bytes per message |
| GET | `/health` | `{"status":"ok","gpu_available":bool,...}` |

### License

Use and modify according to your project policy. Third-party models (YOLO, MediaPipe) follow their respective licenses.

---

## 中文

### 简介

以 **YOLOv8** 做人体检测、**MediaPipe Hands** 做手部与招手判断，面向无人零售车、扬招等场景：在画面中标出行人，并将 **招手**（含「打招呼式」与「打车式」摆动）统一识别为目标手势。

- **技术栈**：FastAPI、Ultralytics YOLOv8n、MediaPipe Hands、OpenCV  
- **部署**：Docker（CUDA 基础镜像）；**GPU 非必须** — 无 CUDA 时 **自动使用 CPU** 推理。

### 功能要点

- **`POST /detect`**：上传单张图片，返回每人边界框与手势（`is_waving`、`confidence`、`wave_style`：`greeting` / `hail` / `mixed`）。  
- **WebSocket `/ws`**：逐帧发送图像二进制，返回 JSON 目标列表。  
- **`GET /health`**：健康检查与是否使用 GPU。

### 快速启动（Docker）

```bash
docker compose up -d --build
```

浏览器或客户端访问：`http://localhost:8010`（宿主机 **8010** 映射容器 **8000**）。

构建镜像时会下载 `yolov8n.pt`；本地 `./models` 可通过卷覆盖。

### 参数与调优

- 招手阈值、置信度计算见 **`src/gesture_recognizer.py`**（打招呼 / 打车两种模式与 `wave_style`）。  
- 检测设备选择见 **`src/yolo_detector.py`**（有 CUDA 用 GPU，否则 CPU）。

### 接口摘要

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/detect` | `multipart/form-data` 上传图片 |
| WS | `/ws` | 每帧发送图像字节流 |
| GET | `/health` | 服务状态 |

### 许可证

代码按项目约定使用；YOLO、MediaPipe 等遵循各自开源协议。
