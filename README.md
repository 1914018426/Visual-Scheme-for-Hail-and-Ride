# Hailuo Vision — 视觉招手即停系统

基于 **YOLO11s-Pose** + **ByteTrack** + **TemporalKeypointTransformer** + **Simple 规则后验过滤** 的实时手势识别系统，专为智能网约车/无人车的"招手即停"场景设计，支持车辆运动状态下对路边乘客招手动作的精确识别。

## 系统特性

| 特性 | 描述 |
|------|------|
| 人体姿态检测 | YOLO11s-Pose，640×640 输入，GPU 半精度推理 (~10-15ms/帧) |
| 多目标跟踪 | ByteTrack 跨帧关联，支持最多 20 人同时检测 |
| Transformer 时序识别 | 12 维 TNLF 特征 × 45 帧滑窗，val_f1=0.897 |
| Simple 规则后验过滤 | 手腕高于手肘 + 面部可见作为硬性规则，过滤挠头等假阳性 |
| Torso-Normalized Local Frame | 以躯干为参考系消除车辆移动导致的伪运动 |
| 前臂朝向代理 | 基于 wrist-elbow 向量估算手掌法向量，无需 MediaPipe Hands |
| 自适应推流 | 根据推理负载动态调节分辨率与 JPEG 质量，维持实时性 |
| 实时日志 | WebSocket 实时日志推送，前端内置日志面板 |
| 多路视频流 | RTSP / RTMP / HTTP / 本地摄像头 / 本地文件 |
| Docker 一键部署 | 容器化编排，NVIDIA GPU 直通 |

## 快速开始

### 前置要求

- Docker >= 20.10
- Docker Compose >= 2.0
- NVIDIA GPU + NVIDIA Container Toolkit（CUDA 12.x 兼容）
- 至少 4GB 可用显存

### 1. 克隆项目

```bash
git clone https://github.com/1914018426/Visual-Scheme-for-Hail-and-Ride
cd Visual-Scheme-for-Hail-and-Ride
```

### 2. 准备模型

首次启动时会自动下载 YOLO 模型。如需离线部署，预先将模型放入 `./models/`：

```bash
mkdir -p models
# 下载 yolo11s-pose（约 16MB）
wget -O models/yolo11s-pose.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt
```

> Transformer 模型位于 `./models/transformer/waving_transformer_real.pt`，已包含在仓库中。

### 3. 配置摄像头

编辑 `docker-compose.yml`，将 `CAMERA_FRONT` 替换为你的视频源：

```yaml
environment:
  - CAMERA_FRONT=rtmp://your-rtmp-server/live/stream
  # 或 RTSP: rtsp://192.168.1.100:554/stream
  # 或本地摄像头: 0
  # 或本地文件: /data/video.mp4
```

### 4. 启动服务

```bash
docker compose up -d
```

首次构建可能需要 3-5 分钟（下载 PyTorch、Ultralytics 等依赖）。

### 5. 访问系统

| 入口 | 地址 | 说明 |
|------|------|------|
| Web 界面 | http://localhost:18080 | Nginx 统一代理（推荐） |
| 后端 API | http://localhost:8001/api/docs | Swagger/OpenAPI 文档 |
| 后端直连 | http://localhost:8001 | FastAPI 服务 |

### 停止服务

```bash
docker compose down
```

---

## 算法架构

### 1. 人体姿态检测 — YOLO11s-Pose + ByteTrack

- **模型**：`yolo11s-pose.pt`（~16MB，速度优先）
- **输入**：640×640，fp16 半精度
- **置信度阈值**：`0.35`
- **最大检测数**：`20` 人
- **跟踪器**：ByteTrack，`bytetrack.yaml`

COCO17 关键点索引：

| 索引 | 关键点 | 索引 | 关键点 |
|------|--------|------|--------|
| 0 | 鼻子 | 9 | 左手腕 |
| 1 | 左眼 | 10 | 右手腕 |
| 2 | 右眼 | 11 | 左髋 |
| 3 | 左耳 | 12 | 右髋 |
| 4 | 右耳 | 13 | 左膝 |
| 5 | 左肩 | 14 | 右膝 |
| 6 | 右肩 | 15 | 左踝 |
| 7 | 左肘 | 16 | 右踝 |
| 8 | 右肘 | | |

### 2. Torso-Normalized Local Frame (TNLF)

以人体自身为参考系，消除车辆移动对轨迹分析的干扰：

```
origin      = (left_shoulder + right_shoulder) / 2
e_x         = normalize(right_shoulder - left_shoulder)
e_y         = normalize(mid_hip - origin)
torso_scale = |mid_hip - origin|

wrist_local = (dot(wrist - origin, e_x) / torso_scale,
               dot(wrist - origin, e_y) / torso_scale)
```

**单位**：躯干长度（torso_units）。车辆匀速行驶时，静止路人的 `wrist_local` 方差趋近于 0。

### 3. Transformer 时序识别器

基于 **TemporalKeypointTransformer**，输入为 45 帧 × 12 维 TNLF 特征：

- `d_model=64`，2 层 encoder，4 头注意力
- 输入特征：左右 wrist_local (x,y)、velocity、theta1、theta2、ext_ratio、active_arm 等
- TorchScript 导出，首次加载后常驻内存
- 模型路径：`./models/transformer/waving_transformer_real.pt`

### 4. Simple 规则后验过滤

Transformer 为主检测器，Simple 引擎仅用于过滤假阳性（如挠头）。**硬性规则不可变**：

1. **面部可见**：鼻子置信度 ≥ 0.25，且至少一只眼睛可见
2. **手腕高于手肘**：至少一侧手腕在图像坐标中高于手肘（wrist_y < elbow_y）

**过滤逻辑**：

| Transformer | Simple 周期检测 | 结果 |
|-------------|-----------------|------|
| NONE | — | NONE |
| WAVING (conf ≤ 0.7) | 未通过 | NONE |
| WAVING (conf > 0.7) | 未通过 | WAVING（移动状态下放宽周期检测） |
| WAVING | 通过 | WAVING（取 Transformer 置信度） |

### 5. 前臂朝向代理（无需 MediaPipe）

当 `ENABLE_HAND_DETECTION=false`（默认）时，使用 wrist-elbow 向量作为手掌法向量的代理：

```python
palm_normal = normalize(wrist - elbow)  # 前臂方向近似掌心朝向
```

相比 MediaPipe Hands，可提升 4-8 倍推流帧率，且无 GIL 争抢问题。

### 6. 自适应推流

后端根据单帧"推理+编码"耗时动态调节：

- **超时**（> 100ms）：降低短边分辨率（-12px）、降低 JPEG 质量（-2）
- **余量**（< 38ms）：提升短边分辨率（+8px）、提升 JPEG 质量（+1）

分辨率范围：`ADAPTIVE_MIN_SHORT_SIDE` ~ `ADAPTIVE_MAX_SHORT_SIDE`。

---

## 项目结构

```
├── docker-compose.yml          # Docker Compose 编排
├── nginx.conf                  # Nginx 反向代理
├── README.md
├── backend/                    # FastAPI 后端
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── entrypoint.sh
│   └── app/
│       ├── main.py               # FastAPI 入口
│       ├── config.py             # 配置中心（环境变量驱动）
│       ├── ai/
│       │   ├── detector.py       # YOLO + ByteTrack + 绘制
│       │   ├── gesture.py        # 手势引擎（Simple / Transformer / Hybrid）
│       │   ├── local_frame.py    # TNLF 局部参考系
│       │   ├── facing.py         # 面部过滤（零模型）
│       │   ├── slerp.py          # 法向量平滑
│       │   ├── iri.py            # IRI 意图刚性指数
│       │   └── transformer/      # Transformer 模型定义与训练脚本
│       ├── api/
│       │   ├── routes.py         # REST API（摄像头管理、配置）
│       │   ├── ws.py             # WebSocket 视频推流
│       │   └── logs.py           # WebSocket 日志广播
│       └── stream/
│           ├── handler.py        # 视频流解码（OpenCV + FFmpeg）
│           └── manager.py        # 多路流管理
├── frontend/                   # React + Vite + Tailwind
│   ├── Dockerfile
│   ├── nginx-default.conf
│   └── src/
│       ├── components/
│       │   ├── VideoPanel.tsx
│       │   ├── GestureOverlay.tsx
│       │   ├── VideoGrid.tsx
│       │   └── LogPanel.tsx      # 实时日志面板
│       ├── hooks/
│       │   ├── useWebSocket.ts
│       │   └── useCameraConfig.ts
│       └── types/
│           └── index.ts
└── models/                     # 模型持久化目录（运行时自动生成）
    ├── yolo11s-pose.pt
    └── transformer/
        └── waving_transformer_real.pt
```

---

## 配置说明

所有参数通过 `docker-compose.yml` 环境变量配置，无需修改代码：

### AI 推理

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `YOLO_MODEL` | `yolo11s-pose.pt` | 姿态检测模型 |
| `AI_INFERENCE_IMGSZ` | `640` | YOLO 输入分辨率 |
| `AI_INFERENCE_HALF` | `true` | fp16 半精度推理 |
| `AI_CONF_THRESHOLD` | `0.35` | 人体检测置信度阈值 |
| `AI_MAX_DETECTIONS` | `20` | 最大检测人数 |
| `ENABLE_TRACKING` | `true` | ByteTrack 跟踪开关 |
| `ENABLE_HAND_DETECTION` | `false` | MediaPipe Hands（高耗 CPU，默认关闭） |

### 视频流

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `STREAM_FPS` | `15` | 采集/推流目标帧率 |
| `STREAM_WIDTH` | `1280` | 采集分辨率宽 |
| `STREAM_HEIGHT` | `720` | 采集分辨率高 |
| `STREAM_BUFFER_SIZE` | `5` | 帧缓冲队列大小 |
| `STREAM_LOW_LATENCY` | `true` | 低延迟模式（网络流排空缓冲） |
| `CAMERA_FRONT` | *(必填)* | 视频源地址 |
| `OPENCV_FFMPEG_CAPTURE_OPTIONS` | `rtsp_transport;tcp...` | FFmpeg 低延迟参数 |

### 手势引擎

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `GESTURE_ENGINE` | `simple-transformer` | 引擎模式：`simple` / `transformer` / `simple-transformer` / `hybrid` / `triplelock` |
| `TRANSFORMER_MODEL_PATH` | `/app/models/transformer/...` | Transformer 模型路径 |
| `TRANSFORMER_CONFIDENCE_THRESHOLD` | `0.5` | Transformer 输出阈值 |

### 姿态锁（硬性规则）

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `GESTURE_THETA1_HAILING_MIN` | `25.0` | 肩-肘抬起角度 θ1（°） |
| `GESTURE_THETA2_STRAIGHT_MIN` | `15.0` | 肘-腕伸直角度 θ2（°） |
| `GESTURE_ARM_EXTENSION_MIN` | `0.10` | 手臂伸展比例 |
| `GESTURE_POSE_MIN_FRAMES` | `3` | 姿态锁最小持续帧数 |

### 运动锁（周期性检测，软规则）

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `GESTURE_MOTION_FREQ_MIN` | `0.35` | 频率下限（Hz） |
| `GESTURE_MOTION_FREQ_MAX` | `3.0` | 频率上限（Hz） |
| `GESTURE_MOTION_AMP_MIN` | `0.1` | 最小振幅（torso_units） |
| `GESTURE_MOTION_SPEED_MIN` | `0.05` | 速度释放阈值 |
| `GESTURE_MOTION_MIN_FRAMES` | `3` | 运动锁最小持续帧数 |

### 面部过滤

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `GESTURE_FACING_HARD_THRESHOLD` | `0.25` | 硬过滤阈值（面部不可见直接丢弃） |
| `GESTURE_FACING_SOFT_THRESHOLD` | `0.6` | 软过滤上限 |

### 推流自适应

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `WS_FRAME_BUDGET_MS` | `100` | 单帧推理+编码预算（ms） |
| `WS_PUSH_INTERVAL` | `0.04` | 推流间隔（s） |
| `JPEG_QUALITY` | `75` | MJPEG 编码质量 |
| `ADAPTIVE_MIN_SHORT_SIDE` | `384` | 自适应最小短边 |
| `ADAPTIVE_MAX_SHORT_SIDE` | `720` | 自适应最大短边 |

---

## 引擎模式对比

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `simple-transformer`（推荐） | Transformer 主检 + Simple 后滤 | 平衡精度与召回，默认 |
| `transformer` | 纯 Transformer | 高召回，可能误判挠头 |
| `simple` | 纯规则引擎 | 低耗 CPU，精度有限 |
| `hybrid` | Transformer + TripleLock | 实验性，双重验证 |
| `triplelock` | 姿态锁+朝向锁+运动锁 | 早期方案，已被 Transformer 替代 |

---

## 开发模式

### 前端

```bash
cd frontend
npm install
npm run dev
```

### 后端

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 常见问题

### Q1: 构建时模型下载超时？

预先将 `.pt` 模型放入 `./models/`，或配置代理后构建：

```bash
export HTTP_PROXY=http://host.docker.internal:7890
docker compose up -d --build
```

国内环境已内置 `HF_ENDPOINT=https://hf-mirror.com` 和阿里云 PyPI 镜像。

### Q2: 视频流卡顿或延迟高？

1. 检查网络流是否使用 TCP 传输（`rtsp_transport;tcp`）
2. 降低 `STREAM_WIDTH` / `STREAM_HEIGHT`
3. 降低 `JPEG_QUALITY`
4. 确保 GPU 正常直通（`nvidia-smi` 在容器内可见）

### Q3: 车辆移动时检测不到招手？

1. 确认使用 `simple-transformer` 或 `transformer` 引擎（`GESTURE_ENGINE`）
2. 检查 YOLO 肩/髋关键点置信度是否过低（影响 TNLF 计算）
3. 尝试降低 `TRANSFORMER_CONFIDENCE_THRESHOLD`（如 0.45）

### Q4: 静止时挠头被误判为招手？

使用 `simple-transformer` 引擎（默认）。Simple 规则的周期性检测可有效过滤挠头等非周期性动作。

---

## 端口说明

| 端口 | 服务 | 用途 |
|------|------|------|
| 18080 | Nginx | 统一入口（推荐） |
| 8001 | Backend | REST API + WebSocket（直接访问） |
| 5173 | Frontend | 前端页面（直接访问） |

---

## 许可证

MIT License
