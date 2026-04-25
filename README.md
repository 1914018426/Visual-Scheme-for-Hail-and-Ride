# AI Gesture Detection System

基于 YOLOv8-pose 与 MediaPipe Hands 的实时手势识别检测系统，支持"打招呼"与"打车"等手势的精确识别与方向决策。

## 系统特性

| 特性 | 描述 |
|------|------|
| 人体姿态检测 | 基于 YOLOv8-pose，实时检测人体 17 个关键点 |
| 手势识别 | 支持 `greeting`（打招呼）、`hailing`（打车）、`hand_up`（举手）三种手势 |
| 手掌朝向检测 | 基于 MediaPipe Hands 21 点 landmark，判断手掌是否朝向画面 |
| 时间窗口抗抖动 | 2.5s+ 滑动窗口 + 周期性检测，过滤偶发误动作 |
| 多路视频流 | 支持本地摄像头、RTSP/RTMP/HTTP 网络视频流 |
| 实时推流 | WebSocket 实时推送 MJPEG 帧与检测结果 |
| Docker 部署 | 容器化一键部署，环境隔离，易于维护 |

## 快速开始

### 前置要求

- Docker >= 20.10
- Docker Compose >= 2.0
- NVIDIA GPU（推荐，用于 AI 推理加速）
- 至少 4GB 可用内存

### 一键启动

```bash
# 1. 克隆项目
git clone https://github.com/1914018426/Visual-Scheme-for-Hail-and-Ride
cd hailuo-car

# 2. 启动所有服务（后台模式）
docker compose up -d

# 3. 查看服务状态
docker compose ps

# 4. 访问系统
# Web 界面: http://localhost:18080
# API 文档: http://localhost:8001/api/docs
# 直接后端: http://localhost:8001
```

### 停止服务

```bash
# 停止所有服务
docker compose down

# 停止并删除数据卷
docker compose down -v
```

## 详细部署

### 1. 准备环境

```bash
# 安装 Docker（如未安装）
curl -fsSL https://get.docker.com | sh

# 验证安装
docker --version
docker compose version
```

### 2. 模型权重准备（可选）

首次启动时系统会自动下载 YOLO 模型。为避免构建时下载超时，可预先将模型放到 `./models/` 目录：

```bash
mkdir -p models
# 将 yolov8x-pose.pt 放入 models/ 目录
# 或首次构建后由容器自动下载并持久化到该目录
```

### 3. 构建并启动

```bash
# 构建并启动（首次运行需要构建镜像）
docker compose up -d --build

# 查看构建日志
docker compose logs -f backend

# 等待服务健康检查通过
docker compose ps
```

### 4. GPU 加速（可选）

如需使用 NVIDIA GPU 进行 AI 推理加速，请确保已安装 NVIDIA Container Toolkit，并在 `docker-compose.yml` 的 `backend` 服务下添加：

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## 配置说明

### 手势识别参数

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `GESTURE_MIN_DURATION_S` | `2.5` | 手势最小持续时间（秒） |
| `GESTURE_PALM_FACING_RATIO` | `0.60` | 时间窗口内手掌朝前帧的最小占比 |
| `GESTURE_ARM_POSE_RATIO` | `0.50` | 手臂伸直/高举帧的最小占比 |
| `GESTURE_MOTION_PURITY` | `0.65` | 运动方向纯度（主方向/总位移） |
| `GESTURE_MIN_CYCLES` | `2` | 最小方向反转周期数 |
| `GESTURE_CYCLE_MAX_PERIOD_S` | `1.5` | 单个挥动周期的最大允许时长 |
| `GESTURE_STRAIGHT_ARM_ANGLE` | `120` | 自然伸直手臂的最小夹角（度） |

### AI 推理参数

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `YOLO_MODEL` | `yolov8x-pose.pt` | 姿态检测模型 |
| `AI_CONF_THRESHOLD` | `0.35` | 检测置信度阈值 |
| `AI_INFERENCE_IMGSZ` | `896` | 推理输入分辨率 |
| `ENABLE_HAND_DETECTION` | `true` | 是否启用 MediaPipe Hands |

### 视频流参数

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `STREAM_FPS` | `15` | 视频流帧率 |
| `STREAM_WIDTH` | `1280` | 视频宽度 |
| `STREAM_HEIGHT` | `720` | 视频高度 |
| `JPEG_QUALITY` | `88` | JPEG 压缩质量 |

### 网络代理（可选）

> **国内用户提示**：本项目 Dockerfile 中已配置阿里云 PyPI 镜像与 Hugging Face 国内镜像（`HF_ENDPOINT`），正常情况下无需代理即可构建运行。仅在镜像源失效或需要访问 GitHub 下载模型权重时，才需自行配置代理。

如需在容器内使用宿主机代理，可在启动前设置环境变量并修改 `docker-compose.yml`：

```bash
# 宿主机 shell 中设置（示例，请替换为你自己的代理地址）
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
```

然后在 `docker-compose.yml` 的 `backend.environment` 中添加：

```yaml
- HTTP_PROXY=${HTTP_PROXY:-}
- HTTPS_PROXY=${HTTPS_PROXY:-}
```

## 摄像头配置

系统支持通过前端页面配置多路摄像头（前/后/左/右），并提供配置集管理功能。

| 类型 | URL 格式 | 说明 |
|------|----------|------|
| 本地摄像头 | `0`, `1`, `2`... | 系统摄像头编号 |
| RTSP 流 | `rtsp://user:pass@ip:port/stream` | 网络摄像头 |
| RTMP 流 | `rtmp://server/app/stream` | 直播流 |
| HTTP 流 | `http://ip:port/video` | HTTP 视频流 |
| 本地文件 | `/path/to/video.mp4` | 本地视频文件 |

### 配置集 JSON 导入规范

系统支持通过 JSON 编辑器或文件导入方式批量导入摄像头配置集。JSON 格式如下：

```json
{
  "version": "1.0.0",
  "bundles": [
    {
      "id": "default",
      "name": "默认配置集",
      "defaultPullMethod": "webrtc",
      "webrtcPlayerBaseUrl": "https://example.com/webrtc/",
      "webrtcApiBaseUrl": "https://example.com/index/api/webrtc",
      "rtmpBaseUrl": "rtmp://example.com/live",
      "profiles": [
        {
          "id": "profile1",
          "name": "场景一",
          "vehicleCount": 1,
          "cameras": {
            "front": "camera_front_1",
            "back": "camera_back_1",
            "left": "camera_left_1",
            "right": "camera_right_1"
          }
        }
      ]
    }
  ]
}
```

#### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `version` | string | 是 | 配置文档版本号 |
| `bundles` | array | 是 | 配置集数组，至少包含一个元素 |
| `bundles[].id` | string | 是 | 配置集唯一标识 |
| `bundles[].name` | string | 是 | 配置集显示名称 |
| `bundles[].defaultPullMethod` | string | 是 | 默认拉流方式：`webrtc` 或 `rtmp` |
| `bundles[].webrtcPlayerBaseUrl` | string | 否 | WebRTC 播放器基础 URL |
| `bundles[].webrtcApiBaseUrl` | string | 否 | WebRTC API 基础 URL |
| `bundles[].rtmpBaseUrl` | string | 否 | RTMP 基础 URL |
| `bundles[].profiles` | array | 是 | 场景配置数组 |
| `profiles[].id` | string | 是 | 场景唯一标识 |
| `profiles[].name` | string | 是 | 场景显示名称 |
| `profiles[].vehicleCount` | number | 否 | 车辆数量（仅用于显示） |
| `profiles[].cameras` | object | 是 | 四路摄像头名称映射 |
| `cameras.front` | string | 是 | 前视摄像头标识 |
| `cameras.back` | string | 是 | 后视摄像头标识 |
| `cameras.left` | string | 是 | 左视摄像头标识 |
| `cameras.right` | string | 是 | 右视摄像头标识 |

#### 使用方式

1. **文件导入**：点击"导入 JSON"按钮，选择 `.json` 文件
2. **在线编辑**：在 JSON 编辑器中直接修改，点击"应用 JSON"生效
3. **配置集管理**：
   - **删除配置集**：配置集下拉框旁的"删除"按钮（至少保留一个）
   - **删除场景**：场景下拉框旁的"删除"按钮（至少保留一个）
   - **清除全部**：底部"清除全部"按钮，清除所有自定义配置并恢复默认

## API 文档

| 项目 | 地址 |
|------|------|
| API 基础路径 | `http://localhost:8001/api` |
| Swagger UI | `http://localhost:8001/api/docs` |
| 健康检查 | `GET /api/health` |
| WebSocket 视频流 | `WS /ws/video` |

### WebSocket 消息格式

服务端推送的实时消息包含以下字段：

```json
{
  "camera_id": "front",
  "frame": "<base64_jpeg>",
  "direction": "front",
  "confidence": 0.85,
  "detections": [
    {
      "bbox": [100, 200, 300, 400],
      "gesture": "hailing",
      "gesture_conf": 0.82
    }
  ],
  "timestamp": 1703000000.000
}
```

## 手势识别逻辑

### 打招呼（greeting）
- 手掌面朝向画面
- 手臂自然状态
- 手腕**左右挥动**
- 动作持续 **2.5 秒以上**

### 打车（hailing）
- 手掌面朝向画面
- 手臂自然伸直或高举
- 手腕/手臂**上下挥动**
- 动作持续 **2.5 秒以上**

### 举手（hand_up）
- 手臂伸直上举
- 无持续周期性挥动
- 可作为瞬时反馈状态

## 技术栈

### 后端

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.10+ | 编程语言 |
| FastAPI | 0.100+ | Web 框架 |
| Uvicorn | 0.23+ | ASGI 服务器 |
| Ultralytics YOLO | 8.0+ | 姿态检测 |
| MediaPipe | 0.10.8 | 手部关键点检测 |
| OpenCV | 4.8+ | 视频处理 |
| NumPy | 1.24+ | 数值计算 |

### 前端

| 技术 | 版本 | 用途 |
|------|------|------|
| React | 18+ | 前端框架 |
| Vite | 4.4+ | 构建工具 |
| Tailwind CSS | 3.3+ | 样式框架 |
| Lucide React | 0.3+ | 图标库 |

### 基础设施

| 技术 | 版本 | 用途 |
|------|------|------|
| Docker | 20.10+ | 容器化 |
| Docker Compose | 2.0+ | 编排工具 |
| Nginx | 1.25+ | 反向代理 |
| CUDA | 11.8+ | GPU 加速（可选） |

## 目录结构

```
hailuo-car/
├── docker-compose.yml      # Docker Compose 配置
├── nginx.conf              # Nginx 反向代理配置
├── README.md               # 项目说明
├── backend/                # 后端服务
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── entrypoint.sh
│   └── app/                # 后端源码
│       ├── main.py
│       ├── config.py
│       ├── ai/             # AI 推理（姿态/手势/方向）
│       ├── api/            # API 路由与 WebSocket
│       └── stream/         # 视频流处理
├── frontend/               # 前端服务
│   ├── Dockerfile
│   ├── package.json
│   └── src/                # 前端源码
│       ├── components/
│       ├── hooks/
│       └── types/
└── models/                 # 模型权重持久化目录（运行自动生成）
```

## 开发模式

```bash
# 前端开发
cd frontend
npm install
npm run dev

# 后端开发
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 常见问题

### Q1: 构建时模型下载超时？

**A:** 确保宿主机已设置代理，或预先将 `.pt` 模型文件放入 `./models/` 目录：

```bash
export HTTP_PROXY=http://host.docker.internal:7890
export HTTPS_PROXY=http://host.docker.internal:7890
docker compose up -d --build
```

### Q2: MediaPipe Hands 加载失败？

**A:** 本项目固定使用 `mediapipe==0.10.8`。更高版本（如 0.10.33+）移除了 `mp.solutions.hands` API，不兼容。

### Q3: 视频流卡顿或延迟高？

**A:** 可通过以下方式优化：
1. 降低分辨率：`STREAM_WIDTH` / `STREAM_HEIGHT`
2. 降低帧率：`STREAM_FPS`
3. 降低 JPEG 质量：`JPEG_QUALITY`
4. 使用 GPU 加速

### Q4: 如何更新 AI 模型？

**A:** 将新的 `.pt` 模型文件放入 `./models/` 目录，重启后端：

```bash
docker compose restart backend
```

## 端口说明

| 端口 | 服务 | 用途 |
|------|------|------|
| 18080 | Nginx | 统一入口 |
| 8001 | Backend | 后端 API / WebSocket |
| 5173 | Frontend | 前端页面（直接访问） |

## 许可证

MIT License
