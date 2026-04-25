# Hailuo Car - AI车载智能检测系统

## 项目简介

Hailuo Car 是一款基于 AI 的**车载智能检测系统**，集成**车辆检测**与**手势识别**两大核心功能。系统采用前后端分离架构，通过 Docker 容器化部署，实现一键启动，快速构建车载 AI 检测能力。

### 系统架构图

```
                                用户浏览器
                                    |
                                    v
                              +------------+
                              |   Nginx    |  <-- 统一入口 (端口 8080)
                              |  (8080)    |
                              +----+-------+
                                   |
                    +--------------+---------------+
                    |                              |
                    v                              v
            +--------------+              +--------------+
            |   Frontend   |              |   Backend    |
            |   (Vue3)     |              |  (FastAPI)   |
            |   (80)       |              |  (8000)      |
            +--------------+              +------+-------+
                                                 |
                                    +------------+-----------+
                                    |            |           |
                                    v            v           v
                              +---------+  +---------+  +---------+
                              |  车辆   |  |  手势   |  |  视频  |
                              |  检测   |  |  识别   |  |  流处理 |
                              |  (YOLO) |  |  (YOLO) |  |        |
                              +---------+  +---------+  +---------+
                                    |            |           |
                                    v            v           v
                              +---------+  +---------+  +---------+
                              |  车载   |  |  驾驶   |  |  RTSP/  |
                              |  摄像头 |  |  摄像头  |  |  HTTP   |
                              +---------+  +---------+  +---------+
```

### 系统特性

| 特性 | 描述 |
|------|------|
| 车辆检测 | 基于 YOLO 模型，实时检测车辆、行人、交通标志 |
| 手势识别 | 支持多种驾驶手势识别，如转向、停车、加速等 |
| 实时视频流 | MJPEG 实时推流，低延迟高帧率 |
| WebSocket | 支持双向实时通信，状态即时同步 |
| 多路摄像头 | 支持本地摄像头、RTSP/RTMP/HTTP 网络视频流 |
| Docker 部署 | 容器化一键部署，环境隔离，易于维护 |

---

## 快速开始

### 前置要求

- Docker >= 20.10
- Docker Compose >= 1.29
- NVIDIA GPU（推荐，用于 AI 推理加速）
- 至少 4GB 可用内存

### 一键启动

```bash
# 1. 克隆项目
git clone <项目仓库地址>
cd hailuo-car

# 2. 启动所有服务（后台模式）
docker compose up -d

# 3. 查看服务状态
docker compose ps

# 4. 访问系统
# Web 界面: http://localhost:8080
# API 文档: http://localhost:8080/api/docs
# 直接后端: http://localhost:8000
```

### 停止服务

```bash
# 停止所有服务
docker compose down

# 停止并删除数据卷
docker compose down -v

# 停止并删除镜像（完全清理）
docker compose down --rmi all -v
```

---

## 详细部署步骤

### 方法一：Docker 部署（推荐）

#### 1. 准备环境

```bash
# 安装 Docker（如未安装）
curl -fsSL https://get.docker.com | sh

# 验证安装
docker --version
docker compose version
```

#### 2. 配置项目

```bash
# 进入项目目录
cd hailuo-car

# 确认目录结构
.
├── docker-compose.yml      # Docker Compose 配置
├── nginx.conf              # Nginx 反向代理配置
├── .dockerignore           # Docker 构建忽略文件
├── backend/                # 后端服务目录
│   ├── Dockerfile          # 后端镜像构建文件
│   └── ...                 # 后端源码
└── frontend/               # 前端服务目录
    ├── Dockerfile          # 前端镜像构建文件
    └── ...                 # 前端源码
```

#### 3. 启动服务

```bash
# 构建并启动（首次运行需要构建镜像）
docker compose up -d --build

# 查看构建日志
docker compose logs -f

# 等待所有服务健康检查通过
docker compose ps
```

#### 4. 验证部署

```bash
# 测试健康检查
curl http://localhost:8080/health

# 测试 API
curl http://localhost:8080/api/health

# 查看后端日志
docker logs hailuo-backend -f
```

### 方法二：GPU 加速部署

如需使用 NVIDIA GPU 进行 AI 推理加速，请确保已安装 NVIDIA Container Toolkit：

```bash
# 安装 nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 修改 docker-compose.yml 添加 GPU 支持
# 在 backend 服务下增加:
#    deploy:
#      resources:
#        reservations:
#          devices:
#            - driver: nvidia
#              count: all
#              capabilities: [gpu]
```

### 方法三：开发模式部署

```bash
# 前端开发模式（热重载）
cd frontend && npm install && npm run dev

# 后端开发模式（热重载）
cd backend && pip install -r requirements.txt && uvicorn main:app --reload --port 8000

# Nginx 开发模式
docker run -p 8080:80 -v $(pwd)/nginx.conf:/etc/nginx/nginx.conf:ro nginx:alpine
```

---

## 摄像头配置说明

系统支持多种视频源接入方式，通过环境变量或 API 参数进行配置。

### 支持的摄像头类型

| 类型 | URL 格式 | 说明 |
|------|----------|------|
| 本地摄像头 | `0`, `1`, `2`... | 系统默认摄像头编号 |
| USB 摄像头 | `/dev/video0` | Linux 设备路径 |
| RTSP 摄像头 | `rtsp://user:pass@ip:port/stream` | 网络摄像头 |
| RTMP 流 | `rtmp://server/app/stream` | 直播流 |
| HTTP 流 | `http://ip:port/video` | HTTP 视频流 |
| HLS 流 | `http://ip:port/stream.m3u8` | HLS 直播流 |
| 本地视频 | `file:///path/to/video.mp4` | 本地视频文件 |

### 配置示例

#### 本地摄像头

```bash
# 使用系统默认摄像头（编号 0）
curl -X POST http://localhost:8080/api/stream/start \
  -H "Content-Type: application/json" \
  -d '{"source": "0", "fps": 15}'
```

#### RTSP 网络摄像头

```bash
# 海康威视摄像头
curl -X POST http://localhost:8080/api/stream/start \
  -H "Content-Type: application/json" \
  -d '{"source": "rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101"}'

# 大华摄像头
curl -X POST http://localhost:8080/api/stream/start \
  -H "Content-Type: application/json" \
  -d '{"source": "rtsp://admin:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"}'
```

#### HTTP 视频流

```bash
# IP 摄像头 HTTP 流
curl -X POST http://localhost:8080/api/stream/start \
  -H "Content-Type: application/json" \
  -d '{"source": "http://192.168.1.100:8080/video"}'
```

### 环境变量配置

在 `docker-compose.yml` 中可通过环境变量调整视频流参数：

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `STREAM_FPS` | 15 | 视频流帧率（FPS） |
| `STREAM_WIDTH` | 640 | 视频流宽度（像素） |
| `STREAM_HEIGHT` | 480 | 视频流高度（像素） |
| `JPEG_QUALITY` | 85 | JPEG 压缩质量（1-100） |
| `DETECTION_CONFIDENCE` | 0.5 | 车辆检测置信度阈值 |
| `GESTURE_CONFIDENCE` | 0.7 | 手势识别置信度阈值 |

---

## API 文档

### 基础信息

| 项目 | 地址 |
|------|------|
| API 基础路径 | `http://localhost:8080/api` |
| Swagger UI | `http://localhost:8080/api/docs` |
| ReDoc | `http://localhost:8080/api/redoc` |

### 核心 API 列表

#### 1. 健康检查

```http
GET /api/health
```

**响应示例：**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T08:30:00Z",
  "version": "1.0.0"
}
```

#### 2. 启动视频流

```http
POST /api/stream/start
Content-Type: application/json

{
  "source": "0",
  "fps": 15,
  "width": 640,
  "height": 480
}
```

**响应示例：**
```json
{
  "stream_id": "stream_001",
  "status": "started",
  "url": "/api/stream/stream_001"
}
```

#### 3. 停止视频流

```http
POST /api/stream/stop
Content-Type: application/json

{
  "stream_id": "stream_001"
}
```

#### 4. 获取 MJPEG 视频流

```http
GET /api/stream/{stream_id}
```

返回 `multipart/x-mixed-replace` 格式的 MJPEG 流。

#### 5. 获取检测状态

```http
GET /api/detection/status
```

**响应示例：**
```json
{
  "is_running": true,
  "stream_id": "stream_001",
  "detection_types": ["vehicle", "gesture"],
  "fps": 15,
  "resolution": "640x480"
}
```

#### 6. 车辆检测结果

```http
GET /api/detection/vehicles
```

**响应示例：**
```json
{
  "count": 3,
  "vehicles": [
    {
      "id": 1,
      "class": "car",
      "confidence": 0.92,
      "bbox": [100, 200, 300, 400]
    }
  ]
}
```

#### 7. 手势识别结果

```http
GET /api/detection/gestures
```

**响应示例：**
```json
{
  "gesture": "turn_left",
  "confidence": 0.85,
  "hand_position": [320, 240]
}
```

#### 8. WebSocket 实时通信

```
WS /ws/stream
```

连接 WebSocket 后，服务端实时推送检测结果：
```json
{
  "type": "detection",
  "timestamp": "2024-01-15T08:30:00Z",
  "vehicles": [...],
  "gestures": [...],
  "fps": 15
}
```

---

## 技术栈

### 后端技术

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.10+ | 编程语言 |
| FastAPI | 0.100+ | Web 框架 |
| Uvicorn | 0.23+ | ASGI 服务器 |
| OpenCV | 4.8+ | 视频处理 |
| Ultralytics YOLO | 8.0+ | AI 检测模型 |
| NumPy | 1.24+ | 数值计算 |
| WebSockets | 11.0+ | 实时通信 |

### 前端技术

| 技术 | 版本 | 用途 |
|------|------|------|
| Vue.js | 3.3+ | 前端框架 |
| Vite | 4.4+ | 构建工具 |
| Element Plus | 2.3+ | UI 组件库 |
| ECharts | 5.4+ | 数据可视化 |
| Axios | 1.5+ | HTTP 客户端 |

### 基础设施

| 技术 | 版本 | 用途 |
|------|------|------|
| Docker | 20.10+ | 容器化 |
| Docker Compose | 1.29+ | 编排工具 |
| Nginx | 1.25+ (alpine) | 反向代理 |
| CUDA | 11.8+ | GPU 加速（可选） |

---

## 开发指南

### 目录结构

```
hailuo-car/
├── docker-compose.yml          # Docker Compose 主配置
├── nginx.conf                   # Nginx 反向代理配置
├── .dockerignore                # Docker 构建忽略规则
├── backend/                     # 后端服务
│   ├── Dockerfile               # 后端镜像构建
│   ├── requirements.txt         # Python 依赖
│   ├── main.py                  # FastAPI 入口
│   ├── models/                  # AI 模型目录
│   ├── routers/                 # API 路由
│   ├── services/                # 业务逻辑
│   └── utils/                   # 工具函数
└── frontend/                    # 前端服务
    ├── Dockerfile               # 前端镜像构建
    ├── package.json             # Node.js 依赖
    ├── vite.config.ts           # Vite 配置
    ├── src/                     # 源码目录
    │   ├── components/          # 组件
    │   ├── views/               # 页面
    │   ├── api/                 # API 接口
    │   └── stores/              # 状态管理
    └── public/                  # 静态资源
```

### 本地开发流程

```bash
# 1. 启动后端（终端 1）
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 2. 启动前端（终端 2）
cd frontend
npm install
npm run dev

# 3. 访问开发服务器
# 前端: http://localhost:5173
# 后端: http://localhost:8000
# API 文档: http://localhost:8000/docs
```

### 构建镜像

```bash
# 构建后端镜像
docker build -t hailuo-backend:latest ./backend

# 构建前端镜像
docker build -t hailuo-frontend:latest ./frontend

# 构建并启动全部服务
docker compose up -d --build
```

### 调试技巧

```bash
# 查看容器日志
docker logs hailuo-backend -f --tail 100
docker logs hailuo-frontend -f --tail 100
docker logs hailuo-nginx -f --tail 100

# 进入容器调试
docker exec -it hailuo-backend /bin/sh
docker exec -it hailuo-backend python -c "import cv2; print(cv2.__version__)"

# 检查网络连接
docker exec -it hailuo-nginx ping backend
docker exec -it hailuo-nginx curl http://backend:8000/api/health
```

---

## 常见问题 FAQ

### Q1: docker compose up 启动失败怎么办？

**A:** 请按以下步骤排查：

```bash
# 1. 检查 Docker 服务状态
docker info

# 2. 确认 docker-compose.yml 语法
docker compose config

# 3. 单独构建检查
docker compose build --no-cache

# 4. 查看详细日志
docker compose logs

# 5. 检查端口占用
sudo lsof -i :8000
sudo lsof -i :8080
sudo lsof -i :5173
```

常见原因：端口冲突、镜像构建失败、内存不足。

### Q2: 如何修改默认端口？

**A:** 编辑 `docker-compose.yml` 文件：

```yaml
services:
  backend:
    ports:
      - "自定义端口:8000"    # 修改这里

  nginx:
    ports:
      - "自定义端口:80"      # 修改这里
```

修改后需同步更新 `nginx.conf` 中的代理配置。

### Q3: 如何更新 AI 模型？

**A:** 将新的 `.pt` 模型文件放入挂载的 volumes 目录：

```bash
# 模型文件位于 Docker Volume 中
docker volume ls | grep hailuo_models

# 复制新模型到 volume
docker cp new_model.pt hailuo-backend:/app/models/

# 或重启容器重新加载
docker compose restart backend
```

### Q4: 视频流卡顿或延迟高？

**A:** 可通过以下方式优化：

1. **降低分辨率**：将 `STREAM_WIDTH` 和 `STREAM_HEIGHT` 设为更小的值
2. **降低帧率**：将 `STREAM_FPS` 设为 10 或更低
3. **降低 JPEG 质量**：将 `JPEG_QUALITY` 设为 70-80
4. **使用 GPU 加速**：参考 GPU 部署章节
5. **网络优化**：确保摄像头与服务器在同一局域网

### Q5: WebSocket 连接断开？

**A:** Nginx 已配置长连接（`proxy_read_timeout 86400s`），如遇断开：

1. 检查 Nginx 日志：`docker logs hailuo-nginx`
2. 确认后端服务正常：`curl http://localhost:8000/api/health`
3. 检查防火墙是否拦截 WebSocket 连接
4. 前端添加自动重连机制

### Q6: 如何查看系统资源使用情况？

**A:**

```bash
# 查看容器资源使用
docker stats

# 查看后端服务状态
docker compose ps

# 查看系统整体负载
docker exec hailuo-backend top
```

### Q7: 如何备份和恢复数据？

**A:**

```bash
# 备份模型数据卷
docker run --rm -v hailuo-car_hailuo_models:/source -v $(pwd)/backup:/backup alpine \
  tar czf /backup/models_backup.tar.gz -C /source .

# 恢复模型数据卷
docker run --rm -v hailuo-car_hailuo_models:/target -v $(pwd)/backup:/backup alpine \
  tar xzf /backup/models_backup.tar.gz -C /target
```

### Q8: 如何添加自定义检测类别？

**A:** 需要训练自定义 YOLO 模型并替换：

1. 准备标注数据集
2. 使用 Ultralytics 训练：`yolo detect train data=custom.yaml model=yolov8n.pt`
3. 导出模型：`yolo export model=best.pt format=onnx`
4. 替换 `/app/models/` 下的模型文件
5. 重启后端服务：`docker compose restart backend`

---

## 端口说明

| 端口 | 服务 | 用途 | 访问方式 |
|------|------|------|----------|
| 8080 | Nginx | 统一入口 | http://localhost:8080 |
| 8000 | Backend | 后端 API | http://localhost:8000 |
| 5173 | Frontend | 前端页面 | http://localhost:5173（开发） |

---

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

## 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。
