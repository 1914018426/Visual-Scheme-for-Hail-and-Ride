# AI Gesture Detection System

基于 YOLO11-Pose 与 MediaPipe Hands 的实时手势识别检测系统，支持"打招呼"与"打车"等手势的精确识别与方向决策。

## 系统特性

| 特性 | 描述 |
|------|------|
| 人体姿态检测 | 基于 YOLO11-Pose，实时检测人体 17 个关键点 |
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
# 将 yolo11x-pose.pt 放入 models/ 目录
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
| `YOLO_MODEL` | `yolo11x-pose.pt` | 姿态检测模型 |
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

## 算法实现详解

本章节深入讲解系统核心识别算法的实现原理，帮助开发者理解从原始视频帧到手势判定结果的完整链路。

### 整体流程

```
视频帧 → YOLO11-Pose 人体检测 → ByteTrack 多目标跟踪
                                          ↓
                    ┌─────────────────────┼─────────────────────┐
                    ↓                     ↓                     ↓
            躯干归一化坐标           MediaPipe Hands        速度/方向计算
            θ1-θ2 角度链              手掌朝向检测              ↓
                    ↓                     ↓              周期性运动检测
                    └───────────┬─────────┘                     ↓
                                ↓                        帧级状态机
                          意图分类（greeting / hailing / hand_up）
                                ↓
                          绘制骨骼 + 轨迹线 → WebSocket 推送
```

### 1. 人体检测：YOLO11-Pose

系统使用 **YOLO11x-Pose**（Ultralytics 8.3+）进行人体姿态检测，输出 COCO 格式的 17 个关键点：

| 索引 | 关键点 | 索引 | 关键点 |
|------|--------|------|--------|
| 0 | 鼻子 | 9 | 左手腕 |
| 5 | 左肩 | 10 | 右手腕 |
| 6 | 右肩 | 11 | 左髋 |
| 7 | 左肘 | 12 | 右髋 |
| 8 | 右肘 | ... | ... |

**关键参数**：
- `conf=0.35`：检测置信度阈值，过滤低质量检测框
- `imgsz=896`：推理输入分辨率，在精度与速度间平衡
- `half=True`：FP16 半精度推理，RTX 4090 上提速约 30%
- `max_det=20`：单帧最大检测人数

> 模型权重首次启动时自动从国内镜像（hf-mirror、腾讯云、上海交大）下载，无需手动准备。

### 2. 多目标跟踪：ByteTrack

检测到人体后，使用 **ByteTrack** 算法为每个人分配稳定的跨帧 track_id。

**为什么不用 YOLO 自带的简单索引？**
- 简单索引 (`person_0`, `person_1`) 每帧重新分配，同一个人在相邻帧可能 ID 不同
- ByteTrack 基于 IoU 和运动预测，能稳定跟踪同一人 30 帧以上（约 2 秒）

**关键配置** (`bytetrack.yaml`)：
```yaml
tracker_type: bytetrack
track_high_thresh: 0.25    # 高置信度阈值
track_low_thresh: 0.1      # 低置信度阈值（用于恢复）
new_track_thresh: 0.25     # 新建 track 的最小置信度
track_buffer: 30           # 丢失后保留的帧数
match_thresh: 0.8          # 匹配阈值
fuse_score: True           # Ultralytics 8.3+ 必需字段
```

> 每个 track_id 对应一个独立的手势状态机，不同 person 之间互不干扰。

### 3. 手势识别核心算法

手势识别是本系统最复杂的模块，位于 `backend/app/ai/gesture.py`。它基于**规则引擎**而非深度学习分类器，核心优势是：
- **可解释性强**：每个判定步骤都有明确的阈值和公式
- **零训练数据**：不需要标注手势数据集
- **响应速度快**：纯 CPU 计算，8-12 帧即可判定（约 0.5-0.8s）

#### 3.1 躯干归一化坐标系

原始像素坐标受摄像头距离、分辨率、人体体型影响。为了统一尺度，系统将坐标转换为**躯干单位（Torso Unit, TU）**：

```
torso_size = (|左肩-左髋| + |左肩-右髋| + |右肩-左髋| + |右肩-右髋|) / 4
```

> 参考文献：Leeds University 2025 年姿态归一化研究

所有速度、距离、振幅都以 TU 为单位，保证同一套阈值适用于 3 米外的行人和 10 米外的行人。

#### 3.2 θ1-θ2 角度链

基于 Tunis 出租车招手研究（MDPI 2023），定义两个关键角度：

| 角度 | 计算公式 | 物理意义 | 典型值 |
|------|----------|----------|--------|
| **θ1** | `∠(髋, 肩, 肘)` | 手臂整体抬起程度 | 高举 > 110°，平伸 75-115° |
| **θ2** | `∠(肩, 肘, 腕)` | 手臂伸直程度 | 伸直 > 140° |

**判定逻辑**：
- `θ2 < 140°` → 手臂弯曲，直接排除（不是招手）
- `θ1 > 110°` → 手臂高举 → 候选 **hailing（打车）**
- `75° < θ1 < 115°` → 手臂平伸 → 候选 **greeting（打招呼）**

#### 3.3 速度计算（归一化）

追踪手腕在相邻帧间的位移，计算速度向量：

```python
vx = (wrist_x - last_wrist_x) / dt / torso_size   # TU/s
vy = (wrist_y - last_wrist_y) / dt / torso_size   # TU/s
v_mag = sqrt(vx² + vy²)
```

- **阈值**：`velocity_threshold = 2.5 TU/s`
- 超过阈值认为"正在运动"，否则认为"静止"

#### 3.4 方向追踪与符号变化

为了区分"水平挥动"和"垂直挥动"，系统追踪速度的历史方向：

```
if |vx| > |vy| * 1.2  →  horizontal（水平）
if |vy| > |vx| * 1.2  →  vertical（垂直）
else                   →  diagonal（对角线）
```

同时统计主方向上的**符号变化次数**（来回摆动）：
- 速度从正变负再变正 = 1 个周期
- 至少需要 `sign_change_min = 2` 次符号变化（即 1 个完整来回）

#### 3.5 周期性运动检测（Zero-Crossing + ACF）

这是系统最精密的模块，用于验证挥手动作是否具有人类特有的周期性（约 1-3 Hz）。

**算法步骤**：

1. **去趋势**：减去线性漂移，消除人体整体移动的影响
2. **Zero-Crossing 检测**：统计序列穿过均值的次数，快速估计频率
3. **自相关函数（ACF）峰值检测**：通过 FFT 加速计算自相关，寻找稳定周期
4. **周期一致性验证**：相邻周期长度的变异系数 CV < 0.5

**判定条件**：
```
amplitude > 25 像素           # 挥动幅度足够大
consistency > 0.45            # 周期稳定
cycle_count >= 2              # 至少 2 个完整周期
0.8 Hz <= frequency <= 3.5 Hz # 人类挥手频率范围
```

> 参考文献：SFU Bruce et al., CRV 2016 — 人体周期性运动检测

#### 3.6 手掌朝向检测（纯 2D 几何）

MediaPipe Hands 提供 21 个手部 landmark，系统用纯 2D 几何判断手掌是否朝向摄像头：

1. **手指展开扇形角**：四指指尖相对于手腕的极坐标角度跨度 > 45°
2. **指尖-指根距离比**：指尖到手腕距离 / 指根到手腕距离 > 1.05（手指伸出）
3. **展开手指数**：至少 2 根手指完全展开
4. **拇指位置**：拇指指尖显著远离拇指根关节

所有判断**完全不依赖 z 坐标**（深度），因为单目摄像头的 z 坐标不可靠。

#### 3.7 帧级状态机

每个 `(track_id, side)` 组合拥有一个独立的状态机，状态流转如下：

```
IDLE ──手臂高举──→ HAND_UP
  │
  └──手臂平伸──→ POSED ──开始挥动──→ OSCILLATING ──周期性确认──→ CONFIRMED
                   ↑                                              │
                   └──────────停止挥动─────────────────────────────┘
```

| 状态 | 含义 | 退出条件 |
|------|------|----------|
| **IDLE** | 手臂自然下垂 | 手臂姿势符合（θ1/θ2 通过） |
| **HAND_UP** | 手臂举起但未挥动 | 检测到来回摆动 |
| **POSED** | 手臂平伸，等待挥动 | 检测到来回摆动 |
| **OSCILLATING** | 正在挥动，等待确认 | 周期性检测通过 → CONFIRMED；停止太久 → IDLE |
| **CONFIRMED** | 手势已确认 | 停止挥动 → IDLE；姿势改变 → 衰减置信度 |

**关键参数**：
- `confirm_frames = 5`：连续挥动 5 帧才进入 CONFIRMED
- `stop_reset_frames = 6`：停止挥动 6 帧后重置
- `ema_alpha = 0.35`：置信度指数移动平均，平滑抖动

#### 3.8 意图分类

当状态机进入 CONFIRMED 时，根据以下规则分类：

| 条件 | 结果 | 说明 |
|------|------|------|
| 垂直挥动为主 (>55%) + 手臂高举 | **hailing** | 打车：高举手臂上下挥 |
| 水平挥动为主 (>55%) + 手臂平伸 | **greeting** | 打招呼：平伸手臂左右挥 |
| 对角线为主 + 手臂高举 | **hailing** | 高举斜向挥也归类为打车 |
| 无明确方向 + 手臂举起 | **hand_up** | 仅举手，无周期性挥动 |

手掌朝向摄像头时，置信度额外 +0.08~0.1。

### 4. MediaPipe Hands 手部关键点增强

YOLO11-Pose 只输出手腕位置（2D 点），无法区分手指状态。系统使用 **MediaPipe Hands** 在检测到的人体 ROI 区域内进一步检测手部 21 点 landmark：

```
0: wrist          5-8: 食指      13-16: 无名指
1-4: 拇指         9-12: 中指     17-20: 小指
```

**使用策略**：
1. 在人体 bbox 基础上扩大 20% 作为 ROI
2. 运行 MediaPipe Hands（21 点检测）
3. 根据 wrist landmark (索引 0) 与 pose left/right wrist (索引 9/10) 的像素距离，将手部匹配到左侧或右侧
4. 匹配距离上限为肩宽的一半（约 30-60 像素），防止误匹配

**用途**：
- 手掌朝向检测（见 3.6 节）
- 前端绘制手部骨骼连线（21 点连线）

### 5. 轨迹绘制与抗抖动

系统在每帧绘制手腕的运动轨迹线，帮助调试和可视化。

**轨迹管理**：
- 按 **摄像头 ID + track_id** 双重隔离，避免跨摄像头混淆
- 每个 track 固定追踪**单侧手腕**（首次出现时选定 left/right，后续不再切换）
- 轨迹长度限制为 15 帧（约 1 秒），减少历史残留
- **运动距离过滤**：如果两帧间手腕跳跃超过 100 像素，认为 track_id 已复用，清空轨迹

**绘制样式**：
| 手势 | 轨迹颜色 | 标记样式 |
|------|----------|----------|
| hailing | 红色 | 大红圈 + 橙内圈 |
| greeting | 青色 | 青圈 + 黄内圈 |
| hand_up | 黄色 | 黄圈 + 红内圈 |
| 无手势 | 浅青色 | 仅轨迹线 |

### 6. 性能优化

| 优化措施 | 实现方式 | 效果 |
|----------|----------|------|
| **FP16 半精度** | `half=True` | RTX 4090 提速 ~30% |
| **ByteTrack 跟踪** | 替代每帧重分配索引 | 减少状态机重置，提升稳定性 |
| **ROI 裁剪** | MediaPipe 只在人体 bbox 内检测 | 减少处理像素数 |
| **推理分辨率** | 896×896（非原始 1920×1080） | 降低计算量，精度损失 < 3% |
| **TensorRT 导出脚本** | `export_tensorrt.py` | 未来可进一步提速 2-3x |

## 技术栈

### 后端

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.10+ | 编程语言 |
| FastAPI | 0.100+ | Web 框架 |
| Uvicorn | 0.23+ | ASGI 服务器 |
| Ultralytics YOLO | 8.3+ | 姿态检测（YOLO11-Pose） |
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
