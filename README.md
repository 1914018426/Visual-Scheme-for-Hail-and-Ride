# Hailuo Vision — 视觉招手即停系统

基于 **YOLO11s-Pose** + **ByteTrack** + **TemporalKeypointTransformer** + **Simple 规则后验过滤** 的实时手势识别系统，专为智能网约车/无人车的"招手即停"场景设计，支持车辆运动状态下对路边乘客招手动作的精确识别。系统内置完整的 **DataLab 数据实验室**，提供录制管理、消融实验、统计分析和可视化报告导出能力。

---

## 目录

1. [系统特性](#系统特性)
2. [系统架构](#系统架构)
3. [快速开始](#快速开始)
4. [详细部署指南](#详细部署指南)
5. [算法详解](#算法详解)
6. [DataLab 数据实验室](#datalab-数据实验室)
7. [Transformer 训练详解](#transformer-训练详解)
8. [配置说明](#配置说明)
9. [引擎模式对比](#引擎模式对比)
10. [API 接口文档](#api-接口文档)
11. [开发模式](#开发模式)
12. [故障排查](#故障排查)
13. [项目结构](#项目结构)

---

## 系统特性

| 特性 | 描述 |
|------|------|
| 人体姿态检测 | YOLO11s-Pose，640x640 输入，GPU 半精度推理 (~10-15ms/帧) |
| 多目标跟踪 | ByteTrack 跨帧关联，支持最多 20 人同时检测 |
| Transformer 时序识别 | 12 维 TNLF 特征 x 45 帧滑窗，val_f1=0.897 |
| Simple 规则后验过滤 | 面部可见 + 手腕高于手肘 + 周期性运动，过滤挠头等假阳性 |
| Torso-Normalized Local Frame | 以躯干为参考系消除车辆移动导致的伪运动 |
| 前臂方向向量 (FDV) | 基于 wrist-elbow 向量描述手臂伸出方向，轻量级朝向锁 |
| 自适应推流 | 根据推理负载动态调节分辨率与 JPEG 质量，维持实时性 |
| 实时日志 | WebSocket 实时日志推送，前端内置日志面板 |
| 多路视频流 | RTSP / RTMP / HTTP / 本地摄像头 / 本地文件 |
| DataLab 实验平台 | 录制控制、消融实验、统计分析、SVG/PNG 图表导出 |
| Docker 一键部署 | 容器化编排，NVIDIA GPU 直通 |

---

## 系统架构

### 整体架构图

```
+------------------+     +------------------+     +------------------+
|   视频源层        |     |   前端展示层      |     |   实验管理层      |
|  RTSP/RTMP/HTTP  |     |  React + Vite    |     |  DataLab Page    |
|  本地摄像头/文件  |     |  Tailwind CSS    |     |  录制/实验/报告   |
+------------------+     +------------------+     +------------------+
         |                       |                       |
         v                       v                       v
+------------------+     +------------------+     +------------------+
|   视频流处理层    |<--->|   Nginx 代理层    |<--->|   FastAPI 后端   |
|  StreamHandler   |     |  端口 18080       |     |  REST + WebSocket|
|  采集/缓冲/重连   |     |  静态资源/反向代理 |     |  /api /ws /logs  |
+------------------+     +------------------+     +------------------+
                             |                           |
                             v                           v
                    +------------------+     +----------------------+
                    |   前端开发服务器  |     |   AI 推理流水线       |
                    |    端口 5173     |     |  PoseDetector        |
                    +------------------+     |  - YOLO11-Pose       |
                                             |  - ByteTrack         |
                                             |  - MediaPipe Hands   |
                                             +----------------------+
                                                       |
                                                       v
                                             +----------------------+
                                             |   手势识别引擎层       |
                                             |SimpleTransformerHybrid|
                                             |  - Simple 规则引擎     |
                                             |  - Transformer 时序    |
                                             |  - TripleLock ( legacy)|
                                             +----------------------+
                                                       |
                                                       v
                                             +----------------------+
                                             |   DataLab 后端        |
                                             |  - GestureRecorder    |
                                             |  - AblationRunner     |
                                             |  - AblationAnalyzer   |
                                             |  - ChartGenerator     |
                                             +----------------------+
```

### 数据流架构

```
Frame Capture (OpenCV + FFmpeg)
    |
    v
YOLO11-Pose Inference (GPU, fp16)
    |
    v
ByteTrack Multi-Object Tracking
    |
    v
TNLF Feature Extraction (per person)
    |
    +---> MediaPipe Hands (optional, CPU)
    |
    v
Gesture Engine Pipeline
    |
    +---> Simple Engine: face visible + wrist>elbow + periodic
    +---> Transformer Engine: 45-frame TNLF window -> sigmoid
    +---> Hybrid Fusion
    |
    v
Visualization + WebSocket Push
    |
    +---> MJPEG Stream (adaptive quality)
    +---> Detection JSON Overlay
    +---> Log Broadcast
    +---> DataLab Recording (optional)
```

---

## 快速开始

### 前置要求

- Docker >= 20.10
- Docker Compose >= 2.0
- NVIDIA GPU + NVIDIA Container Toolkit（CUDA 12.x 兼容）
- 至少 4GB 可用显存
- 8GB 系统内存（推荐 16GB）

### 1. 克隆项目

```bash
git clone https://github.com/1914018426/Hailuo-Vision.git
cd Hailuo-Vision
```

### 2. 准备模型

首次启动时会自动下载 YOLO 模型。如需离线部署，预先将模型放入 `./models/`：

```bash
mkdir -p models
# 下载 yolo11s-pose（约 16MB）
wget -O models/yolo11s-pose.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt
```

Transformer 模型位于 `./models/transformer/waving_transformer_real.pt`，已包含在仓库中。

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
| 后端 API | http://localhost:18080/api/docs | Swagger/OpenAPI 文档 |
| 后端直连 | http://localhost:8001 | FastAPI 服务 |
| 前端直连 | http://localhost:5173 | React 开发服务器 |

### 停止服务

```bash
docker compose down
```

---

## 详细部署指南

### 环境准备

#### NVIDIA Container Toolkit 安装

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 验证
nvidia-smi
```

#### Docker Buildx 缓存配置（可选，加速重建）

```bash
# 启用 buildx
docker buildx create --use --name hailuo-builder
docker buildx inspect --bootstrap

# 使用缓存构建
docker compose build --build-arg BUILDKIT_INLINE_CACHE=1
```

### 模型预下载（离线部署）

```bash
mkdir -p models/transformer

# YOLO 姿态检测模型
wget -O models/yolo11s-pose.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt

# 或更高精度版本（推理更慢）
# wget -O models/yolo11x-pose.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt

# Transformer 模型（已包含在仓库）
# models/transformer/waving_transformer_real.pt
```

### 多摄像头配置

系统支持最多 4 路摄像头同时接入，通过环境变量配置：

```yaml
environment:
  - CAMERA_FRONT=rtsp://192.168.1.101:554/stream1
  - CAMERA_BACK=rtsp://192.168.1.102:554/stream2
  - CAMERA_LEFT=0                    # 本地摄像头设备ID
  - CAMERA_RIGHT=/data/record.mp4    # 本地视频文件
```

摄像头协议支持详情：

| 协议 | 格式示例 | 说明 |
|------|---------|------|
| RTSP | `rtsp://host:554/path` | 推荐使用 TCP 传输 |
| RTMP | `rtmp://host/live/stream` | 直播流协议 |
| HTTP | `http://host/video.mp4` | HTTP 视频流 |
| 本地设备 | `0`, `1`, `2` | 数字字符串表示设备ID |
| 本地文件 | `/path/to/video.mp4` | 支持 AVI/MP4/MKV |

### 网络流低延迟优化

对于 RTSP/RTMP 网络流，系统已内置 FFmpeg 低延迟参数：

```
rtsp_transport;tcp|buffer_size;1024|max_delay;100000|fflags;nobuffer|flags;low_delay
```

如需自定义，修改 `docker-compose.yml` 中的 `OPENCV_FFMPEG_CAPTURE_OPTIONS`。

### 数据持久化

```yaml
volumes:
  - ./models:/app/models:rw          # 模型文件
  - ./.ultralytics:/app/.ultralytics:rw  # Ultralytics 配置缓存
  - ./datasets:/app/datasets:ro      # 数据集（训练用）
  - ./data:/app/data:rw              # DataLab 录制数据与实验结果
```

### 国内网络环境适配

已在 `docker-compose.yml` 中内置：

```yaml
environment:
  - HF_ENDPOINT=https://hf-mirror.com    # HuggingFace 国内镜像
```

PyPI 镜像在 `backend/Dockerfile` 中配置为阿里云镜像。

---

## 算法详解

### 1. 人体姿态检测 — YOLO11s-Pose + ByteTrack

#### YOLO11-Pose 模型

- **模型**：`yolo11s-pose.pt`（~16MB，速度优先）或 `yolo11x-pose.pt`（~112MB，精度优先）
- **输入**：640x640，fp16 半精度
- **置信度阈值**：`0.35`
- **最大检测数**：`20` 人
- **推理时延**：~10-15ms/帧（RTX 4090，CUDA 12）

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

#### ByteTrack 多目标跟踪

ByteTrack 通过关联高分检测框和低分检测框，有效减少遮挡导致的 ID 切换。配置参数位于 `backend/app/ai/bytetrack.yaml`：

```yaml
track_thresh: 0.5      # 高分阈值
track_buffer: 30       # 丢失跟踪的最大帧数
match_thresh: 0.8      # 匹配阈值
min_box_area: 10       # 最小边界框面积
```

当 ByteTrack 临时丢失 ID 时，系统通过空间最近邻匹配上一帧中心（阈值：max(80px, 0.35 * 上一帧对角线)），避免退化为帧内索引导致轨迹跳变。

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

**关键设计**：所有空间特征均使用 `torso_units`（躯干长度归一化），因此同一套模型参数适用于近处大人和远处小人。

**EMA 平滑**：对 origin、e_x、e_y、torso_scale、wrist_local 做 EMA 平滑（alpha=0.5），消除 YOLO 关键点帧间抖动。

### 3. Transformer 时序识别器

基于 **TemporalKeypointTransformer**，输入为 45 帧 x 12 维 TNLF 特征：

#### 模型架构

```
Input: [B, T, F]  T=45 frames, F=12 features
  |
  v
Input Projection (Linear: 12 -> d_model)
  |
  v
CLS Token + Positional Encoding (Sinusoidal)
  |
  v
Transformer Encoder x n_layers (Pre-LN, GELU, Multi-Head Self-Attention)
  |
  v
CLS Token Output
  |
  v
Classification Head: LayerNorm -> Linear(64) -> GELU -> Dropout -> Linear(1)
  |
  v
Output: [B, 1] sigmoid confidence
```

**超参数**（`waving_transformer_real.pt`）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `d_model` | 64 | 隐藏维度 |
| `n_head` | 4 | 注意力头数 |
| `n_layers` | 2 | Encoder 层数 |
| `dim_feedforward` | 256 | FFN 中间维度 |
| `dropout` | 0.1 | Dropout 率 |
| `seq_len` | 45 | 输入帧数（3秒 @ 15fps） |
| `n_features` | 12 | 每帧特征维度 |

**初始化策略**：
- Xavier Uniform 初始化所有权重
- CLS token 用 `std=0.02` 的正态分布初始化
- 最后一层分类器 bias 零初始化，使模型初始输出 `p=0.5`

#### 输入特征（12 维 TNLF）

| 索引 | 特征 | 说明 |
|------|------|------|
| 0 | `wlx_l` | 左手腕 TNLF x 坐标 |
| 1 | `wly_l` | 左手腕 TNLF y 坐标 |
| 2 | `wlx_r` | 右手腕 TNLF x 坐标 |
| 3 | `wly_r` | 右手腕 TNLF y 坐标 |
| 4 | `vel_mag` | 活跃手臂速度幅值（torso_units/s） |
| 5 | `theta1` | 肩-肘-髋夹角（手臂抬起度，度） |
| 6 | `theta2` | 肩-肘-腕夹角（前臂伸直度，度） |
| 7 | `ext_ratio` | 手臂伸展比例 |shoulder-wrist| / (|SE|+|EW|) |
| 8 | `pn_x` | 前臂方向向量 x |
| 9 | `pn_y` | 前臂方向向量 y |
| 10 | `pn_z` | 前臂方向向量 z（固定 0.5） |
| 11 | `valid` | TNLF 计算有效性标志（0/1） |

#### 推理流程

1. 维护每 track_id/side 一个 45 帧滑窗
2. 每帧提取 12 维 TNLF 特征
3. 窗口填满后输入 Transformer
4. 输出 sigmoid 置信度
5. EMA 平滑（alpha=0.35）+ hold 机制（15帧）
6. 面部硬过滤（鼻子置信度 < 0.25 直接丢弃）

### 4. Simple 规则后验过滤

Simple 引擎是三重条件规则引擎，用于过滤 Transformer 的假阳性：

#### 条件 1：面部可见

- 鼻子置信度 >= 0.25
- 至少一只眼睛可见（左眼或右眼置信度 >= 0.25）

#### 条件 2：手腕高于手肘

- 至少一侧手腕在图像坐标中高于手肘（wrist_y < elbow_y）
- 肘部和腕部关键点置信度均 >= 0.3

#### 条件 3：周期性运动

使用 wrist_local 轨迹（或 wrist-elbow 向量）进行周期性检测：

1. **3-5帧滑动平均**：过滤车辆悬挂高频颠簸
2. **去趋势**：线性回归减趋势
3. **Zero-crossing 计数**：检测符号变化
4. **频率估算**：cycles / duration，需在 [0.35, 3.0] Hz 范围内
5. **振幅检查**：振幅需超过阈值（torso_units）
6. **周期一致性**：std/mean < 0.65

#### 混合逻辑（Simple-Transformer）

| Transformer | Simple 周期检测 | 结果 |
|-------------|-----------------|------|
| NONE | — | NONE |
| WAVING (conf <= 0.7) | 未通过 | NONE |
| WAVING (conf > 0.7) | 未通过 | WAVING（高置信度绕过） |
| WAVING | 通过 | WAVING（取 Transformer 置信度） |

### 5. 前臂方向向量（Forearm Direction Vector, FDV）

当 `ENABLE_HAND_DETECTION=false`（默认）时，系统不运行 MediaPipe Hands，而是直接从 YOLO-Pose 输出的肩/肘/腕关键点计算一个**前臂方向向量**，作为朝向锁的输入。

**定义**：

```python
forearm = wrist - elbow                          # 前臂在图像平面的 2D 方向
FDV     = [normalize(forearm).x,
           normalize(forearm).y,
           0.5]                                   # Z 分量固定为正
```

**FDV 不是手掌法向量**。在解剖学上，自然状态下手掌平面通常与前臂在同一平面（或仅有小幅夹角），真实的手掌法向量（垂直于手掌平面）通常**垂直于**前臂方向。FDV 与手掌法向量在几何上不等价，不可混为一谈。

**FDV 的工程作用**：

FDV 描述的是**"手臂朝哪个方向伸出"**，用于区分两类姿态：

| 场景 | 手臂姿态 | FDV 特征 | 朝向锁结果 |
|------|----------|----------|-----------|
| **招手** | 手臂朝前上方伸出，前臂大致指向车头 | XY 分量小，Z=0.5 主导 | 夹角小，通过 |
| **走路摆臂** | 手臂在身体两侧侧向摆动 | XY 分量大 | 夹角大，拒绝 |
| **手臂下垂** | 前臂朝下 | XY 分量小但 theta1 低 | 已被姿态锁拒绝 |

**为什么不用 MediaPipe？**

MediaPipe Hands 能从 21 个手部关键点计算出真实的 3D 手掌法向量，解剖学上更精确。但代价是：
- 每帧需独立运行两次（左右手各一次）
- CPU 密集，引发 GIL 争抢
- 在 15fps 多路推流下，帧率下降 4-8 倍

FDV 是一个**轻量级折衷**：它牺牲了手掌朝向的精确度，换取了极低的计算开销，且对"手臂朝前伸出"这一核心招手特征仍具有足够的区分能力。Transformer 模型在训练时也是用同样的 FDV 特征，它学到的是"前臂朝前上方伸出"这一模式，而非真正的"掌心朝车"。

**升级路径**：若对掌心朝向有更高精度要求，可设置 `ENABLE_HAND_DETECTION=true` 重新启用 MediaPipe Hands，或引入深度相机。

### 6. 三锁合取引擎（TripleLock，legacy）

早期规则引擎，已被 Transformer 替代，但仍保留作为 fallback：

| 锁 | 判定条件 | 最小持续 | 释放条件 |
|----|---------|---------|---------|
| 姿态锁 | theta1 > 25deg 且 theta2 > 15deg 且 ext>0.1 | 3帧 | 任一条件不满足 |
| 朝向锁 | 法向量与 Z 轴夹角 < 55deg | 5帧 | 夹角 > 60deg |
| 运动锁 | FFT 主导频 0.35~3Hz 且幅度 > 0.1 TU | 3帧 | 周期消失或速度<0.05 |

三锁同时满足 -> confirmed_hailing，保持 15 帧后释放。

### 7. 车辆振动补偿

#### 躯干关键点 EMA 平滑

对肩/髋四个关键点做 EMA 平滑（alpha=0.35），消除平台抖动对 TNLF 计算的影响：

```python
TORSO_INDICES = [5, 6, 11, 12]  # L/R shoulder, L/R hip
smoothed_kpt = alpha * raw_kpt + (1 - alpha) * prev_smoothed_kpt
```

#### 静止状态过滤

当躯干中心点历史位移均值 < 0.8 px/帧 时，判定为静止目标，降低误检灵敏度。

#### 原始手腕像素范围过滤

即使 TNLF 显示周期性运动，若原始手腕像素坐标的 x/y 范围均 < 15px，强制判为静止（过滤小幅抖动）。

### 8. 自适应推流

后端根据单帧"推理+编码"耗时动态调节：

```python
if latency > WS_FRAME_BUDGET_MS:       # 超时（默认 60ms）
    short_side -= 12px                  # 降低分辨率
    jpeg_quality -= 2                   # 降低质量
elif latency < WS_FRAME_BUDGET_MS * 0.38:  # 余量充足
    short_side += 8px
    jpeg_quality += 1
```

分辨率范围：`ADAPTIVE_MIN_SHORT_SIDE` ~ `ADAPTIVE_MAX_SHORT_SIDE`（默认 320~480px）。

**设计原理**：
- 4 路 720p MJPEG 推流约 20MB/s，React 渲染主线程出现 1s+ 卡顿
- 降至 480p 短边后，单帧约 80KB，浏览器可稳定 60Hz 重绘
- JPEG 质量 65 在视觉差异上接近 75，但数据量约 30% 更小

---

## DataLab 数据实验室

DataLab 是内置的实验与评估平台，支持录制管理、消融实验、统计分析和报告导出。

### 核心功能

| 功能 | 说明 |
|------|------|
| 录制控制 | 手动/自动手势触发/自动连续 三种录制模式 |
| 视频导入 | 导入本地视频文件（如 HMDB51/UCF101 片段）作为实验素材 |
| 引擎对比实验 | 多引擎（Simple/Transformer/Hybrid 等）逐帧对比 |
| 组件消融实验 | 逐一移除组件，评估各模块贡献 |
| 阈值扫描 | 扫描置信度阈值，生成 PR/ROC 曲线 |
| 场景分析 | 按速度/距离/左右手分组统计 |
| 全量实验套件 | 自动运行正样本+负样本的完整实验组合 |
| 统计分析 | Precision/Recall/F1、一致率矩阵、时序一致性、校准度 |
| 图表导出 | SVG 条形图/折线图/热力图/雷达图/分组图，支持转 PNG |

### 数据模型

```
RecordingSession          # 录制会话
  -> FrameSnapshot[]      # 逐帧快照（关键点/TNLF/手部）
  -> AblationExperiment   # 关联实验
    -> EngineFrameResult[]  # 逐帧多引擎结果（JSONL）
    -> AnalysisReport       # 分析报告
      -> EngineStats[]      # 单引擎统计
      -> AgreementMatrix    # 一致率矩阵
      -> PRCurvePoint[]     # PR 曲线
      -> ScenarioStats[]    # 场景统计
      -> TemporalMetrics[]  # 时序指标
```

### REST API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/datalab/status` | 获取当前状态 |
| POST | `/api/datalab/recordings/start` | 开始录制 |
| POST | `/api/datalab/recordings/{id}/stop` | 停止录制 |
| GET | `/api/datalab/recordings` | 列出所有录制 |
| GET | `/api/datalab/recordings/{id}` | 获取单个录制 |
| POST | `/api/datalab/recordings/{id}/label` | 打标签 |
| POST | `/api/datalab/experiments/start` | 启动实验 |
| POST | `/api/datalab/experiments/{id}/cancel` | 取消实验 |
| GET | `/api/datalab/experiments/{id}/progress` | 获取进度 |
| GET | `/api/datalab/experiments/{id}/report` | 获取报告 |
| POST | `/api/datalab/full-suite/start` | 启动全量套件 |
| POST | `/api/datalab/import-video` | 导入视频 |
| GET | `/api/datalab/charts/{chart_type}` | 获取图表 |
| GET | `/api/datalab/export/{experiment_id}` | 导出 ZIP |

### 前端界面

DataLab 前端位于 `frontend/src/datalab/`，包含：

- **DataLabPage**: 主页面，录制控制与实验列表
- **AnalysisDashboard**: 分析仪表盘，展示统计卡片与图表
- **HighlightCard**: 关键指标高亮卡片

---

## Transformer 训练详解

### 模型架构

见 [算法详解/Transformer 时序识别器](#3-transformer-时序识别器)。

### 训练数据

#### 1. 合成数据（SyntheticDataset）

当缺乏真实数据集时，使用参数化生成器创建训练样本：

**正样本 — `generate_waving_sequence`**：
- 手腕在 TNLF 中做横向正弦振荡（`freq ~ U(0.4, 2.5) Hz`，`amp ~ U(0.15, 0.45) torso_units`）
- 手臂抬起（`wly < -0.3`，即手腕在肩膀上方）
- theta1 in [35deg, 90deg]，theta2 in [20deg, 60deg]，ext_ratio ~ 0.6
- 前臂方向向量（FDV）随挥手周期性摆动
- 添加 `noise_std ~ U(0.01, 0.06)` 的高斯噪声模拟估计误差
- 5% 帧随机 dropout（模拟关键点遮挡）

**负样本 — 4 种生成器均衡采样**：

| 生成器 | 模拟场景 | 关键区分特征 |
|--------|----------|-------------|
| `generate_walking_sequence` | 走路摆臂 | 双臂反相摆动、频率 0.6-1.2Hz、手臂低于肩膀、FDV 朝下 |
| `generate_standing_sequence` | 站立静止 | 双臂自然下垂、几乎无运动、低 theta1、高 theta2 |
| `generate_phone_use_sequence` | 看手机/挠头 | 手臂抬起但静止、肘部弯曲（theta2 60-100deg）、无周期性 |
| `generate_random_gesture_sequence` | 随机手臂动作 | 布朗运动、无规律、手掌方向随机 |

#### 2. 真实视频数据（HMDB51 + UCF101）

从公开动作数据集中提取真实视频片段，经 YOLO11-Pose 关键点检测后转换为 TNLF 特征：

**HMDB51**（`MichiganNLP/hmdb` via HuggingFace）：
- **正样本**：`wave` 类别（~100 段挥手视频）
- **负样本**：`walk`, `run`, `stand`, `sit`, `turn`, `talk`, `shake_hands`, `hug` 等 27 个类别

**UCF101**（`quchenyuan/UCF101-ZIP` via HuggingFace）：
- **负样本**：`Walking`, `Running`, `JumpingJack` 等全身运动类别

**处理流程**：
```
原始视频 (AVI/MP4)
  -> OpenCV 逐帧读取
  -> YOLO11-Pose 检测人体 17 关键点
  -> 计算 TNLF (wrist_local, theta1/2, ext_ratio, FDV)
  -> 滑动窗口切分 (45 帧 / 步长 15 帧)
  -> 保存为 .npz (X: [N, 45, 12], y: [N])
```

#### 3. 数据增强（TemporalAugmentation）

训练时在线应用：
- **高斯噪声**：`noise_std=0.02`，增强对关键点估计抖动的鲁棒性
- **侧边 dropout**：10% 概率随机清零单侧手腕特征，模拟单侧遮挡
- **特征 dropout**：5% 概率随机清零单个特征维度
- **时序裁剪+缩放**：随机裁剪 80%-100% 时序长度，再用线性插值恢复 45 帧，模拟不同速度的动作

### 训练流程

```bash
cd backend

# 1. 生成/加载合成数据
python -m app.ai.transformer.train \
  --n_samples 20000 \
  --epochs 100 \
  --batch_size 256 \
  --output_dir ./models/transformer

# 2. （可选）处理真实视频数据
# 先下载数据集：
#   HF_ENDPOINT=https://hf-mirror.com hf download MichiganNLP/hmdb --repo-type dataset --local-dir datasets/hmdb51
#   HF_ENDPOINT=https://hf-mirror.com hf download quchenyuan/UCF101-ZIP --repo-type dataset --local-dir datasets/ucf101
# 然后提取特征：
#   python -m app.ai.transformer.real_data_pipeline --datasets hmdb51,ucf101 --data_dir ../datasets --output_dir ../data

# 3. 导出 TorchScript（部署用）
python -m app.ai.transformer.export \
  --checkpoint ./models/transformer/best_model.pt \
  --norm_stats ./models/transformer/norm_stats.npz \
  --output ./models/transformer/waving_transformer.pt
```

**训练配置**：
- Optimizer: AdamW (`lr=1e-3`, `weight_decay=1e-4`)
- Scheduler: CosineAnnealingLR
- Loss: BCEWithLogitsLoss（带标签平滑 `smoothing=0.05`）
- 早停：验证集 F1 连续 10 个 epoch 不提升则停止
- 最佳模型选择依据：验证集 F1（而非 loss）

**归一化**：训练前计算训练集的 per-feature `mean` 和 `std`，导出时封装进 TorchScript：`x_norm = (x - mean) / std`。

### 数据集下载

由于原始视频数据集总计约 **11GB**，超出 Git 仓库容量，请自行下载：

```bash
# 安装 huggingface-cli
pip install huggingface-hub

# 下载 HMDB51（约 2GB）
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  MichiganNLP/hmdb --repo-type dataset --local-dir datasets/hmdb51

# 下载 UCF101（约 7GB）
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  quchenyuan/UCF101-ZIP --repo-type dataset --local-dir datasets/ucf101
```

本仓库已包含 **预提取的 TNLF 特征文件**（`data/processed/real_data_seq45.npz`，约 400KB），可直接用于复现训练或作为小规模验证集，无需下载完整视频。

### 模型性能

在验证集上（合成 20% + 真实 HMDB51/UCF101 混合）：

| 指标 | 值 |
|------|-----|
| F1 | 0.897 |
| Precision | 0.884 |
| Recall | 0.910 |
| 模型大小 | ~180 KB（TorchScript） |
| 推理时延 | ~2-5 ms（RTX 4090，CUDA 12） |

---

## 配置说明

所有参数通过 `docker-compose.yml` 环境变量配置，无需修改代码。

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
| `GESTURE_ENGINE` | `simple-transformer` | 引擎模式 |
| `TRANSFORMER_MODEL_PATH` | `/app/models/transformer/...` | Transformer 模型路径 |
| `TRANSFORMER_CONFIDENCE_THRESHOLD` | `0.5` | Transformer 输出阈值 |

### 姿态锁（硬性规则）

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `GESTURE_THETA1_HAILING_MIN` | `25.0` | 肩-肘抬起角度 theta1（deg） |
| `GESTURE_THETA2_STRAIGHT_MIN` | `15.0` | 肘-腕伸直角度 theta2（deg） |
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
| `WS_FRAME_BUDGET_MS` | `60` | 单帧推理+编码预算（ms） |
| `WS_PUSH_INTERVAL` | `0.04` | 推流间隔（s） |
| `JPEG_QUALITY` | `65` | MJPEG 编码质量 |
| `ADAPTIVE_MIN_SHORT_SIDE` | `320` | 自适应最小短边 |
| `ADAPTIVE_MAX_SHORT_SIDE` | `480` | 自适应最大短边 |

### DataLab

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `DATALAB_RECORDINGS_DIR` | `data/datalab/recordings` | 录制数据目录 |
| `DATALAB_EXPERIMENTS_DIR` | `data/datalab/experiments` | 实验结果目录 |
| `DATALAB_AUTO_GESTURE_BUFFER_S` | `5.0` | 自动录制缓冲时长 |
| `DATALAB_MAX_RECORDING_FRAMES` | `9000` | 最大录制帧数 |

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

## API 接口文档

### REST API

完整 API 文档通过 Swagger UI 提供：http://localhost:18080/api/docs

#### 摄像头管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/cameras` | 列出所有摄像头 |
| POST | `/api/cameras` | 添加摄像头 |
| DELETE | `/api/cameras/{id}` | 删除摄像头 |
| POST | `/api/cameras/{id}/start` | 启动视频流 |
| POST | `/api/cameras/{id}/stop` | 停止视频流 |

#### 健康检查

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 服务健康状态 |

### WebSocket 接口

| 路径 | 说明 | 消息格式 |
|------|------|---------|
| `/ws/video` | 视频流 + 检测结果 | binary MJPEG / JSON |
| `/ws/logs` | 实时日志推送 | JSON {level, message, timestamp} |

#### WebSocket 视频流协议

连接建立后，客户端持续接收二进制 MJPEG 帧，格式为：

```
[2 bytes: JPEG length (big-endian)] [JPEG data] [JSON metadata]
```

JSON 元数据包含当前帧的检测框、关键点、手势识别结果等。

---

## 开发模式

### 前端

```bash
cd frontend
npm install
npm run dev
```

开发服务器运行在 http://localhost:5173。

### 后端

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 单独运行 DataLab 脚本

```bash
# 诊断引擎状态
cd backend
python -m scripts.diag_engines

# Transformer 推理基准测试
python -m scripts.bench_transformer

# 端到端视频测试
python -m scripts.test_e2e_video --video /path/to/video.mp4
```

---

## 故障排查

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
5. 减少 `ADAPTIVE_MAX_SHORT_SIDE`（如降至 360）
6. 增大 `WS_FRAME_BUDGET_MS`（如 80ms）

### Q3: 车辆移动时检测不到招手？

1. 确认使用 `simple-transformer` 或 `transformer` 引擎（`GESTURE_ENGINE`）
2. 检查 YOLO 肩/髋关键点置信度是否过低（影响 TNLF 计算）
3. 尝试降低 `TRANSFORMER_CONFIDENCE_THRESHOLD`（如 0.45）
4. 检查 `GESTURE_MOTION_FREQ_MIN` 是否过高（年长者挥手慢，建议保持 0.35）

### Q4: 静止时挠头被误判为招手？

使用 `simple-transformer` 引擎（默认）。Simple 规则的周期性检测可有效过滤挠头等非周期性动作。如仍误判：

1. 降低 `GESTURE_MOTION_AMP_MIN`（提高振幅门槛）
2. 提高 `GESTURE_PERIOD_CONSISTENCY_MIN`（要求更稳定的周期）
3. 检查 `RAW_WRIST_RANGE_THRESHOLD`（确保静态过滤生效）

### Q5: DataLab 实验运行失败？

1. 检查 `data/datalab/` 目录是否有写权限
2. 确认录制文件存在且非空
3. 查看后端日志中的 `AblationRunner` 错误信息
4. 全量套件实验需要至少一个正样本和一个负样本录制

---

## 项目结构

```
.
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
│       │   ├── bytetrack.yaml    # ByteTrack 配置
│       │   └── transformer/      # Transformer 模型定义与训练脚本
│       │       ├── model.py      # TemporalKeypointTransformer
│       │       ├── engine.py     # 推理引擎
│       │       ├── train.py      # 训练脚本
│       │       ├── export.py     # TorchScript 导出
│       │       └── real_data_pipeline.py  # 真实数据特征提取
│       ├── api/
│       │   ├── routes.py         # REST API（摄像头管理、配置）
│       │   ├── ws.py             # WebSocket 视频推流
│       │   └── logs.py           # WebSocket 日志广播
│       ├── stream/
│       │   ├── handler.py        # 视频流解码（OpenCV + FFmpeg）
│       │   └── manager.py        # 多路流管理
│       └── datalab/
│           ├── api.py            # DataLab REST API
│           ├── models.py         # Pydantic 数据模型
│           ├── persistence.py    # 文件系统持久化
│           ├── recorder.py       # 录制管理器
│           ├── ablation.py       # 消融实验运行器
│           ├── analyzer.py       # 统计分析器
│           ├── charts.py         # SVG/PNG 图表生成
│           └── video_importer.py # 视频导入器
├── frontend/                   # React + Vite + Tailwind
│   ├── Dockerfile
│   ├── nginx-default.conf
│   └── src/
│       ├── App.tsx
│       ├── components/
│       │   ├── VideoPanel.tsx    # 单路视频面板
│       │   ├── VideoGrid.tsx     # 视频网格布局
│       │   ├── GestureOverlay.tsx # 手势叠加层
│       │   ├── StatusBar.tsx     # 状态栏
│       │   ├── LogPanel.tsx      # 实时日志面板
│       │   └── CameraConfig.tsx  # 摄像头配置对话框
│       ├── hooks/
│       │   ├── useWebSocket.ts   # 视频 WebSocket
│       │   ├── useLogWebSocket.ts # 日志 WebSocket
│       │   └── useCameraConfig.ts # 摄像头配置管理
│       ├── datalab/
│       │   ├── DataLabPage.tsx   # DataLab 主页面
│       │   ├── AnalysisDashboard.tsx # 分析仪表盘
│       │   └── HighlightCard.tsx # 指标卡片
│       └── types/
│           └── index.ts
├── models/                     # 模型持久化目录
│   ├── yolo11s-pose.pt
│   └── transformer/
│       └── waving_transformer_real.pt
├── data/                       # 数据目录
│   ├── processed/
│   │   └── real_data_seq45.npz # 预提取 TNLF 特征
│   └── datalab/                # DataLab 录制与实验结果
└── datasets/                   # 原始数据集（HMDB51/UCF101）
```

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
