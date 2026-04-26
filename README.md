# Hailuo Car — 视觉招手即停系统

基于 **YOLO11-Pose** + **MediaPipe Hands** + **θ1-θ2 角度链状态机** 的实时手势识别系统，支持"招手"与"举手"两种手势的精确识别，用于智能网约车/出租车的乘客招手场景。

## 系统特性

| 特性 | 描述 |
|------|------|
| 人体姿态检测 | YOLO11x-Pose，实时检测人体 17 个关键点，GPU 半精度推理 |
| 精确手部 ROI | 基于 YOLO wrist 位置 crop 每只手的独立 ROI，运行 MediaPipe Hands（max_num_hands=1），左右手天然正确 |
| 手掌朝向检测 | MediaPipe 21 点 landmark + 3D Z 深度 + 手掌平面法向量，判断掌心是否朝向摄像头 |
| θ1-θ2 角度链 | 基于 Tunis taxi-hailing（MDPI 2023）论文的关节角度链， torso-normalized 坐标系 |
| 意图驱动状态机 | idle → posed/hand_up → oscillating → confirmed，过滤偶发误动作 |
| 挥手轨迹容错 | 手腕运动路径经过肩膀时自动跳过 ext_ratio 硬过滤，避免关键点混淆导致漏检 |
| 多路视频流 | 本地摄像头、RTSP/RTMP/HTTP 网络视频流 |
| 实时推流 | WebSocket 推送 MJPEG 帧 + 检测结果 + 方向决策 |
| Docker 部署 | 容器化一键部署，GPU 直通 |

## 快速开始

### 前置要求

- Docker >= 20.10
- Docker Compose >= 2.0
- NVIDIA GPU（推荐 RTX 4090 级别，CUDA 12.x）
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
docker compose down
```

---

## 算法详解

### 1. 人体姿态检测 — YOLO11-Pose

采用 **YOLO11x-Pose** 模型，在单帧中同时完成目标检测与人体 17 关键点回归：

- **输入分辨率**：`896×896`（半精度 `fp16`）
- **置信度阈值**：`0.35`，过滤低质量检测
- **最大检测数**：`20` 人
- **跟踪器**：ByteTrack，跨帧 `track_id` 关联

输出关键点索引（COCO 格式）：

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

### 2. 精确手部 ROI — MediaPipe Hands

传统方案在全图运行 MediaPipe Hands，存在左右手混淆、远距离手漏检的问题。本系统采用 **YOLO wrist 引导的精确 ROI** 策略：

```
ROI 尺寸 = shoulder_to_shoulder_width × 1.5
ROI 中心 = YOLO wrist 位置
左右手分别检测，max_num_hands=1
```

优势：
- 左右手天然分离，无需后续左右判定
- 远距离小手也能被精确 crop 后检测
- 运行效率更高（只处理手部区域）

### 3. 手掌朝向检测

基于 MediaPipe Hands 输出的 21 个 3D landmark：

**3D Z 深度法**：
- 指尖（index_finger_tip, middle_finger_tip, ring_finger_tip, pinky_tip）的 Z 坐标
- 与 MCP 关节（掌指关节）Z 坐标比较
- 4 个指尖中 ≥3 个 Z 值大于 MCP → 指尖朝外 → 掌心朝向摄像头

**手掌平面法向量**：
- 取 wrist、index_mcp、pinky_mcp 构成三角形
- 计算法向量，判断法向量是否指向摄像头（Z 轴正方向）

手掌朝向作为置信度加分项（`+0.15`），提升"对着车招手"的识别准确率。

### 4. θ1-θ2 角度链 — 手臂姿势判断

参考 Tunis taxi-hailing gesture recognition（MDPI 2023）论文，定义 torso-normalized 坐标系下的 θ1-θ2 角度链：

```
θ1 = ∠(hip, shoulder, elbow)      — 手臂整体抬起程度
θ2 = ∠(shoulder, elbow, wrist)    — 前臂伸直/弯曲程度
ext_ratio = |shoulder-wrist| / (|shoulder-elbow| + |elbow-wrist|)
                                         — 手臂伸展比例（伸直≈1.0，折叠<1.0）
```

**关键点容错**：hip 置信度 < 0.3 时，用 shoulder 向下偏移 torso_size 估算，恢复 θ1 计算。

**手腕靠近肩膀时的特殊处理**：当 `shoulder-wrist` 像素距离 < 0.25×躯干时（挥手轨迹经过肩膀 / YOLO 关键点混淆），跳过 `ext_ratio` 硬过滤，避免手臂折叠误判。

#### 姿势判定逻辑

| 条件 | 阈值 | 说明 |
|------|------|------|
| θ1 > 25° | `theta1_hailing_min` | 手臂大幅抬起（站立招手） |
| 15° < θ1 ≤ 150° | `theta1_greeting_min/max` | 手臂平伸（坐着挥手） |
| θ2 > 15° | `theta2_straight_min` | 前臂不折叠 |
| ext_ratio > 0.10 | `arm_extension_min` | 手臂伸展（手腕靠近肩膀时自动豁免） |
| wrist_above_elbow > -0.05×ts | — | 手腕不低于手肘太多 |
| wrist_above_shoulder / ts > 0.05 | — | 手腕在肩膀上方（坐着举手） |

### 5. 意图驱动状态机

每帧对左右手分别运行独立状态机，取置信度高者输出。

```
              ┌──────────┐
     无手臂   │   idle   │◄──────────────────────────┐
     姿势     └────┬─────┘                           │
                   │ is_raised / is_forward          │
                   ▼                                 │
        ┌─────────────────────┐                      │
        │  posed  │  hand_up  │                      │
        │ (平伸)  │  (高举)    │                     │
        └────┬────┴─────┬─────┘                      │
             │          │ is_moving & direction_ok   │
             │          ▼                            │
             │   ┌─────────────┐                     │
             └──►│ oscillating │─────────────────────┤
                 │  (检测摆动)  │  stop_frames>=15    │
                 └──────┬──────┘                     │
                        │ frames>=3 & purity_ok      │
                        ▼                            │
                 ┌─────────────┐                     │
                 │  confirmed  │─────────────────────┘
                 │  (输出手势)  │  pose降级 / stop
                 └─────────────┘
```

**状态说明**：

| 状态 | 输出 | 说明 |
|------|------|------|
| idle | `none` | 手臂自然下垂 |
| posed | `none` | 手臂平伸，等待挥动（已进入候选） |
| hand_up | `hand_up` | 手臂高举，等待挥动。持续即输出，让用户确认姿势正确 |
| oscillating | `none` | 检测到来回摆动，等待确认帧数达标 |
| confirmed | `waving` / `hand_up` | 运动质量通过，输出最终手势 |

**确认条件**（oscillating → confirmed）：
- `consecutive_wave_frames >= 3`（连续挥动帧数）
- `motion_purity >= 0.10`（方向历史中有效运动帧占比）
- `fast_mode=true` 时跳过周期性检测，快速响应

**衰减保持**（confirmed 态）：
- 停止挥动后保持 `stop_reset_frames=15` 帧（约 1 秒）
- 置信度按 `decay = max(0.3, 1.0 - frames * 0.015)` 衰减

### 6. 速度归一化与方向追踪

```
vx = (wrist_x - last_wrist_x) / dt / torso_size   [TU/s]
vy = (wrist_y - last_wrist_y) / dt / torso_size   [TU/s]
v_mag = sqrt(vx² + vy²)
```

TU（Torso Unit）= 躯干高度像素，消除不同距离/分辨率的影响。

**方向判定**：
- `|vx| > |vy| × 1.2` → `horizontal`（水平挥动）
- `|vy| > |vx| × 1.2` → `vertical`（垂直挥动）
- 否则 → `diagonal`

**符号变化追踪**：追踪主方向（horizontal/vertical）上的速度符号变化次数，用于判定来回摆动。

### 7. 手势统一输出

内部状态机保留 `greeting`（水平挥动）与 `hailing`（垂直挥动）的区分用于调试，但**对外统一输出 `waving`**。

`_classify_intent` 判定逻辑：
- `v_ratio >= 0.50`（垂直为主）+ `is_raised` → `waving`
- `h_ratio >= 0.50`（水平为主）+ `is_forward/is_raised` → `waving`
- `d_ratio >= 0.40`（对角线）+ 姿势满足 → `waving`
- 有运动但方向不明确 + 姿势满足 → `waving`
- 无运动历史 → `hand_up`

### 8. 身体面向过滤

```python
facing_score = body_facing_score()  # 基于肩-髋宽度比
if facing_score < 0.05:
    is_posed = False  # 背对摄像头的人不太可能是对着车招手
```

---

## 项目结构

```
├── docker-compose.yml      # Docker Compose 配置
├── nginx.conf              # Nginx 反向代理配置
├── README.md               # 项目说明
├── backend/                # 后端服务
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── entrypoint.sh
│   └── app/
│       ├── main.py           # FastAPI 入口
│       ├── config.py         # 配置中心（支持环境变量覆盖）
│       ├── ai/
│       │   ├── detector.py   # YOLO11-Pose + ByteTrack + MediaPipe Hands
│       │   ├── gesture.py    # θ1-θ2 角度链 + 状态机核心
│       │   ├── gesture_stgcn.py  # ST-GCN 备选方案（未启用）
│       │   └── direction.py  # 摄像头方向映射
│       ├── api/
│       │   ├── routes.py     # REST API
│       │   └── ws.py         # WebSocket 视频推流
│       └── stream/
│           ├── handler.py    # 视频流解码
│           └── manager.py    # 多路流管理
├── frontend/               # 前端服务（React + Vite + Tailwind）
│   ├── Dockerfile
│   ├── nginx-default.conf
│   ├── package.json
│   └── src/
│       ├── components/
│       │   ├── VideoPanel.tsx     # 视频面板 + 手势标签
│       │   ├── GestureOverlay.tsx # SVG 骨骼叠加
│       │   └── VideoGrid.tsx      # 多摄像头网格
│       ├── hooks/
│       │   └── useWebSocket.ts    # WebSocket 连接 + 帧解析
│       └── types/
│           └── index.ts           # TypeScript 类型定义
└── models/                 # 模型权重持久化目录（运行时自动生成）
```

---

## 配置说明

所有参数均通过 `docker-compose.yml` 环境变量配置，无需改代码：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `YOLO_MODEL` | `yolo11x-pose.pt` | 姿态检测模型 |
| `AI_CONF_THRESHOLD` | `0.35` | 人体检测置信度阈值 |
| `AI_MAX_DETECTIONS` | `20` | 最大检测人数 |
| `STREAM_FPS` | `15` | 推流帧率 |
| `JPEG_QUALITY` | `88` | MJPEG 质量 |
| `ENABLE_HAND_DETECTION` | `true` | MediaPipe Hands 开关 |
| `ENABLE_TRACKING` | `true` | ByteTrack 跟踪开关 |
| `GESTURE_THETA1_HAILING_MIN` | `25.0` | θ1 高举阈值（°） |
| `GESTURE_THETA1_GREETING_MIN` | `15.0` | θ1 平伸下限（°） |
| `GESTURE_THETA1_GREETING_MAX` | `150.0` | θ1 平伸上限（°） |
| `GESTURE_THETA2_STRAIGHT_MIN` | `15.0` | θ2 伸直阈值（°） |
| `GESTURE_ARM_EXTENSION_MIN` | `0.10` | 手臂伸展比例最小值 |
| `GESTURE_VELOCITY_THRESHOLD` | `0.3` | 速度阈值（TU/s） |
| `GESTURE_CONFIRM_FRAMES` | `3` | 确认所需连续挥动帧数 |
| `GESTURE_STOP_RESET_FRAMES` | `15` | 停止后重置帧数 |
| `GESTURE_FAST_MODE` | `true` | 快速模式（跳过周期性检测） |

---

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

---

## 常见问题

### Q1: 构建时模型下载超时？

**A:** 预先将 `.pt` 模型文件放入 `./models/` 目录，或设置代理后构建：

```bash
export HTTP_PROXY=http://host.docker.internal:7890
docker compose up -d --build
```

### Q2: MediaPipe Hands 加载失败？

**A:** 本项目固定使用 `mediapipe==0.10.8`。更高版本（如 0.10.33+）移除了 `mp.solutions.hands` API。

### Q3: 视频流卡顿或延迟高？

**A:** 可通过以下方式优化：
1. 降低分辨率：`STREAM_WIDTH` / `STREAM_HEIGHT`
2. 降低帧率：`STREAM_FPS`
3. 降低 JPEG 质量：`JPEG_QUALITY`

### Q4: 手势识别不灵敏？

**A:** 调整以下参数：
- 降低 `GESTURE_THETA1_HAILING_MIN`：允许手臂更低的角度
- 降低 `GESTURE_ARM_EXTENSION_MIN`：允许更弯曲的手臂
- 降低 `GESTURE_VELOCITY_THRESHOLD`：允许更慢的挥动
- 降低 `GESTURE_CONFIRM_FRAMES`：更快确认

---

## 端口说明

| 端口 | 服务 | 用途 |
|------|------|------|
| 18080 | Nginx | 统一入口（推荐） |
| 8001 | Backend | 后端 API / WebSocket（直接访问） |
| 5173 | Frontend | 前端页面（直接访问） |

---

## 许可证

MIT License
