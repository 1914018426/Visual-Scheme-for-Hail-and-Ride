# Hailuo Vison — 视觉招手即停系统

基于 **YOLO11-Pose** + **MediaPipe Hands** + **Torso-Normalized Local Frame (TNLF)** + **三锁合取机制** 的实时手势识别系统，支持"招手"手势的精确识别，用于智能网约车/出租车的乘客招手场景。

## 系统特性

| 特性 | 描述 |
|------|------|
| 人体姿态检测 | YOLO11x-Pose，实时检测人体 17 个关键点，GPU 半精度推理 |
| 精确手部 ROI | 基于 YOLO wrist 位置 crop 每只手的独立 ROI，运行 MediaPipe Hands（max_num_hands=1），左右手天然正确 |
| Torso-Normalized Local Frame | 以肩中点为原点、躯干长度为单位的局部参考系，消除车体移动导致的伪运动 |
| 三锁合取机制 | 姿态锁 + 朝向锁 + 运动锁 同时满足才确认手势，替换传统状态机 |
| 手掌朝向检测 | MediaPipe 21 点 landmark → 手掌平面法向量 → SLERP 平滑 → 朝向锁判定 |
| 面部过滤（零模型） | 基于 YOLO11-Pose 17 点的双眼对称性 + 肩髋解剖比，无需额外人脸模型 |
| IRI 意图刚性指数 | 手掌法向量在手臂局部标架中的稳定性评分，过滤走路摆臂等伪手势 |
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

### 2. Torso-Normalized Local Frame (TNLF)

**核心问题**：传统系统以画面像素为参考系绘制手腕轨迹，当车辆移动时，静止路人的手腕在画面中会产生伪运动，导致误触发。

**解决方案**：将手腕坐标转换到人体局部参考系。

```
原点       = (left_shoulder + right_shoulder) / 2     # 肩中点
e_x        = normalize(right_shoulder - left_shoulder) # 肩宽方向
e_y        = normalize(mid_hip - origin)               # 躯干方向
torso_scale = |mid_hip - origin|                        # 归一化单位

wrist_local = (dot(wrist - origin, e_x) / torso_scale,
               dot(wrist - origin, e_y) / torso_scale)   # 单位：躯干长度
```

**验证标准**：车子匀速直线行驶时，人站在路边不动，手自然下垂，`wrist_local` 序列的方差趋近于 0，运动锁不触发。

### 3. 精确手部 ROI — MediaPipe Hands

传统方案在全图运行 MediaPipe Hands，存在左右手混淆、远距离手漏检的问题。本系统采用 **YOLO wrist 引导的精确 ROI** 策略：

```
ROI 尺寸 = shoulder_to_shoulder_width × 1.5
ROI 中心 = YOLO wrist 位置
左右手分别检测，max_num_hands=1
```

**MediaPipe 降采样**：手臂未抬起时（`wrist_local[1] >= -0.2`）每 3 帧调用一次；抬起后每帧调用，降低推理开销。

**输出截断**：MediaPipe 调用层只返回手掌平面法向量 `n`（3D 单位向量），禁止向上层暴露 21 点 landmark。

### 4. 手掌法向量平滑 — SLERP

每帧新法向量 `n_new` 与历史平滑值 `n_smooth` 做球面线性插值（SLERP），保持单位长度：

```python
def slerp(n_prev, n_curr, alpha=0.3):
    dot = np.clip(np.dot(n_prev, n_curr), -1.0, 1.0)
    if dot > 0.9995:
        return n_prev * (1-alpha) + n_curr * alpha
    theta_0 = np.arccos(dot)
    theta = theta_0 * alpha
    return (n_prev * np.sin(theta_0 - theta) + n_curr * np.sin(theta)) / np.sin(theta_0)
```

朝向锁判定基于 `n_smooth` 而非单帧 `n`，避免帧间抖动。

### 5. θ1-θ2 角度链 — 姿态锁

参考 Tunis taxi-hailing gesture recognition（MDPI 2023）论文，定义关节角度链：

```
θ1 = ∠(hip, shoulder, elbow)      — 手臂整体抬起程度
θ2 = ∠(shoulder, elbow, wrist)    — 前臂伸直/弯曲程度
ext_ratio = |shoulder-wrist| / (|shoulder-elbow| + |elbow-wrist|)
                                         — 手臂伸展比例（伸直≈1.0，折叠<1.0）
```

**姿态锁判定条件**：`θ1 > 25°` 且 `θ2 > 15°` 且 `ext_ratio > 0.1`，连续满足 **3 帧**。

### 6. 三锁合取机制（Triple-Lock Conjunction）

删除传统的 `idle → posed → oscillating → confirmed` 状态机，替换为三个独立锁的合取：

| 锁 | 判定条件 | 最小持续帧数 | 释放条件 |
|----|---------|-------------|---------|
| **姿态锁** | `θ1 > 25°` 且 `θ2 > 15°` 且 `ext > 0.1` | 3 帧 | 任一角度条件不满足 |
| **朝向锁** | `n_smooth` 与摄像头视线方向夹角 `< 45°`（掌心朝车） | 5 帧 | 夹角 `> 60°` |
| **运动锁** | FFT 主导频率 `0.5~3Hz` 且振幅 `> 0.1 torso/s` | 3 帧 | 周期性消失或速度 `< 0.05` |

**确认逻辑**：
- 三锁同时满足 → 输出 `confirmed_hailing`，保持 **15 帧**（约 1 秒）后衰减
- 任一锁断开 → 若仍在保持期内，输出衰减中的置信度；超出保持期 → `none`

**最终意图分数**：
```
S = Pose_score × R_iri × Motion_score × F_human
```

### 7. 面部过滤层（零模型）

基于 YOLO11-Pose 17 点关键点，**无需额外人脸模型**：

```python
eye_sym = min(d_leye, d_reye) / max(d_leye, d_reye)        # 双眼到鼻子距离对称性
body_score = 1 - |shoulder_width / hip_width - 1.25| / 0.8  # 肩髋解剖比
F_human = 0.6 × face_conf × eye_sym + 0.4 × max(0, body_score)
```

**硬过滤**：`F_human < 0.25` → 直接丢弃该目标，不进入后续手势判断。

**软调制**：`F_human ∈ [0.25, 0.6]` → 最终意图分数乘以 `0.5 + 0.5 × F_human`。

### 8. IRI — 意图刚性指数

在手臂局部坐标系中，计算手掌法向量相对于手臂的稳定性：

```
手臂标架：
  origin = elbow
  e_x    = normalize(shoulder - elbow)   # 上臂方向
  e_y    = normalize(wrist - elbow)      # 前臂方向
  e_z    = cross(e_x, e_y)               # 垂直于手臂平面

n_local = [dot(n_world, e_x), dot(n_world, e_y), dot(n_world, e_z)]
```

滑动窗口（15 帧）内，计算 `n_local` 的球面集中度：

```
R_iri = ||mean(n_local over window)||  ∈ [0, 1]
```

- `R_iri ≈ 1`：手掌法向量在手臂标架中非常稳定 → 真实招手（手臂周期性运动时手掌朝向保持恒定）
- `R_iri ≈ 0`：手掌法向量剧烈变化 → 走路摆臂等伪手势

IRI 作为乘法因子融入最终意图分数，**非阻塞项**。

### 9. 运动锁 — 周期性检测（基于 wrist_local）

**输入**：`wrist_local` 时序序列（不再是画面像素坐标）。

使用 **zero-crossing + 自相关函数 (ACF)** 检测稳定的周期性运动：

1. 去趋势（线性漂移去除）
2. Zero-crossing 计数估计频率
3. ACF 峰值检测（FFT 加速）
4. 频率一致性校验（zc vs acf）
5. 周期一致性（变异系数 CV）

人类挥手典型频率范围：**0.5–3 Hz**。

### 10. 手势统一输出

三锁同时满足时对外统一输出 `waving`（招手），内部保留调试信息。

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
│       │   ├── gesture.py    # 三锁合取机制核心
│       │   ├── local_frame.py    # TNLF 局部参考系（纯函数）
│       │   ├── facing.py         # 面部过滤（零模型，纯函数）
│       │   ├── slerp.py          # SLERP 法向量平滑（纯函数）
│       │   ├── iri.py            # IRI 意图刚性指数（纯函数）
│       │   ├── gesture_stgcn.py  # ST-GCN 备选方案（未启用）
│       │   └── direction.py      # 摄像头方向映射
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
| **姿态锁** | | |
| `GESTURE_THETA1_HAILING_MIN` | `25.0` | θ1 抬起阈值（°） |
| `GESTURE_THETA2_STRAIGHT_MIN` | `15.0` | θ2 伸直阈值（°） |
| `GESTURE_ARM_EXTENSION_MIN` | `0.10` | 手臂伸展比例最小值 |
| `GESTURE_POSE_MIN_FRAMES` | `3` | 姿态锁最小持续帧数 |
| **朝向锁** | | |
| `GESTURE_ORIENTATION_LOCK_ANGLE` | `45.0` | 掌心朝车夹角阈值（°） |
| `GESTURE_ORIENTATION_RELEASE_ANGLE` | `60.0` | 朝向锁释放角度（°） |
| `GESTURE_ORIENTATION_MIN_FRAMES` | `5` | 朝向锁最小持续帧数 |
| **运动锁** | | |
| `GESTURE_MOTION_FREQ_MIN` | `0.5` | 运动锁频率下限（Hz） |
| `GESTURE_MOTION_FREQ_MAX` | `3.0` | 运动锁频率上限（Hz） |
| `GESTURE_MOTION_AMP_MIN` | `0.1` | 运动锁最小振幅（torso_units） |
| `GESTURE_MOTION_SPEED_MIN` | `0.05` | 运动锁速度释放阈值（TU/s） |
| `GESTURE_MOTION_MIN_FRAMES` | `3` | 运动锁最小持续帧数 |
| **确认与保持** | | |
| `GESTURE_HOLD_MAX_FRAMES` | `15` | 确认后保持帧数 |
| **面部过滤** | | |
| `GESTURE_FACING_HARD_THRESHOLD` | `0.25` | 硬过滤阈值 |
| `GESTURE_FACING_SOFT_THRESHOLD` | `0.6` | 软过滤上限 |
| **周期性检测** | | |
| `GESTURE_PERIOD_MIN_CYCLES` | `2` | 最小完整周期数 |
| `GESTURE_PERIOD_CONSISTENCY_MIN` | `0.5` | 最小周期一致性 |
| `GESTURE_PERIOD_MIN_FREQ` | `0.5` | 周期性检测频率下限 |
| `GESTURE_PERIOD_MAX_FREQ` | `3.0` | 周期性检测频率上限 |
| **置信度** | | |
| `GESTURE_EMA_ALPHA` | `0.35` | EMA 平滑系数 |
| `GESTURE_CONFIDENCE_THRESHOLD` | `0.55` | 输出阈值 |

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
- 降低 `GESTURE_MOTION_AMP_MIN`：允许更小幅度的挥动
- 降低 `GESTURE_POSE_MIN_FRAMES` / `GESTURE_MOTION_MIN_FRAMES`：更快确认

### Q5: 车体移动时静止路人被误触发？

**A:** 本系统已采用 **TNLF 局部参考系**，所有轨迹、速度、周期性判断均基于人体相对坐标。若仍有误触发，请检查：
- YOLO 肩/髋关键点置信度是否过低（导致 `torso_scale` 不稳定）
- `GESTURE_MOTION_SPEED_MIN` 是否设置过低

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
