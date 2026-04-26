FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖（src 目录使用 backend/requirements.txt）
COPY backend/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 下载YOLO11模型（国内镜像优先，失败则走GitHub）
RUN mkdir -p /app/models && \
    (wget --timeout=60 -O /app/models/yolo11n.pt \
        https://hf-mirror.com/Ultralytics/yolo11/resolve/main/yolo11n.pt 2>/dev/null || \
     wget --timeout=60 -O /app/models/yolo11n.pt \
        https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) || \
    echo "[warn] yolo11n.pt download failed, will retry at runtime"

# 复制代码
COPY src/ /app/src/

EXPOSE 8010 8011

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8010"]
