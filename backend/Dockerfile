FROM python:3.10-slim

# 构建期下载的权重文件名（与 compose 中 YOLO_MODEL 保持一致）
ARG YOLO_MODEL=yolo11x-pose.pt
ENV YOLO_MODEL=${YOLO_MODEL}

# 设置工作目录
WORKDIR /app

# 安装系统依赖（OpenCV、FFmpeg等运行时库）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装Python依赖 (force-rebuild 2026-04-25)
COPY requirements.txt .
# 显式安装 CUDA 版 PyTorch，避免在 CPU-only 构建环境中误装 CPU 版本
RUN pip install --no-cache-dir \
    --default-timeout=300 \
    --retries 5 \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 || \
    pip install --no-cache-dir \
    --default-timeout=300 \
    --retries 5 \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com \
    torch torchvision

RUN pip install --no-cache-dir \
    --default-timeout=300 \
    --retries 5 \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com \
    -r requirements.txt

# 只复制模型下载所需最小文件，避免源码改动导致模型层缓存失效
COPY docker/download-model.sh /tmp/download-model.sh
COPY model-bake /tmp/model-bake

# 构建期代理：宿主机代理请用 host.docker.internal:端口（compose 已加 extra_hosts），
# 勿用 127.0.0.1（在构建容器内指向容器自身，代理不会生效）。
# BuildKit 会把当前 shell 的 HTTP_PROXY/HTTPS_PROXY 作为 build-arg 传入，此处需声明才会生效。
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy
ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy}

# 模型：优先使用构建上下文内的 model-bake/${YOLO_MODEL}（可事先放到 backend/model-bake/ 避免构建期下载）
# 否则再走 download-model.sh（此步骤尽量独立于源码变更，提升缓存命中）
# 注意：compose 请勿再把「空」命名卷挂到 /app/models，否则会覆盖镜像内权重
# 若构建期下载失败，不中断构建；应用启动后会通过 _download_yolo_weights_cn 再次尝试
RUN mkdir -p /app/models /app/.ultralytics \
    && chmod +x /tmp/download-model.sh \
    && if [ -f "/tmp/model-bake/${YOLO_MODEL}" ]; then \
         sz=$(wc -c < "/tmp/model-bake/${YOLO_MODEL}" | tr -d ' '); \
         if [ "$sz" -ge 100000 ]; then \
           cp "/tmp/model-bake/${YOLO_MODEL}" "/app/models/${YOLO_MODEL}" \
           && echo "[build] using baked model model-bake/${YOLO_MODEL} ($sz bytes)"; \
         else \
           echo "[build] model-bake/${YOLO_MODEL} too small ($sz), downloading..." \
           && (/tmp/download-model.sh "${YOLO_MODEL}" "/app/models/${YOLO_MODEL}" || echo "[build] model download failed, will retry at runtime"); \
         fi; \
       else \
         (/tmp/download-model.sh "${YOLO_MODEL}" "/app/models/${YOLO_MODEL}" || echo "[build] model download failed, will retry at runtime"); \
       fi

# 复制应用代码（放在模型层之后，避免每次改代码都触发模型重下）
COPY . .

# 暴露服务端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# 使用entrypoint脚本启动
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
