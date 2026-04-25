#!/bin/bash
set -e

mkdir -p /app/models /app/.ultralytics

# 不在此阻塞下载大权重：否则在 curl 完成前 uvicorn 不启动，健康检查与网关会一直失败。
# 权重由应用启动后写入 /app/models（Docker 卷持久化），见 app.ai.detector._download_yolo_weights_cn

echo "启动 Hailuo Car Backend Server..."
if [ "${UVICORN_RELOAD:-0}" = "1" ] || [ "${UVICORN_RELOAD:-}" = "true" ]; then
    exec uvicorn app.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 1 \
        --loop uvloop \
        --http httptools \
        --ws websockets \
        --log-level info \
        --access-log \
        --reload \
        --reload-dir /app/app
else
    exec uvicorn app.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 1 \
        --loop uvloop \
        --http httptools \
        --ws websockets \
        --log-level info \
        --access-log
fi
