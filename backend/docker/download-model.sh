#!/bin/sh
# 构建阶段下载 YOLO 权重到固定路径（多源依次尝试，任一成即退出 0）
# 用法: download-model.sh <文件名> <目标绝对路径>
# 例:   download-model.sh yolov8x-pose.pt /app/models/yolov8x-pose.pt

set -eu
MODEL="${1:?model filename}"
DEST="${2:?dest path}"
MIN_BYTES="${MIN_BYTES:-100000}"

mkdir -p "$(dirname "$DEST")"
TMP="${DEST}.part"
rm -f "$TMP"

# 顺序：国内 HF 镜像 → GitHub Release（构建机通常可直连 GitHub）
# hf-mirror 大文件可能较慢，单次给足时间；失败则换下一源
try_curl() {
  _url="$1"
  _max="${2:-3600}"
  echo "[download-model] trying: $_url"
  if curl -fL --connect-timeout 30 --max-time "$_max" -o "$TMP" "$_url"; then
    sz=$(wc -c < "$TMP" | tr -d ' ')
    if [ "$sz" -ge "$MIN_BYTES" ]; then
      mv -f "$TMP" "$DEST"
      echo "[download-model] saved $DEST ($sz bytes)"
      return 0
    fi
    echo "[download-model] too small ($sz bytes), discard"
  else
    echo "[download-model] curl failed for $_url"
  fi
  rm -f "$TMP"
  return 1
}

# Hugging Face 国内镜像（与 huggingface 仓库路径一致）
for repo in "Ultralytics/YOLOv8" "Ultralytics/yolov8"; do
  if try_curl "https://hf-mirror.com/${repo}/resolve/main/${MODEL}" 1800; then
    exit 0
  fi
done

# GitHub 官方资源（非代理）
for tag in v8.4.0 v8.3.0 v8.2.0 v8.1.0; do
  if try_curl "https://github.com/ultralytics/assets/releases/download/${tag}/${MODEL}" 7200; then
    exit 0
  fi
done

echo "[download-model] ERROR: all sources failed for ${MODEL}"
exit 1
