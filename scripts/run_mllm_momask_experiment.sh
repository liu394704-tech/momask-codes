#!/usr/bin/env bash
# MLLM（中转台 GPT-4o）→ MoMask：跑通整条 live_emotion_monitor 管线。
# 默认：gen_t2m 不渲染 MP4，只写 joints/*.npy 与 animations/*/*.bvh（渲染代码仍保留，可用环境变量打开）。
#
# 使用前请设置:
#   export OPENAI_API_KEY="sk-..."
#   export OPENAI_BASE_URL="https://你的中转台/v1"
# 可选:
#   export LIVE_MONITOR_GPU_ID=0
#   export LIVE_MONITOR_MAX_VISION_OK=1   # 成功分析 1 轮后结束工作线程（便于实验）
#   export LIVE_MONITOR_MOMASK_RENDER_VIDEO=1  # 需要 MP4 时再开
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPU="${LIVE_MONITOR_GPU_ID:-0}"
MAXOK="${LIVE_MONITOR_MAX_VISION_OK:-1}"

exec python3 live_emotion_monitor.py \
  --momask \
  --gpu-id "$GPU" \
  --max-vision-ok "$MAXOK"
