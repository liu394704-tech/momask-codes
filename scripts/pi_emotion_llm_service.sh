#!/usr/bin/env bash
# Background loop used by tonypi-emotion-llm.service.
# Stops the stock TonyPi app so this process can take the camera.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

systemctl stop tonypi 2>/dev/null || systemctl stop TonyPi 2>/dev/null || true

export DECIDE_BACKEND="${DECIDE_BACKEND:-edge_auto}"
export EDGE_LLM_N_THREADS="${EDGE_LLM_N_THREADS:-4}"
export EDGE_LLM_GGUF="${EDGE_LLM_GGUF:-$ROOT/models/edge_llm/qwen2.5-1.5b-instruct-q4_k_m.gguf}"
export ENABLE_MOMASK=0
export ENABLE_AUDIO_SER="${ENABLE_AUDIO_SER:-1}"
export AUDIO_SER_CACHE="${AUDIO_SER_CACHE:-$ROOT/models/audio_ser}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec /usr/bin/python3 -m pipeline.run_pi_emotion_llm \
  --decide-backend "$DECIDE_BACKEND" \
  --no-momask
