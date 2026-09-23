#!/usr/bin/env bash
# Run this in the TonyPi VNC terminal AFTER the prepared folder is on the Pi.
#
#   bash ~/pi_ready/pi_vnc_setup_and_run.sh
#   bash ~/pi_ready/pi_vnc_setup_and_run.sh /home/pi/pi_ready
#
# The folder must contain:
#   pipeline_emotion_llm_pi.tgz
#   edge_llm/qwen2.5-1.5b-instruct-q4_k_m.gguf
#   audio_ser/          (optional)
#   wheels/             (optional, aarch64)
set -euo pipefail

READY="${1:-$HOME/pi_ready}"
ROOT="${PI_REPO:-$HOME/RBM-project/momask-codes}"
GGUF_NAME="qwen2.5-1.5b-instruct-q4_k_m.gguf"

echo "== VNC setup =="
echo "ready: $READY"
echo "repo:  $ROOT"

if [[ ! -f "$READY/pipeline_emotion_llm_pi.tgz" ]]; then
  echo "missing $READY/pipeline_emotion_llm_pi.tgz" >&2
  echo "Copy the pi_ready folder onto this robot first (VNC file manager or USB)." >&2
  exit 1
fi

mkdir -p "$ROOT"
tar -xzf "$READY/pipeline_emotion_llm_pi.tgz" -C "$ROOT"
mkdir -p "$ROOT/models/edge_llm" "$ROOT/models/audio_ser"

if [[ -f "$READY/edge_llm/$GGUF_NAME" ]]; then
  cp -f "$READY/edge_llm/$GGUF_NAME" "$ROOT/models/edge_llm/$GGUF_NAME"
  ls -lh "$ROOT/models/edge_llm/$GGUF_NAME"
else
  echo "GGUF not in $READY/edge_llm — Decide will use rules."
fi

if [[ -d "$READY/audio_ser" ]] && [[ -n "$(find "$READY/audio_ser" -type f | head -n 1)" ]]; then
  cp -a "$READY/audio_ser"/. "$ROOT/models/audio_ser"/
  echo "SER files copied"
fi

if [[ -f "$ROOT/venv_inference/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/venv_inference/bin/activate"
fi

if [[ -d "$READY/wheels" ]] && [[ -n "$(find "$READY/wheels" -name '*.whl' | head -n 1)" ]]; then
  echo "== offline pip from $READY/wheels =="
  pip install --no-index --find-links "$READY/wheels" llama-cpp-python || echo "llama-cpp wheel skipped"
  pip install --no-index --find-links "$READY/wheels" funasr modelscope soundfile || echo "SER runtime skipped"
fi

echo "== stop stock TonyPi (releases the camera) =="
sudo systemctl stop tonypi 2>/dev/null || sudo systemctl stop TonyPi 2>/dev/null || true

cd "$ROOT"
export DECIDE_BACKEND=edge_auto
export EDGE_LLM_GGUF="$ROOT/models/edge_llm/$GGUF_NAME"
export EDGE_LLM_N_THREADS="${EDGE_LLM_N_THREADS:-4}"
export ENABLE_MOMASK=0
export ENABLE_AUDIO_SER=1
export AUDIO_SER_CACHE="$ROOT/models/audio_ser"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "== S0 mock (no servos) =="
python3 -m pipeline.run_pi_emotion_llm --once --mock-perception --simulate \
  --decide-backend edge_auto --no-momask \
  --transcript '你好，我有点累，陪我一下'

echo "== live loop: face -> phrase -> ActionGroup. Ctrl-C to stop. =="
exec python3 -m pipeline.run_pi_emotion_llm --decide-backend edge_auto --no-momask
