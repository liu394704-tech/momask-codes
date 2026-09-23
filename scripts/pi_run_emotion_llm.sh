#!/usr/bin/env bash
# Deploy / run the Pi emotion -> edge LLM -> ActionGroup loop.
# MoMask default OFF. Set ENABLE_MOMASK=1 (or pass --momask) to generate joints.
set -euo pipefail

ROOT="${1:-$PWD}"
cd "$ROOT"

if [[ -f venv_inference/bin/activate ]]; then
  # shellcheck disable=SC1091
  source venv_inference/bin/activate
fi

export DECIDE_BACKEND="${DECIDE_BACKEND:-edge_llm}"
export EDGE_LLM_N_THREADS="${EDGE_LLM_N_THREADS:-4}"
export EDGE_LLM_GGUF="${EDGE_LLM_GGUF:-$ROOT/models/edge_llm/qwen2.5-1.5b-instruct-q4_k_m.gguf}"
export WHISPER_SIZE="${WHISPER_SIZE:-tiny}"
export ENABLE_MOMASK="${ENABLE_MOMASK:-0}"

MOMASK_FLAGS=(--no-momask)
if [[ "${ENABLE_MOMASK}" == "1" || "${ENABLE_MOMASK}" == "true" || "${ENABLE_MOMASK}" == "on" ]]; then
  MOMASK_FLAGS=(--momask)
  echo "== MoMask SWITCH: ON (joints.npy, ActionGroup still runs first) =="
else
  echo "== MoMask SWITCH: OFF (ActionGroup only) =="
fi

echo "== stop stock TonyPi (camera owner) =="
sudo systemctl stop tonypi 2>/dev/null || sudo systemctl stop TonyPi 2>/dev/null || true

echo "== S0 mock perception (no servos) =="
S0_EXTRA=()
if [[ "${MOMASK_FLAGS[0]}" == "--momask" ]]; then
  S0_EXTRA=(--momask-dry-run)
fi
python -m pipeline.run_pi_emotion_llm --once --mock-perception --simulate \
  --decide-backend "${DECIDE_BACKEND}" \
  --transcript '你好，我有点累，陪我一下' \
  "${MOMASK_FLAGS[@]}" "${S0_EXTRA[@]}"

echo "== live loop (Ctrl-C to stop) =="
echo "Say 小幻小幻 then 前进/你好/我有点累; or just smile at the camera."
echo "Toggle later: ENABLE_MOMASK=1 $0    or    python -m pipeline.run_pi_emotion_llm --momask"
exec python -m pipeline.run_pi_emotion_llm --decide-backend "${DECIDE_BACKEND}" "${MOMASK_FLAGS[@]}"
