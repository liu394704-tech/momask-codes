#!/usr/bin/env bash
# Install on-device speech emotion (emotion2vec+ seed) on the TonyPi.
# Needs LAN/internet to pip + ModelScope. On HW* hotspot, only a local copy works.
# Not the cloud 中转站.
set -euo pipefail

ROOT="${1:-$PWD}"
cd "$ROOT"

if [[ -f venv_inference/bin/activate ]]; then
  # shellcheck disable=SC1091
  source venv_inference/bin/activate
fi

export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export AUDIO_SER_MODEL="${AUDIO_SER_MODEL:-iic/emotion2vec_plus_seed}"
export AUDIO_SER_CACHE="${AUDIO_SER_CACHE:-$ROOT/models/audio_ser}"
mkdir -p "$AUDIO_SER_CACHE"

if [[ -x "$ROOT/scripts/pi_stage_weights_offline.sh" ]]; then
  bash "$ROOT/scripts/pi_stage_weights_offline.sh" "${WEIGHTS_SRC:-}" || true
fi

_can_reach_hub() {
  [[ "${OFFLINE:-0}" == "1" ]] && return 1
  if ! ip route show default 2>/dev/null | grep -q .; then
    return 1
  fi
  ping -c 1 -W 2 223.5.5.5 >/dev/null 2>&1 || ping -c 1 -W 2 1.1.1.1 >/dev/null 2>&1
}

if ! _can_reach_hub; then
  echo "no internet (HW hotspot / OFFLINE=1). Skip pip + ModelScope."
  echo "SER files, if any, must already be in $AUDIO_SER_CACHE"
  echo "Face + 778 phrases still run. audio_emotion stays empty without weights."
  exit 0
fi

echo "== pip: funasr + modelscope (CPU SER) =="
pip install -U pip
pip install "funasr>=1.1.0" modelscope soundfile

echo "== download $AUDIO_SER_MODEL into $AUDIO_SER_CACHE =="
python - <<PY
import os
from funasr import AutoModel
cache = os.environ["AUDIO_SER_CACHE"]
model_id = os.environ["AUDIO_SER_MODEL"]
print("loading", model_id)
AutoModel(model=model_id, hub="ms", cache_dir=cache, disable_update=True, device="cpu")
print("ok", cache)
PY

echo "DONE. Persist:"
echo "  export ENABLE_AUDIO_SER=1"
echo "  export AUDIO_SER_MODEL=$AUDIO_SER_MODEL"
echo "  export AUDIO_SER_CACHE=$AUDIO_SER_CACHE"
echo "Then say 小幻小幻 and speak; the same wav is used for Whisper + SER."
