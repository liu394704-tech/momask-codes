#!/usr/bin/env bash
# Copy edge-model weights onto the TonyPi without the internet.
#
# HW* hotspot (192.168.149.1) is a local AP only. The Pi cannot reach
# ModelScope / HF / PyPI from that network. Face emotion does not need
# this script — it already lives in /home/pi/TonyPi/Functions.
#
# Usage on the Pi (VNC, after the USB or scp drop is visible):
#   bash scripts/pi_stage_weights_offline.sh
#   bash scripts/pi_stage_weights_offline.sh /media/pi/USB
#   bash scripts/pi_stage_weights_offline.sh /tmp/incoming_weights
#
# Expected filenames in the source folder (any depth):
#   qwen2.5-1.5b-instruct-q4_k_m.gguf          → models/edge_llm/
#   emotion2vec*  or  audio_ser/               → models/audio_ser/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-}"
GGUF_NAME="qwen2.5-1.5b-instruct-q4_k_m.gguf"
DEST_LLM="${WEIGHTS_LLM_DIR:-$ROOT/models/edge_llm}"
DEST_SER="${WEIGHTS_SER_DIR:-$ROOT/models/audio_ser}"
mkdir -p "$DEST_LLM" "$DEST_SER"

search_roots=()
if [[ -n "$SRC" ]]; then
  search_roots+=("$SRC")
else
  search_roots+=(
    "$ROOT/models"
    "$HOME/incoming_weights"
    "$HOME/Downloads"
    /media
    /mnt
    /tmp
  )
fi

echo "== offline weight staging (no ModelScope / no pip) =="
echo "repo: $ROOT"

found_gguf=""
found_ser=""
for root in "${search_roots[@]}"; do
  [[ -d "$root" ]] || continue
  if [[ -z "$found_gguf" ]]; then
    hit="$(find "$root" -type f \( -iname '*q4_k_m*.gguf' -o -iname '*Q4_K_M*.gguf' \) 2>/dev/null | head -n 1 || true)"
    if [[ -n "$hit" ]]; then
      found_gguf="$hit"
    fi
  fi
  if [[ -z "$found_ser" ]]; then
    if [[ -d "$root/audio_ser" ]]; then
      found_ser="$root/audio_ser"
    else
      hit="$(find "$root" -type d \( -iname '*emotion2vec*' -o -iname 'audio_ser' \) 2>/dev/null | head -n 1 || true)"
      if [[ -n "$hit" ]]; then
        found_ser="$hit"
      fi
    fi
  fi
done

copied=0
if [[ -f "${DEST_LLM}/${GGUF_NAME}" ]]; then
  echo "GGUF already at ${DEST_LLM}/${GGUF_NAME}"
  ls -lh "${DEST_LLM}/${GGUF_NAME}"
  copied=1
elif [[ -n "$found_gguf" ]]; then
  echo "copy GGUF: $found_gguf"
  cp -f "$found_gguf" "${DEST_LLM}/${GGUF_NAME}"
  ls -lh "${DEST_LLM}/${GGUF_NAME}"
  copied=1
else
  echo "GGUF not found. On a Mac that already has the file, while on HW*:"
  echo "  scp qwen2.5-1.5b-instruct-q4_k_m.gguf cat@192.168.149.1:${DEST_LLM}/"
  echo "or copy it onto a USB stick and re-run this script with the mount path."
fi

if [[ -n "$(find "$DEST_SER" -type f 2>/dev/null | head -n 1 || true)" ]]; then
  echo "SER cache already has files in $DEST_SER"
  copied=1
elif [[ -n "$found_ser" ]]; then
  echo "copy SER tree: $found_ser -> $DEST_SER"
  mkdir -p "$DEST_SER"
  cp -a "$found_ser"/. "$DEST_SER"/
  copied=1
else
  echo "SER weights not found. Face-only loop still runs without them."
fi

echo
echo "== what still works on HW hotspot without extra weights =="
echo "  FaceExpression (already on the robot) + 778 phrases + stock .d6a"
echo "  Decide falls back to edge_rule if the GGUF is missing"
echo "  audio_emotion stays empty until SER files + funasr are present"
if [[ "$copied" -eq 0 ]]; then
  echo "nothing new staged. The live loop does not need this step to move."
  exit 0
fi
echo "staged. export EDGE_LLM_GGUF=${DEST_LLM}/${GGUF_NAME}"
echo "        export AUDIO_SER_CACHE=$DEST_SER"
echo "        export ENABLE_AUDIO_SER=1"
