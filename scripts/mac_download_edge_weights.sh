#!/usr/bin/env bash
# Download edge weights on a computer that HAS internet, then copy them
# to the TonyPi over VNC / USB / HW* scp. Do not run this on the robot
# while it is only the HW AP — that network cannot reach the hubs.
#
#   bash scripts/mac_download_edge_weights.sh
#   bash scripts/mac_download_edge_weights.sh --gguf-only
#
# Output: dist/edge_weights/  (one folder you copy onto the robot)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${WEIGHTS_OUT:-$ROOT/dist/edge_weights}"
GGUF_NAME="qwen2.5-1.5b-instruct-q4_k_m.gguf"
GGUF_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --gguf-only) GGUF_ONLY=1 ;;
    --out=*) OUT="${arg#--out=}" ;;
  esac
done

mkdir -p "$OUT/edge_llm" "$OUT/audio_ser"
cd "$OUT"

echo "== download on this computer (needs internet, not HW*) =="
echo "folder: $OUT"

_have_file() {
  [[ -f "$1" ]] && [[ "$(wc -c < "$1")" -gt 1000000 ]]
}

if _have_file "$OUT/edge_llm/$GGUF_NAME"; then
  echo "GGUF already present"
  ls -lh "$OUT/edge_llm/$GGUF_NAME"
else
  echo "== Qwen2.5-1.5B Instruct Q4_K_M GGUF (~1 GB) =="
  tmp="$OUT/edge_llm/$GGUF_NAME.part"
  urls=(
    "https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    "https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
  )
  ok=0
  if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    echo "need curl or wget" >&2
    exit 1
  fi
  for url in "${urls[@]}"; do
    echo "try $url"
    if command -v curl >/dev/null 2>&1 && curl -L --fail --retry 3 -o "$tmp" "$url"; then
      ok=1
      break
    fi
    if command -v wget >/dev/null 2>&1 && wget -O "$tmp" "$url"; then
      ok=1
      break
    fi
  done
  if [[ "$ok" -eq 1 ]]; then
    mv "$tmp" "$OUT/edge_llm/$GGUF_NAME"
  else
    echo "direct download failed; try ModelScope Python snapshot" >&2
    python3 - <<PY
import glob, os, shutil, sys
out = r"""$OUT/edge_llm"""
target = os.path.join(out, "$GGUF_NAME")
os.makedirs(out, exist_ok=True)
try:
    from modelscope.hub.snapshot_download import snapshot_download
except Exception as exc:
    print("modelscope missing:", exc)
    print("pip install modelscope  then re-run, or download the GGUF in a browser")
    sys.exit(1)
path = snapshot_download(
    "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    cache_dir=os.path.join(out, ".ms_cache"),
    allow_patterns=["*q4_k_m*", "*Q4_K_M*"],
)
hits = glob.glob(os.path.join(path, "**", "*.gguf"), recursive=True)
if not hits:
    sys.exit("no gguf in ModelScope snapshot")
src = sorted(hits, key=lambda p: (("1.5" not in p.lower()), len(p)))[0]
shutil.copy2(src, target)
print("OK ModelScope", target)
PY
  fi
  ls -lh "$OUT/edge_llm/$GGUF_NAME"
fi

if [[ "$GGUF_ONLY" -eq 0 ]]; then
  echo "== optional emotion2vec+ seed (SER) =="
  if find "$OUT/audio_ser" -type f >/dev/null 2>&1 && [[ -n "$(find "$OUT/audio_ser" -type f | head -n 1)" ]]; then
    echo "SER cache already has files"
  else
    python3 - <<PY || echo "SER skipped (pip install modelscope funasr later, or copy only the GGUF)"
import os, shutil, sys
cache = r"""$OUT/audio_ser"""
os.makedirs(cache, exist_ok=True)
try:
    from modelscope.hub.snapshot_download import snapshot_download
except Exception as exc:
    print("modelscope missing, skip SER:", exc)
    sys.exit(2)
path = snapshot_download(
    "iic/emotion2vec_plus_seed",
    cache_dir=cache,
)
print("OK SER", path)
PY
  fi
fi

cat > "$OUT/COPY_TO_ROBOT.txt" <<EOF
Copy this whole folder onto the TonyPi, then stage it.

VNC (recommended while on HW*):
  1. Leave the computer on the internet until this download finishes.
  2. Switch Wi-Fi to the robot HW* AP (password hiwonder).
  3. Open VNC to 192.168.149.1 (user often pi / cat).
  4. In the Pi File Manager, copy
       edge_llm/$GGUF_NAME
     to
       ~/RBM-project/momask-codes/models/edge_llm/
     If you also have audio_ser/, copy that tree to
       ~/RBM-project/momask-codes/models/audio_ser/
  5. Or plug a USB stick: copy this folder onto the stick on the computer,
     plug the stick into the robot, in VNC open /media/pi/* or /media/cat/*.
  6. Pi terminal:
       cd ~/RBM-project/momask-codes
       bash scripts/pi_stage_weights_offline.sh ~/Desktop
       # or: bash scripts/pi_stage_weights_offline.sh /media/pi/USB

Same-network scp (Mac terminal, still on HW*):
  scp edge_llm/$GGUF_NAME cat@192.168.149.1:~/RBM-project/momask-codes/models/edge_llm/

Face emotion + 778 phrases do not need these files.
Qwen needs llama-cpp-python on the Pi; SER needs funasr. Those wheels
still want a one-time LAN install. Missing runtime = rule Decide / no audio_emotion.
EOF

echo
echo "DONE. Next: copy $OUT onto the robot via VNC File Manager or USB."
echo "Then on the Pi: bash scripts/pi_stage_weights_offline.sh <that-folder>"
cat "$OUT/COPY_TO_ROBOT.txt"
