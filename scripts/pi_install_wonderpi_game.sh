#!/usr/bin/env bash
# Point WonderPi's existing 「人脸识别」button (function 6) at the phrase loop.
# The phone app has no free slot for a new button. Other games stay as they are.
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  exec sudo -E bash "$0" "$@"
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TONYPI="${TONYPI_ROOT:-/home/pi/TonyPi}"
if [[ ! -d "$TONYPI/Functions" && -d /home/cat/TonyPi/Functions ]]; then
  TONYPI=/home/cat/TonyPi
fi
RUNNING="$TONYPI/Functions/Running.py"
SHIM="$TONYPI/Functions/EmotionPhrase.py"

if [[ ! -f "$RUNNING" ]]; then
  echo "missing $RUNNING" >&2
  exit 1
fi
if [[ ! -f "$ROOT/pipeline/wonderpi_face_game.py" ]]; then
  echo "missing $ROOT/pipeline/wonderpi_face_game.py" >&2
  exit 1
fi

python3 - "$RUNNING" "$ROOT" "$SHIM" <<'PY'
import pathlib, sys
running, root, shim = sys.argv[1:]
path = pathlib.Path(running)
text = path.read_text(encoding="utf-8")
backup = path.with_suffix(".py.bak_emotion_phrase")
if not backup.exists():
    backup.write_text(text, encoding="utf-8")
if "Functions.EmotionPhrase" not in text:
    needle = "import Functions.FaceDetect as FaceDetect"
    if needle not in text:
        sys.exit("Running.py has no FaceDetect import; not patching")
    text = text.replace(
        needle,
        needle + "\nimport Functions.EmotionPhrase as EmotionPhrase",
        1,
    )
old = "6: FaceDetect,"
new = "6: EmotionPhrase,  # was FaceDetect: multimodal phrase loop"
if old not in text and "6: EmotionPhrase" not in text:
    sys.exit("Running.py has no function 6 FaceDetect entry")
text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
pathlib.Path(shim).write_text(
    "import sys\n"
    "sys.path.insert(0, %r)\n"
    "try:\n"
    "    from pipeline.wonderpi_face_game import init, start, stop, exit, run\n"
    "except Exception as exc:\n"
    "    print('EmotionPhrase import failed:', exc)\n"
    "    def init():\n"
    "        print('EmotionPhrase fallback init')\n"
    "    def start():\n"
    "        print('EmotionPhrase fallback start')\n"
    "    def stop():\n"
    "        print('EmotionPhrase fallback stop')\n"
    "    def exit():\n"
    "        print('EmotionPhrase fallback exit')\n"
    "    def run(img):\n"
    "        return img\n" % root,
    encoding="utf-8",
)
print("patched", running)
print("shim", shim)
print("backup", backup)
PY

systemctl disable --now tonypi-emotion-llm.service 2>/dev/null || true
systemctl stop tonypi.service 2>/dev/null || systemctl stop tonypi 2>/dev/null || true
# The earlier terminal loop keeps the camera, which leaves WonderPi on loading.
pkill -f "pipeline.run_pi_emotion_llm" 2>/dev/null || true
pkill -f "pipeline.bench_pi_latency" 2>/dev/null || true
sleep 1
systemctl enable tonypi.service 2>/dev/null || true
systemctl restart tonypi.service 2>/dev/null || systemctl restart tonypi
sleep 2
if systemctl is-active --quiet tonypi.service || systemctl is-active --quiet tonypi; then
  echo "tonypi is running. WonderPi should show the camera, then enter 人脸识别."
else
  echo "tonypi did not stay up. Last log:" >&2
  journalctl -u tonypi -n 40 --no-pager >&2 || true
  exit 1
fi
echo "Stay on that screen. The picture must appear before the robot waves."
echo "Each phrase is logged with timings at $ROOT/pipeline_runs/latency/wonderpi_latency.csv"
