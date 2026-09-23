#!/usr/bin/env bash
# Pack the Pi emotion-LLM loop (no MoMask weights) for scp to the robot.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p dist
TGZ="$ROOT/dist/pipeline_emotion_llm_pi.tgz"
PACK=(
  pipeline/
  tests/test_pi_emotion_llm.py
  tests/test_preset_phrases.py
  tests/test_tonypi_coords.py
  tests/test_offline_weights.py
  scripts/pi_install_edge_llm.sh
  scripts/pi_download_qwen_gguf.sh
  scripts/pi_run_emotion_llm.sh
  scripts/pi_install_audio_ser.sh
  scripts/pi_stage_weights_offline.sh
  scripts/pi_vnc_setup_and_run.sh
  scripts/pi_emotion_llm_service.sh
  scripts/pi_install_emotion_service.sh
  scripts/pi_install_wonderpi_game.sh
  tests/test_wonderpi_face_game.py
  scripts/mac_download_edge_weights.sh
)
for rel in \
  源码/TonyPi/Functions/FaceExpression.py \
  源码/TonyPi/Functions/EmotionActionScheduler.py \
  源码/TonyPi/Functions/model/geometry_emotion_model.json
do
  if [[ -f "$rel" ]]; then
    PACK+=("$rel")
  else
    echo "skip missing $rel (use stock /home/pi/TonyPi/Functions on the robot)"
  fi
done
TAR_FLAGS=(-czf "$TGZ")
if tar --help 2>/dev/null | grep -q disable-copyfile; then
  TAR_FLAGS=(--disable-copyfile --no-xattrs -czf "$TGZ")
fi
tar "${TAR_FLAGS[@]}" --exclude='__pycache__' --exclude='*.pyc' "${PACK[@]}"
ls -lh "$TGZ"
echo "scp $TGZ cat@<pi-host>:/tmp/"
echo "On Pi: cd ~/RBM-project/momask-codes && tar -xzf /tmp/pipeline_emotion_llm_pi.tgz"
echo "       bash scripts/pi_run_emotion_llm.sh"
