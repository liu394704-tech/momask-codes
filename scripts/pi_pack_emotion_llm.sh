#!/usr/bin/env bash
# Pack the Pi emotion-LLM loop (no MoMask weights) for scp to the robot.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p dist
TGZ="$ROOT/dist/pipeline_emotion_llm_pi.tgz"
tar --disable-copyfile --no-xattrs -czf "$TGZ" \
  pipeline/ \
  tests/test_pi_emotion_llm.py \
  scripts/pi_install_edge_llm.sh \
  scripts/pi_download_qwen_gguf.sh \
  scripts/pi_run_emotion_llm.sh \
  源码/TonyPi/Functions/FaceExpression.py \
  源码/TonyPi/Functions/EmotionActionScheduler.py \
  源码/TonyPi/Functions/model/geometry_emotion_model.json
ls -lh "$TGZ"
echo "scp $TGZ cat@<pi-host>:/tmp/"
echo "On Pi: cd ~/RBM-project/momask-codes && tar -xzf /tmp/pipeline_emotion_llm_pi.tgz"
echo "       bash scripts/pi_run_emotion_llm.sh"
