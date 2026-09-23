#!/usr/bin/env bash
# One command for the whole robot setup.
# Run it twice from the Mac repo:
#   1) on a network that can reach GitHub (updates this checkout)
#   2) on the robot HW hotspot (copies files and rebinds WonderPi 人脸识别)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
HOST="${ROBOT_HOST:-pi@192.168.149.1}"
DEST="${ROBOT_DIR:-/home/pi/RBM-project/momask-codes}"
BRANCH="cursor/cloud-agent-1790042632488-ttam2"

echo "== update local code =="
git fetch origin "$BRANCH" || echo "fetch skipped (no internet). Using the files already in this folder."
if git rev-parse --verify "origin/$BRANCH" >/dev/null 2>&1; then
  git checkout -B "$BRANCH" "origin/$BRANCH"
fi

if ! python3 -c "import socket; socket.create_connection(('192.168.149.1', 22), 3).close()" >/dev/null 2>&1; then
  echo
  echo "代码已更新。现在把电脑改连到机器人的 HW 热点，然后把这一条命令再粘贴一次。"
  exit 0
fi

echo "== copy pipeline to $HOST =="
ssh "$HOST" "mkdir -p '$DEST/pipeline' '$DEST/scripts' '$DEST/pipeline_runs/latency'"
scp "$ROOT"/pipeline/*.py "$HOST:$DEST/pipeline/"
scp "$ROOT/scripts/pi_install_wonderpi_game.sh" "$HOST:$DEST/scripts/pi_install_wonderpi_game.sh"
scp "$ROOT/scripts/mac_deploy_wonderpi_all.sh" "$HOST:$DEST/scripts/mac_deploy_wonderpi_all.sh"

echo "== install WonderPi 人脸识别 button =="
ssh -t "$HOST" "sudo bash '$DEST/scripts/pi_install_wonderpi_game.sh'"

echo
echo "完成。手机连同一个 HW 热点，打开 WonderPi，进入「人脸识别」，停在这个页面。"
echo "对人脸微笑或皱眉，或者说「小幻小幻」后再说一句话。"
echo "延迟表在机器人上: $DEST/pipeline_runs/latency/wonderpi_latency.csv"
