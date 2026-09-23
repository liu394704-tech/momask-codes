#!/usr/bin/env bash
# Install a boot service so the emotion loop runs without VNC.
# WonderPi is not a launcher for this loop. One install, then power-on is enough.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_NAME="tonypi-emotion-llm.service"
UNIT_PATH="/etc/systemd/system/${UNIT_NAME}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "re-run: sudo bash $0" >&2
  exit 1
fi

chmod +x "$ROOT/scripts/pi_emotion_llm_service.sh"

cat > "$UNIT_PATH" <<EOF
[Unit]
Description=TonyPi face emotion to ActionGroup loop
After=network.target

[Service]
Type=simple
WorkingDirectory=${ROOT}
Environment=PYTHONUNBUFFERED=1
ExecStart=${ROOT}/scripts/pi_emotion_llm_service.sh
ExecStopPost=-/bin/systemctl start tonypi
Restart=on-failure
RestartSec=8

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$UNIT_NAME"
systemctl restart "$UNIT_NAME"
systemctl --no-pager --full status "$UNIT_NAME" || true
echo
echo "Installed $UNIT_PATH"
echo "It keeps running after you close WonderPi / VNC."
echo "Logs: journalctl -u $UNIT_NAME -f"
echo "Back to the official WonderPi app:"
echo "  sudo systemctl disable --now $UNIT_NAME"
echo "  sudo systemctl start tonypi"
