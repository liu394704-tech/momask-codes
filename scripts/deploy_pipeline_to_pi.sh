#!/usr/bin/env bash
# Mac -> Raspberry Pi: push cloud pipeline package and run install smoke.
# Requires: sshpass, reachable ssh (cat@host).
set -euo pipefail

HOST="${PI_HOST:-10.194.36.60}"
USER_NAME="${PI_USER:-cat}"
PASS="${PI_PASS:-}"
REMOTE_ROOT="${PI_ROOT:-/home/cat/RBM-project/momask-codes}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TGZ="$ROOT/dist/pipeline_cloud_pi.tgz"

if [[ ! -f "$TGZ" ]]; then
  echo "missing $TGZ — rebuild first" >&2
  exit 1
fi
if [[ -z "$PASS" ]]; then
  echo "Set PI_PASS=... (or export it)" >&2
  exit 1
fi

export SSHPASS="$PASS"
SSH=(sshpass -e ssh -o StrictHostKeyChecking=accept-new -o PreferredAuthentications=password -o PubkeyAuthentication=no)
SCP=(sshpass -e scp -o StrictHostKeyChecking=accept-new -o PreferredAuthentications=password -o PubkeyAuthentication=no)

echo "== health =="
"${SSH[@]}" "${USER_NAME}@${HOST}" 'vcgencmd get_throttled; vcgencmd measure_temp; hostname; python3 -V'

echo "== upload =="
"${SCP[@]}" "$TGZ" "${USER_NAME}@${HOST}:/tmp/pipeline_cloud_pi.tgz"

echo "== unpack + install =="
"${SSH[@]}" "${USER_NAME}@${HOST}" bash -s <<EOF
set -euo pipefail
cd "$REMOTE_ROOT"
tar -xzf /tmp/pipeline_cloud_pi.tgz
bash scripts/pi_install_cloud_pipeline.sh "$REMOTE_ROOT"
EOF

echo "OK — on Pi, export API keys then:"
echo "  source venv_inference/bin/activate"
echo "  python -m pipeline.run_mac_ab --once --mock-perception --dry-run-b"
