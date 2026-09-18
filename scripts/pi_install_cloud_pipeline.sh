#!/usr/bin/env bash
# DEPRECATED for Pi formal path. Use scripts/pi_install_edge_llm.sh instead.
# Kept only so old docs/commands do not hard-fail.
echo "[deprecated] Pi Decide no longer uses OpenAI."
echo "Run: bash scripts/pi_install_edge_llm.sh"
exec bash "$(cd "$(dirname "$0")" && pwd)/pi_install_edge_llm.sh" "$@"
