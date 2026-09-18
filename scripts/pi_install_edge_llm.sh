#!/usr/bin/env bash
# Raspberry Pi: install on-device Decide stack (Qwen2.5-1.5B GGUF + llama-cpp).
# Does NOT install / require OpenAI.
set -euo pipefail

ROOT="${1:-/home/cat/RBM-project/momask-codes}"
cd "$ROOT"

echo "== power / thermal =="
vcgencmd get_throttled || true
vcgencmd measure_temp || true

if [[ ! -f venv_inference/bin/activate ]]; then
  echo "missing venv_inference at $ROOT" >&2
  exit 1
fi
# shellcheck disable=SC1091
source venv_inference/bin/activate
python -V

export DECIDE_BACKEND="${DECIDE_BACKEND:-edge_llm}"
export EDGE_LLM_N_THREADS="${EDGE_LLM_N_THREADS:-4}"
export EDGE_LLM_N_CTX="${EDGE_LLM_N_CTX:-2048}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "== pip: llama-cpp-python (edge LLM runtime) =="
pip install -U pip
# 4GB Pi: never parallel-compile (multi cc1plus OOMs). Single job only.
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
export MAX_JOBS="${MAX_JOBS:-1}"
echo "NOTE: CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL (keep MoMask stopped)."
CMAKE_ARGS="${CMAKE_ARGS:--DLLAMA_NATIVE=ON}" \
  pip install "llama-cpp-python>=0.2.90" || {
    echo "wheel missing — building llama-cpp-python single-threaded (20–60+ min)..."
    CMAKE_ARGS="-DLLAMA_NATIVE=ON" FORCE_CMAKE=1 \
      CMAKE_BUILD_PARALLEL_LEVEL=1 MAX_JOBS=1 \
      pip install --no-cache-dir "llama-cpp-python>=0.2.90"
  }
pip install -q modelscope huggingface_hub || true

echo "== download Qwen2.5-1.5B-Instruct Q4_K_M GGUF (ModelScope / HF mirror) =="
bash scripts/pi_download_qwen_gguf.sh "$ROOT"
export EDGE_LLM_GGUF="${EDGE_LLM_GGUF:-$ROOT/models/edge_llm/qwen2.5-1.5b-instruct-q4_k_m.gguf}"

echo "== import checks =="
python - <<'PY'
import importlib, os
from pathlib import Path
print("DECIDE_BACKEND", os.environ.get("DECIDE_BACKEND"))
print("EDGE_LLM_GGUF", os.environ.get("EDGE_LLM_GGUF"))
print("gguf exists", Path(os.environ["EDGE_LLM_GGUF"]).is_file())
importlib.import_module("llama_cpp")
print("ok llama_cpp")
print("pipeline", importlib.util.find_spec("pipeline"))
PY

echo "== smoke: mock perception + on-device Qwen Decide (no OpenAI) =="
python -m pipeline.run_mac_ab \
  --once --mock-perception --dry-run-b \
  --decide-backend edge_llm \
  --mock-emotion happy --mock-conf 0.72 \
  --transcript '你好，陪我一下'

echo "DONE"
echo "Persist env (optional):"
echo "  echo 'export DECIDE_BACKEND=edge_llm' >> ~/.bashrc"
echo "  echo 'export EDGE_LLM_GGUF=$EDGE_LLM_GGUF' >> ~/.bashrc"
echo "  echo 'export EDGE_LLM_N_THREADS=4' >> ~/.bashrc"
