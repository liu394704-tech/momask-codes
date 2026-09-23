#!/usr/bin/env bash
# Download Qwen2.5-1.5B-Instruct Q4_K_M GGUF for on-device Decide (China-friendly).
# Prefer ModelScope; fall back to HF mirror.
set -euo pipefail

ROOT="${1:-/home/cat/RBM-project/momask-codes}"
OUT_DIR="${ROOT}/models/edge_llm"
FILE="qwen2.5-1.5b-instruct-q4_k_m.gguf"
TARGET="${OUT_DIR}/${FILE}"

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

if [[ -f "$TARGET" ]]; then
  ls -lh "$TARGET"
  echo "already present: $TARGET"
  exit 0
fi

# HW* AP has no ModelScope route. Reuse a USB / scp drop if one is sitting around.
if [[ -x "${ROOT}/scripts/pi_stage_weights_offline.sh" ]]; then
  bash "${ROOT}/scripts/pi_stage_weights_offline.sh" "${WEIGHTS_SRC:-}" || true
  if [[ -f "$TARGET" ]]; then
    ls -lh "$TARGET"
    echo "staged offline: $TARGET"
    exit 0
  fi
fi

_can_reach_hub() {
  [[ "${OFFLINE:-0}" == "1" ]] && return 1
  if ! ip route show default 2>/dev/null | grep -q .; then
    return 1
  fi
  ping -c 1 -W 2 223.5.5.5 >/dev/null 2>&1 || ping -c 1 -W 2 1.1.1.1 >/dev/null 2>&1
}

if ! _can_reach_hub; then
  echo "no internet (HW hotspot / OFFLINE=1). Cannot download $FILE." >&2
  echo "Copy the GGUF onto the Pi over 192.168.149.1, then:" >&2
  echo "  bash scripts/pi_stage_weights_offline.sh /path/to/folder" >&2
  echo "Face + 778 phrases still run without this file." >&2
  exit 0
fi

echo "== download Qwen2.5-1.5B-Instruct Q4_K_M GGUF =="
echo "target: $TARGET"

# Activate venv if present
if [[ -f "${ROOT}/venv_inference/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/venv_inference/bin/activate"
fi

pip install -q -U "modelscope>=1.11" || true

export OUT_DIR="$OUT_DIR"
python - <<'PY'
import os, glob, shutil
out_dir = os.environ["OUT_DIR"]
target = os.path.join(out_dir, "qwen2.5-1.5b-instruct-q4_k_m.gguf")
os.makedirs(out_dir, exist_ok=True)

def find_gguf(root):
    pats = [
        "**/*q4_k_m*.gguf",
        "**/*Q4_K_M*.gguf",
        "**/*q4_k*.gguf",
    ]
    hits = []
    for p in pats:
        hits.extend(glob.glob(os.path.join(root, p), recursive=True))
    # prefer filenames containing 1.5b
    hits = sorted(hits, key=lambda x: (("1.5" not in x.lower()), len(x)))
    return hits[0] if hits else None

ok = False
# 1) ModelScope (mainland China)
try:
    from modelscope.hub.snapshot_download import snapshot_download
    # Official / community GGUF repos (try in order)
    candidates = [
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "LLM-Research/Qwen2.5-1.5B-Instruct-GGUF",
        "qwen/Qwen2.5-1.5B-Instruct-GGUF",
    ]
    for repo in candidates:
        try:
            print("ModelScope try:", repo)
            path = snapshot_download(
                repo,
                cache_dir=os.path.join(out_dir, ".ms_cache"),
                allow_patterns=["*q4_k_m*", "*Q4_K_M*", "*.gguf"],
            )
            hit = find_gguf(path)
            if hit:
                if os.path.abspath(hit) != os.path.abspath(target):
                    shutil.copy2(hit, target)
                print("OK ModelScope ->", target)
                ok = True
                break
        except Exception as e:
            print("ModelScope fail", repo, ":", e)
except Exception as e:
    print("modelscope unavailable:", e)

# 2) HF mirror (China)
if not ok:
    try:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from huggingface_hub import hf_hub_download
        print("HF-mirror download Qwen/Qwen2.5-1.5B-Instruct-GGUF ...")
        path = hf_hub_download(
            repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
            local_dir=out_dir,
            local_dir_use_symlinks=False,
        )
        if os.path.abspath(path) != os.path.abspath(target):
            shutil.copy2(path, target)
        print("OK HF-mirror ->", target)
        ok = True
    except Exception as e:
        print("HF-mirror fail:", e)

if not ok:
    raise SystemExit(
        "download failed. Manual: open ModelScope / hf-mirror, "
        "get qwen2.5-1.5b-instruct-q4_k_m.gguf into models/edge_llm/"
    )
print("size_bytes", os.path.getsize(target))
PY

ls -lh "$TARGET"
echo "DONE: export EDGE_LLM_GGUF=$TARGET"
