#!/usr/bin/env python3
"""
1) Call Zhipu or Aliyun DashScope with a structured-output system prompt.
2) Parse JSON -> text_prompt, motion_length.
3) Run gen_t2m.py as a subprocess (same env / GPU as your shell).

Environment:
  ZHIPU_API_KEY       — 智谱
  DASHSCOPE_API_KEY   — 阿里云 DashScope（控制台创建的 API Key）

Optional:
  ZHIPU_MODEL=glm-4-flash
  DASHSCOPE_MODEL=qwen-turbo

Example:
  export ZHIPU_API_KEY=xxx
  cd /path/to/momask-codes
  python scripts/chat_then_gen.py --provider zhipu --gpu_id 0 --ext llm_demo -- \\
      "用户很生气地挥手拒绝"
"""
from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from llm_bridge.schema import default_system_prompt
from llm_bridge.llm_client import call_zhipu, call_dashscope
from llm_bridge.json_utils import extract_json_object, validate_momask_plan


def main() -> None:
    ap = argparse.ArgumentParser(description="MLLM -> MoMask gen_t2m")
    ap.add_argument("--provider", choices=["zhipu", "dashscope"], required=True)
    ap.add_argument("--gpu_id", type=int, default=-1)
    ap.add_argument("--ext", default="llm_chat_gen")
    ap.add_argument("--lang", default="zh", help="system prompt: zh or en")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="only call LLM and print plan; do not run gen_t2m.py",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--image",
        default="",
        help="local image path; sent as base64 (Zhipu GLM-4V; DashScope as data URL, may depend on model)",
    )
    ap.add_argument(
        "--image-url",
        default="",
        help="public http(s) image URL for DashScope multimodal models",
    )
    ap.add_argument(
        "user_text",
        nargs=argparse.REMAINDER,
        help="user message; if empty, read stdin",
    )
    args = ap.parse_args()

    user = " ".join(args.user_text).strip()
    if not user:
        user = sys.stdin.read().strip()
    if not user:
        print("Error: empty user message.", file=sys.stderr)
        sys.exit(2)

    system = default_system_prompt(args.lang)

    image_b64 = None
    image_url = args.image_url.strip() or None
    if args.image:
        path = os.path.expanduser(args.image)
        with open(path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("ascii")
        if args.provider == "zhipu":
            image_b64 = b64
        else:
            # DashScope: try data URL; if your model rejects it, use OSS URL via --image-url
            ext = os.path.splitext(path)[1].lower()
            mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
            image_url = "data:%s;base64,%s" % (mime, b64)

    print("Calling %s ..." % args.provider, flush=True)
    if args.provider == "zhipu":
        raw = call_zhipu(system, user, image_base64=image_b64)
    else:
        raw = call_dashscope(system, user, image_url=image_url)

    print("--- raw LLM output ---\n", raw[:4000], "\n--- end ---\n", flush=True)

    try:
        data = extract_json_object(raw)
        text_prompt, motion_length = validate_momask_plan(data)
    except Exception as e:
        print("Failed to parse JSON plan:", e, file=sys.stderr)
        sys.exit(3)

    print("text_prompt:", text_prompt, flush=True)
    print("motion_length:", motion_length, flush=True)
    if data.get("notes_zh"):
        print("notes_zh:", data["notes_zh"], flush=True)

    if args.dry_run:
        print("Dry-run: skip gen_t2m.py", flush=True)
        return

    gen_py = os.path.join(ROOT, "gen_t2m.py")
    cmd = [
        sys.executable,
        gen_py,
        "--gpu_id",
        str(args.gpu_id),
        "--ext",
        args.ext,
        "--text_prompt",
        text_prompt,
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if motion_length > 0:
        cmd.extend(["--motion_length", str(motion_length)])

    print("Running:", " ".join(cmd[:6]), "... [--text_prompt ...]", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
