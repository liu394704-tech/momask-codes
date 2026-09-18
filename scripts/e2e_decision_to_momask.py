#!/usr/bin/env python3
"""把决策冒烟结果转成 MoMask bench 输入，并可一键调用 bench_pi_gen。

示例：
  # 仅从 decision CSV 导出 prompt 文件
  python scripts/e2e_decision_to_momask.py --export-only

  # 导出并本地/树莓派跑生成（需在项目根目录、已装好依赖）
  python scripts/e2e_decision_to_momask.py --run-bench --save-joints --threads 4
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "experiment/e2e_spec/decision_smoke_results.csv"
DEFAULT_PROMPTS = ROOT / "experiment/e2e_spec/decision_action_prompts_20.txt"
DEFAULT_OUT = ROOT / "experiment/e2e_spec/momask_from_decision/bench_results.csv"


def export_prompts(csv_path: Path, out_path: Path) -> int:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    lines = []
    for r in rows:
        emo = (r.get("emotion") or "unknown").strip()
        prompt = (r.get("action_prompt") or "").strip()
        if not prompt:
            continue
        lines.append(f"{emo}|{prompt}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision-csv", default=str(DEFAULT_CSV))
    ap.add_argument("--prompts-out", default=str(DEFAULT_PROMPTS))
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--run-bench", action="store_true")
    ap.add_argument("--out-csv", default=str(DEFAULT_OUT))
    ap.add_argument("--time-steps", type=int, default=18)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--save-joints", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    n = export_prompts(Path(args.decision_csv), Path(args.prompts_out))
    print(f"[export] {n} prompts -> {args.prompts_out}")
    if args.export_only or not args.run_bench:
        if not args.run_bench:
            print("提示: 加 --run-bench 可继续调用 scripts/bench_pi_gen.py")
        return

    cmd = [
        sys.executable, str(ROOT / "scripts/bench_pi_gen.py"),
        "--text_path", args.prompts_out,
        "--out_csv", args.out_csv,
        "--time_steps", str(args.time_steps),
        "--warmup", str(args.warmup),
        "--threads", str(args.threads),
    ]
    if args.limit > 0:
        cmd += ["--limit", str(args.limit)]
    if args.save_joints:
        cmd.append("--save_joints")
    print("[run]", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
