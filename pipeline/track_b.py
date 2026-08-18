#!/usr/bin/env python3
"""Track B: action_prompt -> MoMask joints (subprocess) or dry-run."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from .schemas import Decision, TrackBResult

ROOT = Path(__file__).resolve().parents[1]


def run_track_b(
    decision: Decision,
    out_dir: Path,
    dry_run: bool = False,
    gpu_id: int = -1,
    timeout_s: float = 180.0,
) -> TrackBResult:
    prompt = (decision.action_prompt or "").strip()
    if not prompt:
        return TrackBResult(
            ran=False, ok=False, prompt="", error="empty_action_prompt", dry_run=dry_run
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = decision.session_id or "sess"
    ext = "ab_%s_%d" % (session, int(time.time()))

    if dry_run:
        marker = out_dir / ("%s_DRYRUN.txt" % ext)
        marker.write_text(prompt + "\n", encoding="utf-8")
        return TrackBResult(
            ran=True,
            ok=True,
            joints_path=str(marker),
            prompt=prompt,
            elapsed_s=0.0,
            dry_run=True,
        )

    gen = ROOT / "gen_t2m.py"
    if not gen.exists():
        return TrackBResult(
            ran=True, ok=False, prompt=prompt, error="gen_t2m.py missing", dry_run=False
        )

    cmd = [
        sys.executable,
        str(gen),
        "--gpu_id",
        str(gpu_id),
        "--ext",
        ext,
        "--text_prompt",
        prompt,
        "--no_video_render",
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return TrackBResult(
            ran=True,
            ok=False,
            prompt=prompt,
            elapsed_s=time.time() - t0,
            error="momask_timeout",
            dry_run=False,
        )
    except Exception as exc:  # noqa: BLE001
        return TrackBResult(
            ran=True,
            ok=False,
            prompt=prompt,
            elapsed_s=time.time() - t0,
            error=str(exc),
            dry_run=False,
        )

    elapsed = time.time() - t0
    # MoMask writes under generation/<ext>/
    gen_dir = ROOT / "generation" / ext
    joints = list(gen_dir.glob("**/joints/**/*.npy")) if gen_dir.exists() else []
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "momask_failed")[-800:]
        return TrackBResult(
            ran=True, ok=False, prompt=prompt, elapsed_s=elapsed, error=err, dry_run=False
        )
    joints_path = str(joints[0]) if joints else str(gen_dir)
    # Copy pointer into pipeline out_dir
    note = out_dir / ("%s_joints_path.txt" % ext)
    note.write_text(joints_path + "\n" + prompt + "\n", encoding="utf-8")
    return TrackBResult(
        ran=True,
        ok=True,
        joints_path=joints_path,
        prompt=prompt,
        elapsed_s=elapsed,
        dry_run=False,
    )
