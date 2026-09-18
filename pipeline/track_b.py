#!/usr/bin/env python3
"""Track B: action_prompt -> MoMask joints.

Default (interactive): in-process resident models, joints npy only. No BVH IK.
First call in a process pays load (~9–10 s); later calls are generate-only (~2.2 s).

Legacy: set MOMASK_USE_SUBPROCESS=1 or pass a different python_executable to
spawn gen_t2m.py (cold start + IK, ~20–30 s). Do not use that on the robot loop.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from .schemas import Decision, TrackBResult

ROOT = Path(__file__).resolve().parents[1]


def _use_subprocess(python_executable: Optional[str]) -> bool:
    if os.environ.get("MOMASK_USE_SUBPROCESS", "").strip() in ("1", "true", "yes"):
        return True
    if python_executable:
        return os.path.realpath(python_executable) != os.path.realpath(sys.executable)
    return False


def run_track_b(
    decision: Decision,
    out_dir: Path,
    dry_run: bool = False,
    gpu_id: int = -1,
    timeout_s: float = 180.0,
    python_executable: Optional[str] = None,
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

    if not _use_subprocess(python_executable):
        return _run_resident(decision, out_dir, ext, prompt)

    return _run_subprocess(
        prompt, out_dir, ext, gpu_id, timeout_s, python_executable
    )


def _run_resident(
    decision: Decision, out_dir: Path, ext: str, prompt: str
) -> TrackBResult:
    from .momask_runtime import generate

    npy = out_dir / ("%s_joints.npy" % ext)
    t0 = time.perf_counter()
    try:
        path, load_s, gen_s, _t = generate(
            prompt,
            npy,
            motion_length_hint=int(decision.motion_length_hint or 0),
        )
    except Exception as exc:  # noqa: BLE001
        return TrackBResult(
            ran=True,
            ok=False,
            prompt=prompt,
            elapsed_s=time.perf_counter() - t0,
            error=str(exc)[:800],
            dry_run=False,
        )
    note = out_dir / ("%s_joints_path.txt" % ext)
    note.write_text(
        "%s\n%s\nload_s=%.3f gen_s=%.3f\n" % (path, prompt, load_s, gen_s),
        encoding="utf-8",
    )
    return TrackBResult(
        ran=True,
        ok=True,
        joints_path=str(path),
        prompt=prompt,
        elapsed_s=gen_s,
        load_s=load_s,
        gen_s=gen_s,
        dry_run=False,
    )


def _run_subprocess(
    prompt: str,
    out_dir: Path,
    ext: str,
    gpu_id: int,
    timeout_s: float,
    python_executable: Optional[str],
) -> TrackBResult:
    gen = ROOT / "gen_t2m.py"
    if not gen.exists():
        return TrackBResult(
            ran=True, ok=False, prompt=prompt, error="gen_t2m.py missing", dry_run=False
        )

    py = python_executable or sys.executable
    cmd = [
        py,
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
    gen_dir = ROOT / "generation" / ext
    joints = list(gen_dir.glob("**/joints/**/*.npy")) if gen_dir.exists() else []
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "momask_failed")[-800:]
        return TrackBResult(
            ran=True, ok=False, prompt=prompt, elapsed_s=elapsed, error=err, dry_run=False
        )
    joints_path = str(joints[0]) if joints else str(gen_dir)
    note = out_dir / ("%s_joints_path.txt" % ext)
    note.write_text(joints_path + "\n" + prompt + "\n", encoding="utf-8")
    return TrackBResult(
        ran=True,
        ok=True,
        joints_path=joints_path,
        prompt=prompt,
        elapsed_s=elapsed,
        gen_s=elapsed,
        dry_run=False,
    )
