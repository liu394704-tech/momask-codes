#!/usr/bin/env python3
"""4s-budget Mac/Pi hot path: FaceExpression + ring mic + rule Decide + resident MoMask.

Qwen 1.5B is intentionally not on this path. Whisper tiny runs only when the
mic snapshot has speech energy.

  源码/.venv-emotion-arm/bin/python -m pipeline.run_mac_fast_e2e --rounds 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.audio_mac import MicRingBuffer  # noqa: E402
from pipeline.percept_mac import capture_perception_mac, close_vision_session  # noqa: E402

BUDGET_S = 4.0
VAD_RMS = 250.0


def parse_args():
    p = argparse.ArgumentParser(description="Fast E2E: FaceExpression + ring mic + rules + MoMask")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--audio-sec", type=float, default=1.2)
    p.add_argument("--vad-rms", type=float, default=VAD_RMS)
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument(
        "--momask-python",
        default=os.environ.get(
            "MOMASK_PYTHON", "/opt/miniconda3/envs/momask_env/bin/python"
        ),
    )
    p.add_argument("--out-root", default=str(ROOT / "pipeline_runs"))
    p.add_argument("--budget-s", type=float, default=BUDGET_S)
    return p.parse_args()


def start_worker(py: str) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    err_path = Path("/tmp/mac_fast_worker.err")
    err_f = err_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        [py, "-u", "-m", "pipeline.mac_fast_worker"],
        cwd=str(ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=err_f,
        text=True,
        env=env,
    )


def worker_request(proc: subprocess.Popen, payload: dict, timeout_s: float = 120.0) -> dict:
    assert proc.stdin and proc.stdout
    proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
    proc.stdin.flush()
    t_end = time.time() + timeout_s
    while time.time() < t_end:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("worker died; see /tmp/mac_fast_worker.err")
        obj = json.loads(line)
        if obj.get("cmd") == "booted":
            continue
        return obj
    raise TimeoutError("worker timeout")


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(args.out_root) / ("exp_fast_%s" % stamp)
    exp_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "round", "t_vision_s", "t_mic_s", "t_perceive_wall_s",
        "mic_rms", "skip_asr", "t_asr_s", "t_decide_s", "t_momask_s",
        "t_total_s", "budget_ok", "vision_emotion", "vision_conf",
        "face_actions", "transcript", "intent", "action_prompt",
        "joints_path", "ok", "error",
    ]
    csv_path = exp_dir / "timing.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    print("[fast] starting Whisper+MoMask worker (no Qwen)  budget=%.1fs" % args.budget_s, flush=True)
    proc = start_worker(args.momask_python)
    assert proc.stdout

    ring = MicRingBuffer(ring_sec=max(3.0, args.audio_sec + 0.5))
    ring_ok = ring.start()
    print("[fast] mic ring start ok=%s err=%r" % (ring_ok, ring.error), flush=True)

    warmup_box = {}

    def _warmup_vision():
        t0 = time.perf_counter()
        perc = capture_perception_mac(session_id="warmup", camera_index=args.camera_index)
        warmup_box["perception"] = perc
        warmup_box["t_s"] = time.perf_counter() - t0

    th_warm = threading.Thread(target=_warmup_vision)
    th_warm.start()

    boot_line = proc.stdout.readline()
    th_warm.join()
    print(
        "[fast] vision warmup %.2fs calib=%s face=%s emo=%s"
        % (
            float(warmup_box.get("t_s") or 0),
            (warmup_box.get("perception").extras or {}).get("calibrated")
            if warmup_box.get("perception")
            else None,
            getattr(warmup_box.get("perception"), "face_found", None),
            getattr(warmup_box.get("perception"), "vision_emotion", None),
        ),
        flush=True,
    )

    if not boot_line:
        print("[fast] worker failed to boot; see /tmp/mac_fast_worker.err", flush=True)
        ring.stop()
        return 1
    boot = json.loads(boot_line)
    (exp_dir / "worker_boot.json").write_text(
        json.dumps(boot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[fast] worker boot %s" % boot, flush=True)
    print("[fast] look at camera; smile / speak a short phrase when a round starts", flush=True)

    try:
        for i in range(max(1, args.rounds)):
            session = "fast_%s" % uuid.uuid4().hex[:8]
            round_dir = exp_dir / session
            round_dir.mkdir(parents=True, exist_ok=True)
            print("\n===== round %d/%d =====" % (i + 1, args.rounds), flush=True)

            vision_box = {}
            audio_box = {
                "ok": False, "path": None, "err": "", "rms": 0.0,
                "t_mic_s": 0.0, "actual_sec": 0.0,
            }

            def vision_job():
                t0 = time.perf_counter()
                perc = capture_perception_mac(
                    session_id=session, camera_index=args.camera_index
                )
                vision_box["perception"] = perc
                vision_box["t_vision_s"] = time.perf_counter() - t0

            def mic_job():
                wav = str(round_dir / "mic.wav")
                t0 = time.perf_counter()
                if ring_ok and ring.filled_sec() >= max(0.4, args.audio_sec * 0.5):
                    ok, err, rms, actual = ring.snapshot_wav(wav, args.audio_sec)
                else:
                    from pipeline.audio_mac import record_mic_wav, wav_rms_int16

                    ok, err = record_mic_wav(wav, args.audio_sec)
                    rms = wav_rms_int16(wav) if ok else 0.0
                    actual = args.audio_sec if ok else 0.0
                audio_box["t_mic_s"] = time.perf_counter() - t0
                audio_box["ok"] = ok
                audio_box["err"] = err
                audio_box["path"] = wav if ok else None
                audio_box["rms"] = rms
                audio_box["actual_sec"] = actual

            t_wall0 = time.perf_counter()
            th_v = threading.Thread(target=vision_job)
            th_a = threading.Thread(target=mic_job)
            th_v.start()
            th_a.start()
            th_v.join()
            th_a.join()
            t_perceive = time.perf_counter() - t_wall0
            perc = vision_box.get("perception")
            if perc is None:
                print("[fast] vision failed", flush=True)
                continue

            skip_asr = (not audio_box.get("ok")) or float(audio_box.get("rms") or 0) < args.vad_rms
            npy = str(round_dir / "joints.npy")
            t_rest0 = time.perf_counter()
            out = worker_request(
                proc,
                {
                    "cmd": "infer",
                    "wav": audio_box.get("path") or "",
                    "skip_asr": skip_asr,
                    "perception": perc.to_dict(),
                    "out_npy": npy,
                },
            )
            t_total = t_perceive + (time.perf_counter() - t_rest0)
            budget_ok = t_total <= args.budget_s

            row = {
                "round": i + 1,
                "t_vision_s": round(float(vision_box.get("t_vision_s") or 0), 3),
                "t_mic_s": round(float(audio_box.get("t_mic_s") or 0), 3),
                "t_perceive_wall_s": round(t_perceive, 3),
                "mic_rms": round(float(audio_box.get("rms") or 0), 1),
                "skip_asr": skip_asr,
                "t_asr_s": out.get("t_asr_s"),
                "t_decide_s": out.get("t_decide_s"),
                "t_momask_s": out.get("t_momask_s"),
                "t_total_s": round(t_total, 3),
                "budget_ok": budget_ok,
                "vision_emotion": perc.vision_emotion,
                "vision_conf": round(float(perc.vision_conf or 0), 3),
                "face_actions": ",".join(perc.face_actions or []),
                "transcript": (out.get("transcript") or "")[:200],
                "intent": out.get("intent"),
                "action_prompt": (out.get("action_prompt") or "")[:240],
                "joints_path": out.get("joints_path") or "",
                "ok": out.get("ok"),
                "error": " ; ".join(
                    x for x in [
                        perc.error,
                        audio_box.get("err") if not audio_box.get("ok") else None,
                        out.get("asr_error"),
                        out.get("decide_error"),
                        out.get("momask_error"),
                    ] if x
                ),
            }
            with csv_path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            (round_dir / "round.json").write_text(
                json.dumps(
                    {"perception": perc.to_dict(), "backend": out, "timing": row},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                "[timing] vision=%.2fs mic=%.2fs wall=%.2fs asr=%.2fs%s rule=%.3fs momask=%.2fs "
                "total=%.2fs  %s"
                % (
                    row["t_vision_s"],
                    row["t_mic_s"],
                    row["t_perceive_wall_s"],
                    float(row["t_asr_s"] or 0),
                    "(skip)" if skip_asr else "",
                    float(row["t_decide_s"] or 0),
                    float(row["t_momask_s"] or 0),
                    row["t_total_s"],
                    "WITHIN %.1fs" % args.budget_s if budget_ok else "OVER %.1fs" % args.budget_s,
                ),
                flush=True,
            )
            print(
                "[result] face=%s emo=%s conf=%.2f actions=%s transcript=%r"
                % (
                    perc.face_found,
                    perc.vision_emotion,
                    perc.vision_conf,
                    perc.face_actions,
                    row["transcript"],
                ),
                flush=True,
            )
            print(
                "[result] intent=%s prompt=%s joints=%s"
                % (row["intent"], row["action_prompt"], row["joints_path"]),
                flush=True,
            )
    finally:
        try:
            if proc.stdin:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                proc.stdin.flush()
        except Exception:
            pass
        ring.stop()
        close_vision_session()
        try:
            proc.kill()
        except Exception:
            pass

    print("\n[fast] CSV", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
