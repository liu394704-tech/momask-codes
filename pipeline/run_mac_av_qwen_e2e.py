#!/usr/bin/env python3
"""Mac AV experiment: FaceMesh emotion + mic ASR + on-device Qwen + resident MoMask.

Use the emotion venv (mediapipe + PyAudio). Backend worker uses momask_env.

  源码/.venv-emotion-arm/bin/python -m pipeline.run_mac_av_qwen_e2e --rounds 2 --audio-sec 3
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

from pipeline.audio_mac import record_mic_wav  # noqa: E402
from pipeline.percept_mac import capture_perception_mac, close_vision_session  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Mac camera+mic -> Qwen -> MoMask timed E2E")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--audio-sec", type=float, default=3.0)
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument(
        "--momask-python",
        default="/opt/miniconda3/envs/momask_env/bin/python",
    )
    p.add_argument("--out-root", default=str(ROOT / "pipeline_runs"))
    return p.parse_args()


def start_worker(py: str) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("EDGE_LLM_N_THREADS", "6")
    err_path = Path("/tmp/mac_qwen_momask_worker.err")
    err_f = err_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        [py, "-u", "-m", "pipeline.mac_qwen_momask_worker"],
        cwd=str(ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=err_f,
        text=True,
        env=env,
    )


def worker_request(proc: subprocess.Popen, payload: dict, timeout_s: float = 180.0) -> dict:
    assert proc.stdin and proc.stdout
    proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
    proc.stdin.flush()
    t_end = time.time() + timeout_s
    while time.time() < t_end:
        line = proc.stdout.readline()
        if not line:
            err = (proc.stderr.read() if proc.stderr else "")[-800:]
            raise RuntimeError("worker died: %s" % err)
        obj = json.loads(line)
        if obj.get("cmd") == "booted":
            continue
        return obj
    raise TimeoutError("worker timeout")


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(args.out_root) / ("exp_av_qwen_%s" % stamp)
    exp_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "round", "t_vision_s", "t_record_s", "t_perceive_wall_s",
        "t_asr_s", "t_decide_s", "t_momask_s", "t_total_s",
        "vision_emotion", "vision_conf", "transcript",
        "intent", "action_prompt", "joints_path", "ok", "error",
    ]
    csv_path = exp_dir / "timing.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    print("[e2e] starting Qwen+Whisper+MoMask worker (%s)" % args.momask_python, flush=True)
    proc = start_worker(args.momask_python)
    assert proc.stdout
    boot_line = proc.stdout.readline()
    if not boot_line:
        print("[e2e] worker failed to boot; see /tmp/mac_qwen_momask_worker.err", flush=True)
        return 1
    boot = json.loads(boot_line)
    (exp_dir / "worker_boot.json").write_text(
        json.dumps(boot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[e2e] worker boot %s" % boot, flush=True)

    try:
        for i in range(max(1, args.rounds)):
            session = "avq_%s" % uuid.uuid4().hex[:8]
            round_dir = exp_dir / session
            round_dir.mkdir(parents=True, exist_ok=True)
            print("\n===== round %d/%d  look at camera, speak now (%.1fs) =====" % (
                i + 1, args.rounds, args.audio_sec
            ), flush=True)

            vision_box = {}
            audio_box = {"ok": False, "path": None, "err": ""}

            def vision_job():
                t0 = time.perf_counter()
                perc = capture_perception_mac(
                    session_id=session, camera_index=args.camera_index
                )
                vision_box["perception"] = perc
                vision_box["t_vision_s"] = time.perf_counter() - t0

            def mic_job():
                wav = round_dir / ("mic.wav")
                t0 = time.perf_counter()
                ok, err = record_mic_wav(str(wav), args.audio_sec)
                audio_box["t_record_s"] = time.perf_counter() - t0
                audio_box["ok"] = ok
                audio_box["err"] = err
                audio_box["path"] = str(wav) if ok else None

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
                print("[e2e] vision failed", flush=True)
                continue

            npy = str(round_dir / "joints.npy")
            t_rest0 = time.perf_counter()
            out = worker_request(
                proc,
                {
                    "cmd": "infer",
                    "wav": audio_box.get("path") or "",
                    "perception": perc.to_dict(),
                    "out_npy": npy,
                },
            )
            t_total = t_perceive + (time.perf_counter() - t_rest0)

            row = {
                "round": i + 1,
                "t_vision_s": round(float(vision_box.get("t_vision_s") or 0), 3),
                "t_record_s": round(float(audio_box.get("t_record_s") or 0), 3),
                "t_perceive_wall_s": round(t_perceive, 3),
                "t_asr_s": out.get("t_asr_s"),
                "t_decide_s": out.get("t_decide_s"),
                "t_momask_s": out.get("t_momask_s"),
                "t_total_s": round(t_total, 3),
                "vision_emotion": perc.vision_emotion,
                "vision_conf": round(float(perc.vision_conf or 0), 3),
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
                json.dumps({"perception": perc.to_dict(), "backend": out, "timing": row},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print("[timing] vision=%.2fs record=%.2fs wall=%.2fs asr=%.2fs qwen=%.2fs momask=%.2fs total=%.2fs" % (
                row["t_vision_s"], row["t_record_s"], row["t_perceive_wall_s"],
                float(row["t_asr_s"] or 0), float(row["t_decide_s"] or 0),
                float(row["t_momask_s"] or 0), row["t_total_s"],
            ), flush=True)
            print("[result] face=%s emo=%s conf=%.2f transcript=%r" % (
                perc.face_found, perc.vision_emotion, perc.vision_conf, row["transcript"],
            ), flush=True)
            print("[result] intent=%s prompt=%s joints=%s" % (
                row["intent"], row["action_prompt"], row["joints_path"],
            ), flush=True)
            if row["error"]:
                print("[result] error=%s" % row["error"], flush=True)
    finally:
        try:
            if proc.stdin:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                proc.stdin.flush()
        except Exception:
            pass
        close_vision_session()
        try:
            proc.kill()
        except Exception:
            pass

    print("\n[e2e] CSV", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
