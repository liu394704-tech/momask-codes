#!/usr/bin/env python3
"""Mac cloud E2E timed test: camera + mic -> LLM prompt -> MoMask (+ timing CSV).

Normal command (after exporting API key):

  export OPENAI_API_KEY='...'
  export OPENAI_BASE_URL='https://www.dmxapi.cn/v1'
  export OPENAI_MODEL='gpt-4o-mini'
  export WHISPER_MODEL='gpt-4o-transcribe'

  # Use emotion venv for mediapipe; MoMask via momask_env python:
  源码/.venv-emotion-arm/bin/python -m pipeline.run_mac_e2e_timed \\
    --once --mic --momask \\
    --momask-python /opt/miniconda3/envs/momask_env/bin/python \\
    --gpu-id -1

  # Faster: stop before MoMask
  ... --once --mic --dry-run-b
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.arbiter import Arbiter, ArbiterConfig  # noqa: E402
from pipeline.audio_mac import record_mic_wav, whisper_transcribe  # noqa: E402
from pipeline.decide_cloud import cloud_decide, mock_decide  # noqa: E402
from pipeline.percept_mac import capture_perception_mac, mock_perception  # noqa: E402
from pipeline.schemas import Mode, Perception  # noqa: E402
from pipeline.track_a import run_track_a  # noqa: E402
from pipeline.track_b import run_track_b  # noqa: E402

TIMING_FIELDS = [
    "round",
    "session_id",
    "t_vision_s",
    "t_record_s",
    "t_asr_s",
    "t_perceive_wall_s",
    "t_decide_s",
    "t_track_a_s",
    "t_momask_s",
    "t_total_s",
    "vision_emotion",
    "vision_conf",
    "transcript",
    "action_prompt",
    "action_group",
    "effective_mode",
    "degraded",
    "degrade_reason",
    "joints_path",
    "ok",
    "error",
]


def parse_args():
    p = argparse.ArgumentParser(description="Mac timed AV -> LLM -> MoMask E2E")
    p.add_argument("--mode", default="A_parallel_B",
                   choices=["A_only", "B_only", "A_parallel_B"])
    p.add_argument("--once", action="store_true")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--mic", action="store_true", help="record microphone each round")
    p.add_argument("--audio-sec", type=float, default=4.0)
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--mock-perception", action="store_true")
    p.add_argument("--mock-decide", action="store_true")
    p.add_argument("--dry-run-b", action="store_true")
    p.add_argument("--momask", action="store_true",
                   help="run real MoMask (ignored if --dry-run-b)")
    p.add_argument("--gpu-id", type=int, default=-1)
    p.add_argument("--momask-python", default="",
                   help="python for gen_t2m.py (e.g. momask_env)")
    p.add_argument("--confidence-min", type=float, default=0.45)
    p.add_argument("--whisper-model", default=os.environ.get("WHISPER_MODEL", "gpt-4o-transcribe"))
    p.add_argument("--out-root", default=str(ROOT / "pipeline_runs"))
    return p.parse_args()


def _parallel_perceive(
    session_id: str,
    args,
    tmp_dir: Path,
) -> Dict[str, Any]:
    """Run camera emotion and mic record in parallel; then ASR."""
    vision_box: Dict[str, Any] = {}
    audio_box: Dict[str, Any] = {"ok": False, "path": None, "err": ""}

    def vision_job():
        t0 = time.time()
        if args.mock_perception:
            perc = mock_perception(session_id, emotion="happy", conf=0.7)
        else:
            perc = capture_perception_mac(
                session_id=session_id,
                camera_index=args.camera_index,
            )
        vision_box["perception"] = perc
        vision_box["t_vision_s"] = time.time() - t0

    def mic_job():
        if not args.mic:
            audio_box["ok"] = True
            audio_box["t_record_s"] = 0.0
            return
        wav = tmp_dir / ("mic_%s.wav" % session_id)
        t0 = time.time()
        ok, err = record_mic_wav(str(wav), args.audio_sec)
        audio_box["t_record_s"] = time.time() - t0
        audio_box["ok"] = ok
        audio_box["err"] = err
        audio_box["path"] = str(wav) if ok else None

    t_wall0 = time.time()
    th_v = threading.Thread(target=vision_job)
    th_a = threading.Thread(target=mic_job)
    th_v.start()
    th_a.start()
    th_v.join()
    th_a.join()
    t_perceive_wall = time.time() - t_wall0

    perception: Perception = vision_box.get("perception") or Perception(
        session_id=session_id, ts_ms=int(time.time() * 1000), ok=False, error="no_vision"
    )
    t_asr = 0.0
    transcript = ""
    asr_err = None
    if args.mic and audio_box.get("ok") and audio_box.get("path"):
        t1 = time.time()
        transcript, asr_err = whisper_transcribe(
            audio_box["path"], model=args.whisper_model
        )
        t_asr = time.time() - t1
        if asr_err:
            perception.extras["asr_error"] = asr_err
        else:
            perception.transcript = transcript or ""
    elif args.mic and not audio_box.get("ok"):
        perception.extras["mic_error"] = audio_box.get("err") or "mic_failed"

    return {
        "perception": perception,
        "t_vision_s": float(vision_box.get("t_vision_s") or 0.0),
        "t_record_s": float(audio_box.get("t_record_s") or 0.0),
        "t_asr_s": t_asr,
        "t_perceive_wall_s": t_perceive_wall,
        "transcript": perception.transcript,
        "asr_error": asr_err,
        "mic_error": audio_box.get("err") if args.mic and not audio_box.get("ok") else None,
    }


def main():
    args = parse_args()
    if args.momask and args.dry_run_b:
        print("[warn] both --momask and --dry-run-b set; using dry-run-b")
    dry_b = args.dry_run_b or (not args.momask)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(args.out_root) / ("exp_e2e_%s" % stamp)
    exp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = exp_dir / "timing.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=TIMING_FIELDS).writeheader()

    arbiter = Arbiter(ArbiterConfig(mode=Mode(args.mode), confidence_min=args.confidence_min))
    n = 1 if args.once else max(1, args.rounds)
    momask_py = args.momask_python.strip() or None

    print("[e2e] exp_dir=%s" % exp_dir)
    print("[e2e] mode=%s mic=%s dry_run_b=%s momask_py=%s model=%s whisper=%s" % (
        args.mode, args.mic, dry_b, momask_py or sys.executable,
        os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), args.whisper_model,
    ))

    for i in range(n):
        t_total0 = time.time()
        session = "e2e_%s" % uuid.uuid4().hex[:8]
        round_dir = exp_dir / session
        round_dir.mkdir(parents=True, exist_ok=True)

        print("\n===== round %d/%d session=%s =====" % (i + 1, n, session), flush=True)
        print("[e2e] perceiving (camera // mic)... speak now if --mic", flush=True)
        perc_pack = _parallel_perceive(session, args, round_dir)
        perception = perc_pack["perception"]

        t_dec0 = time.time()
        if args.mock_decide or not os.environ.get("OPENAI_API_KEY", "").strip():
            if not args.mock_decide:
                print("[e2e] no OPENAI_API_KEY -> mock_decide", flush=True)
            decision = mock_decide(perception)
        else:
            decision = cloud_decide(perception)
        t_decide = time.time() - t_dec0

        t_a = 0.0
        t_b = 0.0
        track_a_holder = {}
        track_b_holder = {}

        def run_a(dec):
            nonlocal t_a
            t0 = time.time()
            res = run_track_a(dec, simulate=True, execute_robot=False)
            t_a = time.time() - t0
            track_a_holder["r"] = res
            return res

        def run_b(dec):
            nonlocal t_b
            t0 = time.time()
            res = run_track_b(
                dec,
                out_dir=round_dir,
                dry_run=dry_b,
                gpu_id=args.gpu_id,
                python_executable=momask_py,
            )
            t_b = time.time() - t0
            # prefer momask elapsed if provided
            if res.elapsed_s:
                t_b = res.elapsed_s
            track_b_holder["r"] = res
            return res

        result = arbiter.run_round(perception, decision, run_a=run_a, run_b=run_b)
        t_total = time.time() - t_total0

        ok = bool(decision.ok) and (
            result.track_b is None or result.track_b.ok or result.effective_mode == "A_only"
        )
        err_parts = []
        if perception.error:
            err_parts.append("vision:" + str(perception.error))
        if perc_pack.get("mic_error"):
            err_parts.append("mic:" + str(perc_pack["mic_error"]))
        if perc_pack.get("asr_error"):
            err_parts.append("asr:" + str(perc_pack["asr_error"]))
        if not decision.ok:
            err_parts.append("decide:" + str(decision.error))
        if result.track_b and result.track_b.ran and not result.track_b.ok:
            err_parts.append("momask:" + str(result.track_b.error))

        row = {
            "round": i + 1,
            "session_id": session,
            "t_vision_s": round(perc_pack["t_vision_s"], 3),
            "t_record_s": round(perc_pack["t_record_s"], 3),
            "t_asr_s": round(perc_pack["t_asr_s"], 3),
            "t_perceive_wall_s": round(perc_pack["t_perceive_wall_s"], 3),
            "t_decide_s": round(t_decide, 3),
            "t_track_a_s": round(t_a, 3),
            "t_momask_s": round(t_b, 3),
            "t_total_s": round(t_total, 3),
            "vision_emotion": perception.vision_emotion,
            "vision_conf": round(float(perception.vision_conf or 0.0), 3),
            "transcript": (perception.transcript or "")[:200],
            "action_prompt": (decision.action_prompt or "")[:240],
            "action_group": "|".join(decision.action_group or []),
            "effective_mode": result.effective_mode,
            "degraded": result.degraded,
            "degrade_reason": result.degrade_reason or "",
            "joints_path": (result.track_b.joints_path if result.track_b else "") or "",
            "ok": ok,
            "error": " ; ".join(err_parts),
        }
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=TIMING_FIELDS).writerow(row)

        payload = result.to_dict()
        payload["timing"] = {k: row[k] for k in TIMING_FIELDS if k.startswith("t_")}
        (round_dir / "round.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print("[timing] vision=%.2fs record=%.2fs asr=%.2fs perceive_wall=%.2fs decide=%.2fs A=%.2fs B=%.2fs total=%.2fs" % (
            row["t_vision_s"], row["t_record_s"], row["t_asr_s"], row["t_perceive_wall_s"],
            row["t_decide_s"], row["t_track_a_s"], row["t_momask_s"], row["t_total_s"],
        ), flush=True)
        print("[result] emo=%s conf=%.2f transcript=%r" % (
            perception.vision_emotion, perception.vision_conf, perception.transcript,
        ), flush=True)
        print("[result] prompt=%s" % (decision.action_prompt[:140],), flush=True)
        print("[result] mode=%s degraded=%s joints=%s ok=%s" % (
            result.effective_mode, result.degraded,
            row["joints_path"], ok,
        ), flush=True)
        if row["error"]:
            print("[result] error=%s" % row["error"], flush=True)

    print("\n[e2e] timing CSV:", csv_path)
    print("[e2e] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
