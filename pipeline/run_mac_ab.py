#!/usr/bin/env python3
"""Mac A/B pipeline entry (Phase 1 default: A_parallel_B).

Baseline-safe modes:
  # 1) No camera / no API — full mock round (always safe)
  python -m pipeline.run_mac_ab --once --mock-perception --mock-decide --dry-run-b

  # 2) Local face geometry + mock decide (needs 源码 mediapipe env + camera)
  python -m pipeline.run_mac_ab --once --mock-decide --dry-run-b

  # 3) Local face + cloud decide (NEED KEY from you)
  export OPENAI_API_KEY='sk-...'
  export OPENAI_BASE_URL='https://your-gateway/v1'
  export OPENAI_MODEL='gpt-4o-mini'
  python -m pipeline.run_mac_ab --once --dry-run-b

  # 4) Same + real MoMask joints (needs momask_env + checkpoints)
  python -m pipeline.run_mac_ab --once --gpu-id -1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.arbiter import Arbiter, ArbiterConfig  # noqa: E402
from pipeline.decide_cloud import cloud_decide, mock_decide  # noqa: E402
from pipeline.percept_mac import capture_perception_mac, mock_perception  # noqa: E402
from pipeline.track_a import run_track_a  # noqa: E402
from pipeline.track_b import run_track_b  # noqa: E402
from pipeline.schemas import Mode  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Mac A/B arbiter pipeline")
    p.add_argument(
        "--mode",
        default=os.environ.get("AB_MODE", "A_parallel_B"),
        choices=["A_only", "B_only", "A_parallel_B"],
    )
    p.add_argument("--once", action="store_true", help="run a single round and exit")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--mock-perception", action="store_true")
    p.add_argument("--mock-emotion", default="happy")
    p.add_argument("--mock-conf", type=float, default=0.72)
    p.add_argument("--mock-decide", action="store_true")
    p.add_argument("--dry-run-b", action="store_true", help="do not call MoMask")
    p.add_argument("--gpu-id", type=int, default=int(os.environ.get("MOMASK_GPU_ID", "-1")))
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--transcript", default="")
    p.add_argument("--confidence-min", type=float, default=0.45)
    p.add_argument(
        "--out-dir",
        default=str(ROOT / "pipeline_runs"),
        help="per-round JSON logs (gitignored)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = Mode(args.mode)
    arbiter = Arbiter(
        ArbiterConfig(mode=mode, confidence_min=args.confidence_min)
    )

    n = 1 if args.once else max(1, args.rounds)
    print("[mac-ab] mode=%s mock_perception=%s mock_decide=%s dry_run_b=%s" % (
        mode.value, args.mock_perception, args.mock_decide, args.dry_run_b
    ))

    for i in range(n):
        session = "mac_%s" % uuid.uuid4().hex[:8]
        if args.mock_perception:
            perception = mock_perception(
                session, emotion=args.mock_emotion, conf=args.mock_conf
            )
            if args.transcript:
                perception.transcript = args.transcript
        else:
            perception = capture_perception_mac(
                session_id=session,
                camera_index=args.camera_index,
                transcript=args.transcript,
            )

        if args.mock_decide or not os.environ.get("OPENAI_API_KEY", "").strip():
            if not args.mock_decide and not os.environ.get("OPENAI_API_KEY", "").strip():
                print("[mac-ab] OPENAI_API_KEY missing -> using mock_decide")
            decision = mock_decide(perception)
        else:
            decision = cloud_decide(perception)

        def _a(dec):
            return run_track_a(dec, simulate=True, execute_robot=False)

        def _b(dec):
            return run_track_b(
                dec,
                out_dir=out_dir / session,
                dry_run=args.dry_run_b,
                gpu_id=args.gpu_id,
            )

        result = arbiter.run_round(perception, decision, run_a=_a, run_b=_b)
        path = out_dir / ("%s_round.json" % session)
        path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("----- round %d/%d -----" % (i + 1, n))
        print("perception: face=%s emo=%s conf=%.2f transcript=%r" % (
            perception.face_found, perception.vision_emotion, perception.vision_conf,
            perception.transcript,
        ))
        print("decision: emo=%s intent=%s conf=%.2f fallback=%s group=%s" % (
            decision.emotion, decision.intent, decision.confidence, decision.fallback,
            decision.action_group,
        ))
        print("prompt: %s" % (decision.action_prompt[:120] + ("..." if len(decision.action_prompt) > 120 else "")))
        print("arbiter: configured=%s effective=%s degraded=%s reason=%s" % (
            result.mode, result.effective_mode, result.degraded, result.degrade_reason,
        ))
        if result.track_a:
            print("track_a: action=%s intensity=%s simulated=%s (%s)" % (
                result.track_a.action, result.track_a.intensity,
                result.track_a.simulated, result.track_a.detail,
            ))
        if result.track_b:
            print("track_b: ran=%s ok=%s dry_run=%s joints=%s err=%s" % (
                result.track_b.ran, result.track_b.ok, result.track_b.dry_run,
                result.track_b.joints_path, result.track_b.error,
            ))
        print("log:", path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
