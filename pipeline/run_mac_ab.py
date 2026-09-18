#!/usr/bin/env python3
"""Mac A/B pipeline entry (Phase 1 default: A_parallel_B).

Baseline-safe modes:
  # 1) No camera / no API — full mock round (always safe)
  python -m pipeline.run_mac_ab --once --mock-perception --mock-decide --dry-run-b

  # 2) Local face geometry + mock decide (needs 源码 mediapipe env + camera)
  python -m pipeline.run_mac_ab --once --mock-decide --dry-run-b

  # 3) On-device Qwen Decide (Pi formal path; NO OpenAI)
  export DECIDE_BACKEND=edge_llm
  export EDGE_LLM_GGUF=$PWD/models/edge_llm/qwen2.5-1.5b-instruct-q4_k_m.gguf
  python -m pipeline.run_mac_ab --once --mock-perception --dry-run-b

  # 4) Mac-only cloud decide (联调 only)
  export DECIDE_BACKEND=cloud
  export OPENAI_API_KEY='sk-...'

  # 5) Real MoMask joints
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
from pipeline.decide import default_backend, run_decide  # noqa: E402
from pipeline.percept_mac import capture_perception_mac, mock_perception  # noqa: E402
from pipeline.track_a import run_track_a  # noqa: E402
from pipeline.track_b import run_track_b  # noqa: E402
from pipeline.schemas import Mode  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Mac/Pi A/B arbiter pipeline")
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
    p.add_argument("--mock-decide", action="store_true",
                   help="force edge rule decide (alias of --decide-backend mock)")
    p.add_argument(
        "--decide-backend",
        default=os.environ.get("DECIDE_BACKEND", "") or default_backend(),
        choices=["edge", "edge_llm", "edge_auto", "cloud", "mock"],
        help="edge_llm=Qwen on-device (Pi); edge=rules; cloud=OpenAI Mac-only",
    )
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
    backend = "mock" if args.mock_decide else args.decide_backend

    n = 1 if args.once else max(1, args.rounds)
    print("[ab] mode=%s decide=%s mock_perception=%s dry_run_b=%s" % (
        mode.value, backend, args.mock_perception, args.dry_run_b
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

        decision = run_decide(perception, backend=backend)

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
            print("track_b: ran=%s ok=%s dry_run=%s load=%.2fs gen=%.2fs joints=%s err=%s" % (
                result.track_b.ran, result.track_b.ok, result.track_b.dry_run,
                result.track_b.load_s, result.track_b.gen_s or result.track_b.elapsed_s,
                result.track_b.joints_path, result.track_b.error,
            ))
        print("log:", path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
