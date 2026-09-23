#!/usr/bin/env python3
"""Pi live loop: face + WonderEcho + local ASR -> edge LLM -> phrase ActionGroups.

MoMask is optional. Default OFF. Turn on with --momask or ENABLE_MOMASK=1.
When on, Track A still runs first (robot moves immediately); Track B writes
joints.npy in parallel and never drives servos.

Examples:
  python -m pipeline.run_pi_emotion_llm --once --mock-perception --simulate
  python -m pipeline.run_pi_emotion_llm --once --mock-perception --simulate \\
      --momask --momask-dry-run --transcript '你好呀'
  ENABLE_MOMASK=1 python -m pipeline.run_pi_emotion_llm --momask
"""
from __future__ import annotations

import argparse
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

from pipeline.actions import (  # noqa: E402
    is_stop_signal,
    merge_keyword_transcript,
)
from pipeline.audio_pi import (  # noqa: E402
    AudioEvent,
    WonderEchoListener,
    handle_echo_keyword,
)
from pipeline.decide import default_backend, run_decide  # noqa: E402
from pipeline.percept_mac import mock_perception  # noqa: E402
from pipeline.percept_pi import (  # noqa: E402
    capture_perception_pi,
    close_pi_vision_session,
    get_pi_vision_session,
)
from pipeline.arbiter import Arbiter, ArbiterConfig  # noqa: E402
from pipeline.motion_quality import quality_from_joints_path  # noqa: E402
from pipeline.schemas import Decision, Mode, Perception, TrackBResult  # noqa: E402
from pipeline.preset_select import PhraseSelector  # noqa: E402
from pipeline.track_a import run_track_a  # noqa: E402
from pipeline.track_b import run_track_b  # noqa: E402


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def parse_args():
    p = argparse.ArgumentParser(
        description="Pi emotion + keyword + ASR -> edge LLM -> ActionGroup; MoMask optional"
    )
    p.add_argument("--once", action="store_true")
    p.add_argument("--rounds", type=int, default=0, help="0 = run until Ctrl-C")
    p.add_argument("--mock-perception", action="store_true")
    p.add_argument("--mock-emotion", default="happy")
    p.add_argument("--mock-conf", type=float, default=0.72)
    p.add_argument("--transcript", default="")
    p.add_argument("--keyword", default="")
    p.add_argument(
        "--decide-backend",
        default=os.environ.get("DECIDE_BACKEND", "") or default_backend(),
        choices=["edge", "edge_llm", "edge_auto", "cloud", "mock"],
    )
    p.add_argument(
        "--simulate",
        action="store_true",
        help="do not call hiwonder; still print / log the action",
    )
    p.add_argument("--camera-index", type=int, default=-1)
    p.add_argument("--calibrate-frames", type=int, default=25)
    p.add_argument("--sample-frames", type=int, default=8)
    p.add_argument("--audio-sec", type=float, default=3.0)
    p.add_argument("--whisper-size", default=os.environ.get("WHISPER_SIZE", "tiny"))
    p.add_argument("--echo-port", default=os.environ.get("WONDERECHO_PORT", "/dev/ttyUSB0"))
    p.add_argument("--vision-interval", type=float, default=2.5)
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--no-echo", action="store_true", help="skip WonderEcho (vision / --transcript only)")
    p.add_argument("--out-dir", default=str(ROOT / "pipeline_runs"))
    p.add_argument(
        "--momask",
        dest="momask",
        action="store_true",
        default=env_flag("ENABLE_MOMASK", False),
        help="enable Track B MoMask joints (or set ENABLE_MOMASK=1)",
    )
    p.add_argument(
        "--no-momask",
        dest="momask",
        action="store_false",
        help="disable MoMask even if ENABLE_MOMASK=1",
    )
    p.add_argument(
        "--momask-dry-run",
        action="store_true",
        default=env_flag("MOMASK_DRY_RUN", False),
        help="MoMask on, but only write a prompt marker (no weights)",
    )
    p.add_argument("--momask-timeout", type=float, default=float(os.environ.get("MOMASK_TIMEOUT_S", "30") or 30))
    p.add_argument("--gpu-id", type=int, default=int(os.environ.get("MOMASK_GPU_ID", "-1") or -1))
    p.add_argument("--momask-python", default=os.environ.get("MOMASK_PYTHON") or "")
    p.add_argument("--confidence-min", type=float, default=0.45)
    return p.parse_args()


def _ensure_functions_path():
    from pipeline.track_a import _FUNCTIONS

    if _FUNCTIONS not in sys.path:
        sys.path.insert(0, _FUNCTIONS)


def _import_scheduler():
    _ensure_functions_path()
    try:
        from EmotionActionScheduler import EmotionActionScheduler  # type: ignore
    except Exception:
        from pipeline.emotion_scheduler import EmotionActionScheduler  # type: ignore
    return EmotionActionScheduler()


def _apply_stop(decision: Decision, perception: Perception) -> Decision:
    keyword = (perception.extras or {}).get("keyword")
    if is_stop_signal(keyword=keyword, transcript=perception.transcript, intent=decision.intent):
        decision.intent = "stop"
        decision.action_group = ["stand"]
        decision.action_prompt = (
            decision.action_prompt
            or "a person stands still with both arms relaxed at the sides"
        )
        decision.reason = (decision.reason or "") + "|stop_override"
        decision.fallback = False
    return decision


def _build_perception(
    args,
    session: str,
    transcript: str,
    keyword: Optional[str],
    echo_error: Optional[str],
    audio_emotion: Optional[str] = None,
    audio_conf: float = 0.0,
    ser_extra: Optional[Dict[str, Any]] = None,
) -> Perception:
    if args.mock_perception:
        perc = mock_perception(session, emotion=args.mock_emotion, conf=args.mock_conf)
        perc.transcript = merge_keyword_transcript(keyword, transcript or args.transcript)
        if keyword:
            perc.extras["keyword"] = keyword
        if echo_error:
            perc.extras["keyword_unavailable"] = echo_error
        if audio_emotion:
            perc.audio_emotion = audio_emotion
            perc.audio_conf = float(audio_conf or 0.0)
        if ser_extra:
            perc.extras.update(ser_extra)
        return perc

    perc = capture_perception_pi(
        session_id=session,
        camera_index=args.camera_index,
        calibrate_frames=args.calibrate_frames,
        sample_frames=args.sample_frames,
        mirror=args.mirror,
        transcript=transcript or args.transcript,
        keyword=keyword,
        keep_open=True,
    )
    if echo_error:
        perc.extras["keyword_unavailable"] = echo_error
    if audio_emotion:
        perc.audio_emotion = audio_emotion
        perc.audio_conf = float(audio_conf or 0.0)
    if ser_extra:
        perc.extras.update(ser_extra)
    return perc


def _log_round(
    out_dir: Path,
    session: str,
    perception: Perception,
    decision: Decision,
    track_a,
    extra: dict,
    track_b: Optional[TrackBResult] = None,
    momask_on: bool = False,
    quality: Optional[Dict[str, Any]] = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": session,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "perception": perception.to_dict(),
        "decision": decision.to_dict(),
        "track_a": {
            "executed": track_a.executed,
            "action": track_a.action,
            "intensity": track_a.intensity,
            "simulated": track_a.simulated,
            "detail": track_a.detail,
            "phrase_id": getattr(track_a, "phrase_id", None),
            "clips": list(getattr(track_a, "clips", None) or []),
            "recovery": getattr(track_a, "recovery", None),
            "bans": list(getattr(track_a, "bans", None) or []),
            "pulses": list(getattr(track_a, "pulses", None) or []),
            "coords": getattr(track_a, "coords", None) or {},
            "coord_summary": getattr(track_a, "coord_summary", "") or "",
        },
        "momask": bool(momask_on),
        "track_b": None,
        "quality": quality,
        "extra": extra,
    }
    if track_b is not None:
        payload["track_b"] = {
            "ran": track_b.ran,
            "ok": track_b.ok,
            "joints_path": track_b.joints_path,
            "prompt": track_b.prompt,
            "elapsed_s": track_b.elapsed_s,
            "load_s": track_b.load_s,
            "gen_s": track_b.gen_s,
            "error": track_b.error,
            "dry_run": track_b.dry_run,
        }
    path = out_dir / ("%s_emotion_llm.json" % session)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _print_round(perception: Perception, decision: Decision, track_a, path: Path, track_b=None, momask_on=False):
    print("perception: face=%s emo=%s conf=%.2f audio_emo=%s audio_conf=%.2f keyword=%r transcript=%r" % (
        perception.face_found,
        perception.vision_emotion,
        perception.vision_conf,
        perception.audio_emotion,
        perception.audio_conf,
        (perception.extras or {}).get("keyword"),
        perception.transcript,
    ))
    print("decision: emo=%s intent=%s conf=%.2f fallback=%s group=%s" % (
        decision.emotion, decision.intent, decision.confidence,
        decision.fallback, decision.action_group,
    ))
    print("prompt: %s" % (decision.action_prompt[:140]))
    print("track_a: action=%s executed=%s simulated=%s (%s)" % (
        track_a.action, track_a.executed, track_a.simulated, track_a.detail,
    ))
    print("phrase: id=%s clips=%s recovery=%s bans=%s" % (
        getattr(track_a, "phrase_id", None),
        list(getattr(track_a, "clips", None) or []),
        getattr(track_a, "recovery", None),
        list(getattr(track_a, "bans", None) or []),
    ))
    summary = getattr(track_a, "coord_summary", "") or ""
    if summary:
        print(summary)
        peak = ((getattr(track_a, "coords", None) or {}).get("peak_xyz_mm") or {})
        if peak:
            print("coords_peak: %s" % peak)
    if momask_on and track_b is not None:
        print("track_b: ran=%s ok=%s dry_run=%s gen=%.2fs joints=%s err=%s" % (
            track_b.ran, track_b.ok, track_b.dry_run,
            track_b.gen_s or track_b.elapsed_s,
            track_b.joints_path, track_b.error,
        ))
    elif not momask_on:
        print("track_b: skipped (momask=OFF)")
    print("log:", path)


def run_one_round(
    args, session: str, transcript: str, keyword: Optional[str], echo_error: Optional[str],
    scheduler, selector=None, audio_emotion: Optional[str] = None, audio_conf: float = 0.0,
    ser_extra: Optional[Dict[str, Any]] = None,
):
    t0 = time.perf_counter()
    perception = _build_perception(
        args, session, transcript, keyword, echo_error,
        audio_emotion=audio_emotion, audio_conf=audio_conf, ser_extra=ser_extra,
    )
    t_perc = time.perf_counter() - t0
    t1 = time.perf_counter()
    decision = run_decide(perception, backend=args.decide_backend)
    decision = _apply_stop(decision, perception)
    t_dec = time.perf_counter() - t1
    no_audio = not keyword and not (perception.transcript or "").strip()
    do_robot = (not args.simulate) and not (decision.fallback and no_audio)

    mode = Mode.A_PARALLEL_B if args.momask else Mode.A_ONLY
    arbiter = Arbiter(ArbiterConfig(mode=mode, confidence_min=float(args.confidence_min)))
    effective, degraded, skip_reason = arbiter.resolve_effective_mode(decision)

    track_b_box: Dict[str, Any] = {}

    def _run_b():
        track_b_box["r"] = run_track_b(
            decision,
            out_dir=Path(args.out_dir) / session,
            dry_run=bool(args.momask_dry_run),
            gpu_id=int(args.gpu_id),
            timeout_s=float(args.momask_timeout),
            python_executable=(args.momask_python or None),
        )

    worker = None
    if effective in (Mode.B_ONLY, Mode.A_PARALLEL_B):
        worker = threading.Thread(target=_run_b, name="momask-b", daemon=True)
        worker.start()

    track_a = run_track_a(
        decision,
        simulate=not do_robot,
        execute_robot=do_robot,
        keyword=keyword,
        scheduler=scheduler,
        perception=perception,
        selector=selector,
    )

    track_b = None
    if worker is not None:
        worker.join(timeout=max(1.0, float(args.momask_timeout)))
        if worker.is_alive():
            track_b = TrackBResult(
                ran=True, ok=False, prompt=decision.action_prompt,
                error="momask_timeout", elapsed_s=float(args.momask_timeout),
            )
            degraded = True
            skip_reason = skip_reason or "momask_timeout"
        else:
            track_b = track_b_box.get("r")
            if track_b is not None and track_b.ran and not track_b.ok:
                degraded = True
                skip_reason = skip_reason or (track_b.error or "momask_failed")

    quality = None
    if track_b is not None and track_b.ok and track_b.joints_path:
        quality = quality_from_joints_path(track_b.joints_path)

    extra = {
        "t_perception_s": round(t_perc, 3),
        "t_decide_s": round(t_dec, 3),
        "t_total_s": round(time.perf_counter() - t0, 3),
        "decide_backend": args.decide_backend,
        "momask": bool(args.momask),
        "effective_mode": effective.value,
        "degraded": bool(degraded),
        "degrade_reason": skip_reason,
    }
    path = _log_round(
        Path(args.out_dir), session, perception, decision, track_a, extra,
        track_b=track_b, momask_on=bool(args.momask), quality=quality,
    )
    _print_round(perception, decision, track_a, path, track_b=track_b, momask_on=bool(args.momask))
    return perception, decision, track_a


def _poll_audio(args, echo: Optional[WonderEchoListener], round_dir: Path) -> Optional[AudioEvent]:
    if echo is None:
        return None
    keyword = echo.poll()
    if not keyword:
        return None
    if keyword == "wakeup":
        wav = round_dir / ("wakeup_%d.wav" % int(time.time()))
        print("[pi] wakeup — recording %.1fs for ASR" % args.audio_sec, flush=True)
        return handle_echo_keyword(
            "wakeup",
            str(wav),
            duration_sec=args.audio_sec,
            whisper_size=args.whisper_size,
        )
    print("[pi] keyword=%s" % keyword, flush=True)
    return AudioEvent(kind="keyword", keyword=keyword, transcript="")


def main() -> int:
    args = parse_args()
    os.environ.setdefault("DECIDE_BACKEND", args.decide_backend)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scheduler = _import_scheduler()
    selector = PhraseSelector()

    echo = None
    echo_error = None
    if not args.no_echo and not args.mock_perception:
        echo = WonderEchoListener(port=args.echo_port)
        if not echo.open():
            echo_error = echo.error or "keyword_unavailable"
            print("[pi] WonderEcho unavailable: %s (vision-only)" % echo_error, flush=True)
        else:
            print("[pi] WonderEcho on %s" % args.echo_port, flush=True)

    print("[pi] decide=%s simulate=%s momask=%s dry_run=%s" % (
        args.decide_backend,
        bool(args.simulate),
        "ON" if args.momask else "OFF",
        bool(args.momask_dry_run),
    ), flush=True)

    if args.once or args.mock_perception:
        session = "pi_%s" % uuid.uuid4().hex[:8]
        run_one_round(
            args, session,
            transcript=args.transcript,
            keyword=args.keyword or None,
            echo_error=echo_error,
            scheduler=scheduler,
            selector=selector,
        )
        close_pi_vision_session()
        if echo:
            echo.close()
        return 0

    # Live: keep camera open, trigger on keyword/ASR or stable non-neutral face.
    n_done = 0
    last_vision = 0.0
    try:
        if not args.mock_perception:
            print("[pi] opening camera / calibrating neutral face…", flush=True)
            get_pi_vision_session(args.camera_index, args.calibrate_frames, args.mirror)
        while True:
            if args.rounds and n_done >= args.rounds:
                break
            session = "pi_%s" % uuid.uuid4().hex[:8]
            event = _poll_audio(args, echo, out_dir)
            keyword = None
            transcript = args.transcript
            force = False
            stop_now = False

            if not args.mock_perception:
                try:
                    sess = get_pi_vision_session(
                        args.camera_index, args.calibrate_frames, args.mirror
                    )
                    _frame, result = sess.read_processed()
                    if (
                        result is not None
                        and result.found
                        and not result.calibrating
                        and result.emotion
                    ):
                        scheduler.observe(result.emotion, float(result.emotion_score or 0.0))
                except Exception:
                    pass

            if event is not None:
                keyword = event.keyword
                stop_now = is_stop_signal(keyword=keyword, transcript=event.transcript or "")
                if event.kind == "asr":
                    transcript = event.transcript or ""
                    force = True
                elif event.keyword == "wakeup" and not event.transcript:
                    print("[pi] wakeup with empty ASR: %s" % (event.error or ""), flush=True)
                    time.sleep(0.05)
                    continue
                else:
                    force = True
            elif scheduler.can_schedule():
                now = time.time()
                if now - last_vision >= args.vision_interval:
                    stable, _conf = scheduler.voter.vote(now)
                    if stable and stable != "neutral":
                        force = True
                        last_vision = now

            if force and not stop_now and not scheduler.can_schedule():
                time.sleep(0.05)
                continue

            if not force:
                time.sleep(0.03)
                continue

            print("\n===== round %d %s =====" % (n_done + 1, session), flush=True)
            ser_extra = None
            audio_emotion = None
            audio_conf = 0.0
            if event is not None:
                audio_emotion = event.audio_emotion
                audio_conf = float(event.audio_conf or 0.0)
                if event.ser_error or event.audio_emotion:
                    ser_extra = {
                        "ser_error": event.ser_error,
                        "ser_s": event.ser_s,
                    }
            _perc, _dec, track_a = run_one_round(
                args, session,
                transcript=transcript,
                keyword=keyword,
                echo_error=echo_error,
                scheduler=scheduler,
                selector=selector,
                audio_emotion=audio_emotion,
                audio_conf=audio_conf,
                ser_extra=ser_extra,
            )
            n_done += 1
            if args.simulate and track_a.executed:
                # Simulated cooldown so the next vision round is not instant.
                time.sleep(0.2)
            if args.once:
                break
    except KeyboardInterrupt:
        print("\n[pi] stopped", flush=True)
    finally:
        close_pi_vision_session()
        if echo:
            echo.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
