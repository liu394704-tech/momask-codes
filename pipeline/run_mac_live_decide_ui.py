#!/usr/bin/env python3
"""Live Mac window: vision + mic -> cloud ASR/Decide -> MoMask action_prompt.

Shows a camera window with real-time face emotion, and on demand (or auto)
records mic, transcribes, calls cloud LLM, and overlays the resulting
action_prompt for MoMask.

Controls:
  SPACE  record mic -> ASR -> cloud Decide
  D      Decide with current vision only (no mic)
  A      toggle auto round every --auto-sec seconds
  C      recalibrate face baseline
  S      save last decision JSON under pipeline_runs/
  ESC/Q  quit

Example:
  export OPENAI_API_KEY='...'
  export OPENAI_BASE_URL='https://www.dmxapi.cn/v1'
  export OPENAI_MODEL='gpt-4o-mini'
  export WHISPER_MODEL='gpt-4o-transcribe'

  源码/.venv-emotion-arm/bin/python -m pipeline.run_mac_live_decide_ui --mic-sec 3.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FUNCTIONS = ROOT / "源码" / "TonyPi" / "Functions"
if str(FUNCTIONS) not in sys.path:
    sys.path.insert(0, str(FUNCTIONS))

from .audio_mac import record_mic_wav, whisper_transcribe  # noqa: E402
from .decide_cloud import cloud_decide, mock_decide  # noqa: E402
from .schemas import Perception  # noqa: E402

WINDOW = "MoMask Live Decide (vision + mic + LLM)"
POINT_COLORS = [(0, 0, 255), (0, 128, 255), (0, 255, 255), (255, 0, 255)]


@dataclass
class UiState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    # live vision
    face_found: bool = False
    calibrating: bool = True
    calib_pct: float = 0.0
    emotion: str = "-"
    conf: float = 0.0
    intensity: str = "mild"
    scores: Dict[str, float] = field(default_factory=dict)
    fps: float = 0.0
    # pipeline
    busy: bool = False
    phase: str = "idle"  # idle|recording|asr|decide|done|error
    status: str = "press SPACE: mic+Decide | D: vision-only Decide"
    transcript: str = ""
    action_prompt: str = ""
    action_group: List[str] = field(default_factory=list)
    dec_emotion: str = ""
    dec_intent: str = ""
    dec_conf: float = 0.0
    fallback: bool = False
    reason: str = ""
    t_record: float = 0.0
    t_asr: float = 0.0
    t_decide: float = 0.0
    error: str = ""
    auto: bool = False
    round_id: int = 0
    last_saved: str = ""
    last_pack: Optional[Dict[str, Any]] = None


def parse_args():
    p = argparse.ArgumentParser(description="Live vision+mic+LLM decide window")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--mic-sec", type=float, default=3.5)
    p.add_argument("--auto-sec", type=float, default=12.0,
                   help="interval when auto mode (A) is on")
    p.add_argument("--calibration-frames", type=int, default=20)
    p.add_argument("--vote-window", type=int, default=15,
                   help="frames for majority emotion used in Decide")
    p.add_argument("--no-mirror", dest="mirror", action="store_false")
    p.add_argument("--mock-decide", action="store_true")
    p.add_argument("--whisper-model",
                   default=os.environ.get("WHISPER_MODEL", "gpt-4o-transcribe"))
    p.add_argument("--out-root", default=str(ROOT / "pipeline_runs"))
    p.set_defaults(mirror=True)
    return p.parse_args()


def _open_camera(index: int):
    import cv2

    backends = []
    if hasattr(cv2, "CAP_AVFOUNDATION"):
        backends.append(cv2.CAP_AVFOUNDATION)
    backends.append(cv2.CAP_ANY)
    order = [index] + [i for i in range(4) if i != index]
    for idx in order:
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            ok, frame = cap.read()
            if ok and frame is not None:
                print("[live] camera index=%d" % idx)
                return cap
            cap.release()
    return None


def _draw_text(cv2, frame, text, origin, color=(230, 230, 230), scale=0.48, thick=1):
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                cv2.LINE_AA)


def _wrap(text: str, width: int = 54) -> List[str]:
    text = (text or "").replace("\n", " ").strip()
    if not text:
        return ["(empty)"]
    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        if len(trial) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines[:6]


def _majority(votes: Deque[Tuple[str, float]]) -> Tuple[str, float, str]:
    if not votes:
        return "neutral", 0.0, "mild"
    labels = [e for e, _ in votes]
    best = max(set(labels), key=labels.count)
    confs = [c for e, c in votes if e == best]
    conf = float(sum(confs) / max(len(confs), 1))
    intensity = "strong" if conf >= 0.55 else "mild"
    return best, conf, intensity


def _panel(cv2, frame, state: UiState, result) -> None:
    h, w = frame.shape[:2]
    with state.lock:
        emo = state.emotion
        conf = state.conf
        scores = dict(state.scores)
        phase = state.phase
        status = state.status
        transcript = state.transcript
        prompt = state.action_prompt
        group = list(state.action_group)
        dec_e = state.dec_emotion
        dec_i = state.dec_intent
        dec_c = state.dec_conf
        fallback = state.fallback
        t_rec = state.t_record
        t_asr = state.t_asr
        t_dec = state.t_decide
        err = state.error
        auto = state.auto
        fps = state.fps
        busy = state.busy
        calibrating = state.calibrating
        calib_pct = state.calib_pct

    if result is not None and getattr(result, "found", False):
        if result.bbox is not None:
            x, y, bw, bh = result.bbox
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 235, 0), 2)
        if result.points5 is not None:
            for i, (px, py) in enumerate(result.points5):
                cv2.circle(frame, (int(px), int(py)), 3,
                           POINT_COLORS[i % len(POINT_COLORS)], -1)

    pw = min(520, w)
    ph = min(h, 420)
    shaded = frame[:ph, :pw].copy()
    frame[:ph, :pw] = cv2.addWeighted(shaded, 0.28, np.zeros_like(shaded), 0.72, 0)

    y = 22
    emo_color = (0, 235, 0) if emo not in ("-", "neutral") else (200, 200, 200)
    _draw_text(cv2, frame, "VISION: %s  conf=%.2f" % (str(emo).upper(), conf),
               (10, y), emo_color, 0.62, 2)
    y += 24
    if scores:
        for lab in sorted(scores, key=lambda k: -scores[k])[:4]:
            color = (0, 235, 0) if lab == emo else (160, 160, 160)
            _draw_text(cv2, frame, "%-10s %.2f" % (lab, scores[lab]), (10, y), color, 0.42)
            cv2.rectangle(frame, (130, y - 9), (130 + int(90 * scores[lab]), y + 1), color, -1)
            y += 16
    else:
        _draw_text(cv2, frame, "waiting for calibrated face...", (10, y), (160, 160, 160), 0.42)
        y += 18

    phase_color = {
        "idle": (200, 200, 200),
        "recording": (0, 165, 255),
        "asr": (0, 215, 255),
        "decide": (255, 180, 0),
        "done": (0, 235, 0),
        "error": (0, 80, 255),
    }.get(phase, (200, 200, 200))
    y += 6
    _draw_text(cv2, frame, "PHASE: %s%s" % (phase.upper(), " [BUSY]" if busy else ""),
               (10, y), phase_color, 0.55, 2)
    y += 20
    _draw_text(cv2, frame, status[:70], (10, y), (210, 210, 210), 0.42)
    y += 20

    _draw_text(cv2, frame, "ASR transcript:", (10, y), (180, 220, 255), 0.45)
    y += 16
    for line in _wrap(transcript or "(none yet)", 48):
        _draw_text(cv2, frame, line, (10, y), (230, 230, 230), 0.42)
        y += 15

    y += 4
    _draw_text(cv2, frame, "LLM Decide -> MoMask prompt:", (10, y), (180, 255, 180), 0.45)
    y += 16
    for line in _wrap(prompt or "(press SPACE or D)", 48):
        _draw_text(cv2, frame, line, (10, y), (0, 255, 180), 0.43)
        y += 15

    y += 4
    group_s = ",".join(group) if group else "-"
    _draw_text(
        cv2, frame,
        "dec: emo=%s intent=%s conf=%.2f fb=%s  A=%s" % (
            dec_e or "-", dec_i or "-", dec_c, fallback, group_s),
        (10, y), (200, 200, 120), 0.40,
    )
    y += 16
    _draw_text(
        cv2, frame,
        "timing  rec=%.1fs asr=%.1fs decide=%.1fs   auto=%s  fps=%.1f" % (
            t_rec, t_asr, t_dec, "ON" if auto else "off", fps),
        (10, y), (0, 215, 255), 0.40,
    )
    y += 16
    if err:
        _draw_text(cv2, frame, ("ERR: " + err)[:72], (10, y), (0, 80, 255), 0.40)
        y += 16
    _draw_text(cv2, frame, "SPACE mic+Decide | D vision Decide | A auto | C calib | S save | ESC quit",
               (10, y), (170, 170, 170), 0.38)

    if calibrating:
        cv2.rectangle(frame, (0, h - 36), (w, h), (0, 0, 0), -1)
        _draw_text(cv2, frame,
                   "CALIBRATING %d%% — hold a relaxed face" % int(calib_pct * 100),
                   (10, h - 12), (0, 215, 255), 0.55, 2)


def _run_round(
    state: UiState,
    args,
    with_mic: bool,
    session_prefix: str,
    out_dir: Path,
) -> None:
    with state.lock:
        if state.busy:
            return
        state.busy = True
        state.round_id += 1
        rid = state.round_id
        emo = state.emotion if state.emotion not in ("-", "") else "neutral"
        conf = float(state.conf)
        intensity = state.intensity
        face = state.face_found and not state.calibrating
        state.phase = "recording" if with_mic else "decide"
        state.status = "recording mic..." if with_mic else "calling cloud Decide..."
        state.error = ""
        if with_mic:
            state.transcript = ""
        state.t_record = state.t_asr = state.t_decide = 0.0

    session_id = "%s_%d_%s" % (session_prefix, rid, uuid.uuid4().hex[:6])
    transcript = ""
    t_rec = t_asr = 0.0

    try:
        if with_mic:
            wav = out_dir / ("live_mic_%s.wav" % session_id)
            t0 = time.time()
            ok, err = record_mic_wav(str(wav), args.mic_sec)
            t_rec = time.time() - t0
            with state.lock:
                state.t_record = t_rec
                if not ok:
                    state.phase = "error"
                    state.error = "mic: %s" % err
                    state.status = "mic failed"
                    return
                state.phase = "asr"
                state.status = "cloud ASR (%s)..." % args.whisper_model

            t1 = time.time()
            transcript, asr_err = whisper_transcribe(str(wav), model=args.whisper_model)
            t_asr = time.time() - t1
            with state.lock:
                state.t_asr = t_asr
                state.transcript = transcript or ""
                if asr_err:
                    state.error = "asr: %s" % asr_err[:180]
                    # still continue to Decide with vision
                state.phase = "decide"
                state.status = "cloud LLM Decide..."

        perception = Perception(
            session_id=session_id,
            ts_ms=int(time.time() * 1000),
            vision_emotion=emo,
            vision_conf=conf,
            vision_intensity=intensity,
            face_found=face,
            transcript=transcript,
            ok=True,
        )

        with state.lock:
            state.phase = "decide"
            state.status = "cloud LLM Decide..."

        t2 = time.time()
        if args.mock_decide or not os.environ.get("OPENAI_API_KEY", "").strip():
            decision = mock_decide(perception)
        else:
            decision = cloud_decide(perception)
        t_dec = time.time() - t2

        with state.lock:
            state.t_decide = t_dec
            state.dec_emotion = decision.emotion
            state.dec_intent = decision.intent
            state.dec_conf = float(decision.confidence)
            state.action_prompt = decision.action_prompt or ""
            state.action_group = list(decision.action_group or [])
            state.fallback = bool(decision.fallback)
            state.reason = decision.reason or ""
            if decision.error:
                state.error = (state.error + " | " if state.error else "") + decision.error[:160]
                state.phase = "error"
                state.status = "Decide error"
            else:
                state.phase = "done"
                state.status = "prompt ready for MoMask (S to save)"
            state.last_pack = {
                "perception": perception.to_dict(),
                "decision": decision.to_dict(),
                "timing": {"record_s": t_rec, "asr_s": t_asr, "decide_s": t_dec},
            }
    except Exception as exc:  # noqa: BLE001
        with state.lock:
            state.phase = "error"
            state.error = str(exc)[:220]
            state.status = "round failed"
    finally:
        with state.lock:
            state.busy = False


def main():
    import cv2
    import FaceExpression as FX  # type: ignore

    args = parse_args()
    out_root = Path(args.out_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / ("live_decide_%s" % stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("OPENAI_API_KEY", "").strip() and not args.mock_decide:
        print("[live] WARNING: no OPENAI_API_KEY — will use mock_decide")
    print("[live] out_dir=%s" % out_dir)
    print("[live] model=%s whisper=%s" % (
        os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), args.whisper_model))

    analyzer = FX.ExpressionAnalyzer(calibration_frames=args.calibration_frames)
    cap = _open_camera(args.camera_index)
    if cap is None:
        print("[live] camera unavailable — check macOS Camera permission")
        analyzer.close()
        return 1

    state = UiState()
    votes: Deque[Tuple[str, float]] = deque(maxlen=max(5, args.vote_window))
    session_prefix = "live"
    last_auto = 0.0
    fps_t0 = time.time()
    fps_n = 0

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1100, 720)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            if args.mirror:
                frame = cv2.flip(frame, 1)

            result = analyzer.process(frame)
            if result.found and not result.calibrating and result.emotion:
                votes.append((result.emotion, float(result.emotion_score or 0.0)))
            maj_e, maj_c, maj_i = _majority(votes)

            with state.lock:
                state.face_found = bool(result.found)
                state.calibrating = bool(result.calibrating)
                state.calib_pct = float(result.calibration_progress or 0.0)
                if result.found and not result.calibrating:
                    state.emotion = maj_e
                    state.conf = maj_c
                    state.intensity = maj_i
                    if result.emotion_scores:
                        state.scores = dict(result.emotion_scores)
                elif result.calibrating:
                    state.emotion = "-"
                    state.conf = 0.0

            fps_n += 1
            if time.time() - fps_t0 >= 1.0:
                with state.lock:
                    state.fps = fps_n / (time.time() - fps_t0)
                fps_t0 = time.time()
                fps_n = 0

            # auto rounds
            with state.lock:
                auto_on = state.auto
                busy = state.busy
            if auto_on and not busy and (time.time() - last_auto) >= args.auto_sec:
                last_auto = time.time()
                threading.Thread(
                    target=_run_round,
                    args=(state, args, True, session_prefix, out_dir),
                    daemon=True,
                ).start()

            _panel(cv2, frame, state, result)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("c"), ord("C")):
                analyzer.recalibrate()
                votes.clear()
                with state.lock:
                    state.status = "recalibrating — hold relaxed face"
            if key in (ord("a"), ord("A")):
                with state.lock:
                    state.auto = not state.auto
                    state.status = "auto ON every %.1fs" % args.auto_sec if state.auto else "auto OFF"
                last_auto = time.time()
            if key == ord(" "):
                threading.Thread(
                    target=_run_round,
                    args=(state, args, True, session_prefix, out_dir),
                    daemon=True,
                ).start()
            if key in (ord("d"), ord("D")):
                threading.Thread(
                    target=_run_round,
                    args=(state, args, False, session_prefix, out_dir),
                    daemon=True,
                ).start()
            if key in (ord("s"), ord("S")):
                with state.lock:
                    pack = state.last_pack
                if not pack:
                    with state.lock:
                        state.status = "nothing to save yet"
                else:
                    path = out_dir / ("decision_%s.json" % datetime.now().strftime("%H%M%S"))
                    path.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
                    with state.lock:
                        state.last_saved = str(path)
                        state.status = "saved %s" % path.name
                    print("[live] saved", path)
    finally:
        cap.release()
        analyzer.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
