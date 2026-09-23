#!/usr/bin/env python3
"""Time the on-robot path: wakeup -> record -> local ASR -> face -> edge decide.

Writes one CSV row per round under pipeline_runs/latency/. Does not call a
cloud API. Action playback is off unless --move is passed.

  python3 -m pipeline.bench_pi_latency --probe
  python3 -m pipeline.bench_pi_latency --rounds 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.audio_pi import WonderEchoListener, handle_echo_keyword  # noqa: E402
from pipeline.decide import run_decide  # noqa: E402
from pipeline.decide_edge import resolve_gguf_path  # noqa: E402
from pipeline.percept_pi import (  # noqa: E402
    capture_perception_pi,
    close_pi_vision_session,
)
from pipeline.preset_select import PhraseSelector  # noqa: E402
from pipeline.schemas import Perception  # noqa: E402
from pipeline.track_a import run_track_a  # noqa: E402


COLUMNS = [
    "轮次",
    "触发",
    "唤醒等待_s",
    "录音_s",
    "语音识别_s",
    "识别文本",
    "识别错误",
    "语音情绪_s",
    "语音情绪错误",
    "人脸_s",
    "校准_s",
    "人脸情绪",
    "决策_s",
    "请求的决策",
    "实际决策",
    "决策错误",
    "选动作_s",
    "短语",
    "动作播放_s",
    "识别到决策_s",
    "总_s",
]


def _import_ok(module: str) -> str:
    try:
        __import__(module)
    except Exception as exc:  # noqa: BLE001
        return "missing:%s" % str(exc).split("\n", 1)[0][:160]
    return "ok"


def probe_stack() -> Dict[str, Any]:
    """What is actually installed. A missing runtime is not a successful deploy."""
    gguf = resolve_gguf_path()
    echo_port = os.environ.get("WONDERECHO_PORT", "/dev/ttyUSB0")
    return {
        "whisper": _import_ok("whisper"),
        "llama_cpp": _import_ok("llama_cpp"),
        "funasr": _import_ok("funasr"),
        "serial": _import_ok("serial"),
        "cv2": _import_ok("cv2"),
        "gguf_path": gguf,
        "gguf_present": os.path.isfile(gguf),
        "echo_port": echo_port,
        "echo_port_present": os.path.exists(echo_port),
        "asr_deployed": _import_ok("whisper") == "ok",
        "llm_runtime_deployed": _import_ok("llama_cpp") == "ok" and os.path.isfile(gguf),
    }


def decide_impl(decision, requested: str) -> str:
    reason = (decision.reason or "")
    err = decision.error or ""
    if err or reason.startswith("edge_llm_fallback") or reason.startswith("edge_llm_empty"):
        return "edge_rule_fallback"
    if reason.startswith("edge_llm") or requested in ("edge_llm", "llm", "gguf"):
        return "edge_llm"
    return "edge_rule"


def finish_round(
    perception: Perception,
    *,
    trial: int,
    trigger: str,
    requested_backend: str,
    wake_wait_s: float = 0.0,
    record_s: float = 0.0,
    asr_s: float = 0.0,
    asr_error: Optional[str] = None,
    ser_s: float = 0.0,
    ser_error: Optional[str] = None,
    vision_s: float = 0.0,
    calib_s: float = 0.0,
    move: bool = False,
    selector: Optional[PhraseSelector] = None,
    scheduler=None,
) -> Dict[str, Any]:
    """Time decide + phrase selection for an already built perception."""
    t0 = time.perf_counter()
    decision = run_decide(perception, backend=requested_backend)
    t_decide = time.perf_counter() - t0
    picker = selector or PhraseSelector()
    t1 = time.perf_counter()
    track = run_track_a(
        decision,
        simulate=not move,
        execute_robot=bool(move),
        keyword=(perception.extras or {}).get("keyword"),
        scheduler=scheduler,
        perception=perception,
        selector=picker,
    )
    t_phrase = time.perf_counter() - t1
    t_action = t_phrase if move else 0.0
    t_select = 0.0 if move else t_phrase
    to_decision = record_s + asr_s + ser_s + vision_s + t_decide + t_select
    total = wake_wait_s + to_decision + t_action
    return {
        "轮次": trial,
        "触发": trigger,
        "唤醒等待_s": round(wake_wait_s, 3),
        "录音_s": round(record_s, 3),
        "语音识别_s": round(asr_s, 3),
        "识别文本": perception.transcript or "",
        "识别错误": asr_error or "",
        "语音情绪_s": round(ser_s, 3),
        "语音情绪错误": ser_error or "",
        "人脸_s": round(vision_s, 3),
        "校准_s": round(calib_s, 3),
        "人脸情绪": perception.vision_emotion or "",
        "决策_s": round(t_decide, 3),
        "请求的决策": requested_backend,
        "实际决策": decide_impl(decision, requested_backend),
        "决策错误": (decision.error or "")[:180],
        "选动作_s": round(t_select, 3),
        "短语": track.phrase_id or track.action or "",
        "动作播放_s": round(t_action, 3),
        "识别到决策_s": round(to_decision, 3),
        "总_s": round(total, 3),
    }


def append_latency_row(path: Path, row: Dict[str, Any]) -> None:
    """Append one round to a CSV, creating the header on the first write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def _median(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
    if not vals:
        return None
    return round(statistics.median(vals), 3)


def write_report(out_dir: Path, rows: List[Dict[str, Any]], probe: Dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / ("latency_%s.csv" % stamp)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "probe": probe,
        "rounds": len(rows),
        "csv": str(csv_path),
        "median_识别到决策_s": _median(rows, "识别到决策_s"),
        "median_录音_s": _median(rows, "录音_s"),
        "median_语音识别_s": _median(rows, "语音识别_s"),
        "median_人脸_s": _median(rows, "人脸_s"),
        "median_决策_s": _median(rows, "决策_s"),
        "median_选动作_s": _median(rows, "选动作_s"),
        "median_动作播放_s": _median(rows, "动作播放_s"),
        "实际决策": sorted({row["实际决策"] for row in rows}),
    }
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path


def _print_probe(probe: Dict[str, Any]) -> None:
    print("ASR whisper: %s" % probe["whisper"], flush=True)
    print("LLM llama_cpp: %s  gguf: %s" % (
        probe["llama_cpp"], "yes" if probe["gguf_present"] else "no",
    ), flush=True)
    print("SER funasr: %s" % probe["funasr"], flush=True)
    print("WonderEcho %s exists=%s serial=%s" % (
        probe["echo_port"], probe["echo_port_present"], probe["serial"],
    ), flush=True)
    if probe["asr_deployed"] and probe["llm_runtime_deployed"]:
        print("status: ASR and edge LLM runtimes import on this machine", flush=True)
    else:
        print("status: not fully deployed — missing pieces stay blank in the CSV", flush=True)


def _wait_keyword(echo: WonderEchoListener, timeout_s: float):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        word = echo.poll()
        if word:
            return word, time.perf_counter() - t0
        time.sleep(0.02)
    return None, time.perf_counter() - t0


def _capture_vision(args, trial: int, transcript: str, keyword: Optional[str]):
    t0 = time.perf_counter()
    perception = capture_perception_pi(
        "lat_%d_%s" % (trial, uuid.uuid4().hex[:6]),
        camera_index=args.camera_index,
        calibrate_frames=args.calibrate_frames,
        sample_frames=args.sample_frames,
        mirror=args.mirror,
        transcript=transcript,
        keyword=keyword,
        keep_open=True,
    )
    extras = perception.extras or {}
    return perception, time.perf_counter() - t0, float(extras.get("t_calib_s") or 0.0)


def _perception_without_camera(transcript: str, keyword: Optional[str]) -> Perception:
    return Perception(
        session_id="lat_%s" % uuid.uuid4().hex[:6],
        ts_ms=int(time.time() * 1000),
        transcript=transcript,
        face_found=False,
        ok=True,
        extras={"keyword": keyword} if keyword else {},
    )


def run_live(args) -> int:
    probe = probe_stack()
    _print_probe(probe)
    if args.probe:
        return 0

    out_dir = Path(args.out_dir)
    echo = None
    use_echo = not args.mic_only and not args.wav
    if use_echo:
        echo = WonderEchoListener(port=args.echo_port)
        if not echo.open():
            print("WonderEcho unavailable: %s" % (echo.error or "unavailable"), flush=True)
            print("switch to microphone-only rounds. Speak during the recording window.", flush=True)
            use_echo = False
            echo = None

    selector = PhraseSelector()
    rows: List[Dict[str, Any]] = []
    try:
        for trial in range(1, args.rounds + 1):
            wake_wait = 0.0
            record_s = 0.0
            asr_s = 0.0
            asr_error = None
            ser_s = 0.0
            ser_error = None
            transcript = args.transcript
            keyword = None
            trigger = "mic"

            if args.wav:
                trigger = "wav"
                from pipeline.audio_emotion import infer_wav
                from pipeline.audio_mac import local_whisper_transcribe

                text, asr_error, asr_s = local_whisper_transcribe(
                    args.wav, model_size=args.whisper_size, language="zh"
                )
                transcript = text or ""
                ser = infer_wav(args.wav) if os.path.isfile(args.wav) else None
                ser_s = ser.elapsed_s if ser else 0.0
                ser_error = None if ser is None else ser.error
                keyword = "wakeup"
            elif use_echo and echo is not None:
                print("round %d: say the wake word into WonderEcho (timeout %.0fs)" % (
                    trial, args.wake_timeout,
                ), flush=True)
                keyword, wake_wait = _wait_keyword(echo, args.wake_timeout)
                if not keyword:
                    print("round %d: no keyword" % trial, flush=True)
                    rows.append(finish_round(
                        _perception_without_camera("", None),
                        trial=trial,
                        trigger="timeout",
                        requested_backend=args.decide_backend,
                        wake_wait_s=wake_wait,
                        asr_error="no_keyword",
                        selector=selector,
                        move=args.move,
                    ))
                    continue
                trigger = keyword
                if keyword == "wakeup":
                    wav = out_dir / ("wakeup_%d_%d.wav" % (int(time.time()), trial))
                    wav.parent.mkdir(parents=True, exist_ok=True)
                    print("round %d: recording %.1fs" % (trial, args.audio_sec), flush=True)
                    event = handle_echo_keyword(
                        "wakeup",
                        str(wav),
                        duration_sec=args.audio_sec,
                        whisper_size=args.whisper_size,
                    )
                    transcript = event.transcript or ""
                    record_s = event.record_s
                    asr_s = event.asr_s
                    asr_error = event.error
                    ser_s = event.ser_s
                    ser_error = event.ser_error
            else:
                wav = out_dir / ("mic_%d_%d.wav" % (int(time.time()), trial))
                wav.parent.mkdir(parents=True, exist_ok=True)
                print("round %d: speak now, recording %.1fs" % (trial, args.audio_sec), flush=True)
                event = handle_echo_keyword(
                    "wakeup",
                    str(wav),
                    duration_sec=args.audio_sec,
                    whisper_size=args.whisper_size,
                )
                trigger = "mic"
                keyword = "wakeup"
                transcript = event.transcript or ""
                record_s = event.record_s
                asr_s = event.asr_s
                asr_error = event.error
                ser_s = event.ser_s
                ser_error = event.ser_error

            if args.no_vision:
                perception = _perception_without_camera(transcript, keyword)
                vision_s = 0.0
                calib_s = 0.0
            else:
                perception, vision_s, calib_s = _capture_vision(args, trial, transcript, keyword)
            row = finish_round(
                perception,
                trial=trial,
                trigger=trigger,
                requested_backend=args.decide_backend,
                wake_wait_s=wake_wait,
                record_s=record_s,
                asr_s=asr_s,
                asr_error=asr_error,
                ser_s=ser_s,
                ser_error=ser_error,
                vision_s=vision_s,
                calib_s=calib_s,
                move=args.move,
                selector=selector,
            )
            rows.append(row)
            print(
                "round %d  识别到决策=%.3fs  录音=%.3f  语音识别=%.3f  人脸=%.3f  决策=%.3f (%s)  短语=%s" % (
                    trial,
                    row["识别到决策_s"],
                    row["录音_s"],
                    row["语音识别_s"],
                    row["人脸_s"],
                    row["决策_s"],
                    row["实际决策"],
                    row["短语"],
                ),
                flush=True,
            )
    finally:
        close_pi_vision_session()
        if echo is not None:
            echo.close()

    if not rows:
        print("no rounds recorded", flush=True)
        return 1
    csv_path = write_report(out_dir, rows, probe)
    print("csv: %s" % csv_path, flush=True)
    print("median 识别到决策_s: %s" % _median(rows, "识别到决策_s"), flush=True)
    return 0


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Record on-robot ASR + face + edge-decide latency")
    parser.add_argument("--probe", action="store_true", help="print what is installed, then exit")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--decide-backend", default=os.environ.get("DECIDE_BACKEND", "edge_auto"))
    parser.add_argument("--audio-sec", type=float, default=3.0)
    parser.add_argument("--whisper-size", default=os.environ.get("WHISPER_SIZE", "tiny"))
    parser.add_argument("--echo-port", default=os.environ.get("WONDERECHO_PORT", "/dev/ttyUSB0"))
    parser.add_argument("--wake-timeout", type=float, default=25.0)
    parser.add_argument("--mic-only", action="store_true", help="skip WonderEcho and record the mic directly")
    parser.add_argument("--wav", default="", help="transcribe this wav instead of recording")
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument("--camera-index", type=int, default=-1)
    parser.add_argument("--calibrate-frames", type=int, default=25)
    parser.add_argument("--sample-frames", type=int, default=8)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--transcript", default="")
    parser.add_argument("--move", action="store_true", help="also play the ActionGroup and time that")
    parser.add_argument("--out-dir", default=str(ROOT / "pipeline_runs" / "latency"))
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    return run_live(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
