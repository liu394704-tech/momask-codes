#!/usr/bin/env python3
# coding=utf8
"""WonderPi game slot: face + optional keyword -> 778 phrase ActionGroups.

TonyPi's app only has fixed buttons 1-12. This module matches the stock
FaceDetect contract (init/start/stop/exit/run) so WonderPi's existing
「人脸识别」button can launch it. The main TonyPi process already owns the
camera and passes each frame into run(); this module does not open another one.
"""
from __future__ import print_function

import os
import sys
import threading
import time

_RUNNING = False
_BUSY = False
_LOCK = threading.Lock()
_ANALYZER = None
_ANALYZER_ERROR = ""
_SCHEDULER = None
_SELECTOR = None
_ECHO = None
_ECHO_GEN = 0
_PENDING_KEYWORD = None
_STATUS = "idle"
_TRIAL = 0
_LAST_EMOTION = None
_LAST_CONF = 0.0
_LAST_VISION_S = 0.0
_FACE_HELD = 0
_ENTER_ON_START = True
_LATEST = None
_VISION_GEN = 0


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_paths():
    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    for functions in (
        "/home/pi/TonyPi/Functions",
        "/home/cat/TonyPi/Functions",
        os.path.join(root, "源码", "TonyPi", "Functions"),
    ):
        if functions and os.path.isdir(functions) and functions not in sys.path:
            sys.path.insert(0, functions)


def _load_analyzer():
    global _ANALYZER, _ANALYZER_ERROR
    if _ANALYZER is not None or _ANALYZER_ERROR:
        return _ANALYZER
    _ensure_paths()
    try:
        import FaceExpression as FX  # type: ignore

        _ANALYZER = FX.ExpressionAnalyzer(calibration_frames=25)
    except Exception as exc:  # noqa: BLE001
        _ANALYZER_ERROR = str(exc)[:180]
        print("EmotionPhrase analyzer:", _ANALYZER_ERROR)
    return _ANALYZER


def _scheduler():
    global _SCHEDULER
    if _SCHEDULER is None:
        _ensure_paths()
        from pipeline.emotion_scheduler import EmotionActionScheduler

        _SCHEDULER = EmotionActionScheduler()
    return _SCHEDULER


def _selector():
    global _SELECTOR
    if _SELECTOR is None:
        _ensure_paths()
        from pipeline.preset_select import PhraseSelector

        _SELECTOR = PhraseSelector()
    return _SELECTOR


def _overlay(img, text):
    try:
        import cv2
    except Exception:
        return img
    if img is None:
        return img
    cv2.putText(
        img, text[:70], (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
    )
    return img


def latency_csv_path():
    override = os.environ.get("WONDERPI_LATENCY_CSV", "").strip()
    if override:
        return override
    return os.path.join(_repo_root(), "pipeline_runs", "latency", "wonderpi_latency.csv")


def _decide_backend():
    return os.environ.get("DECIDE_BACKEND", "edge_auto").strip() or "edge_auto"


def _claim_busy():
    global _BUSY
    if not _scheduler().can_schedule():
        return False
    with _LOCK:
        if _BUSY:
            return False
        _BUSY = True
    return True


def play_and_log(
    emotion,
    confidence,
    keyword,
    transcript="",
    audio_emotion=None,
    audio_conf=0.0,
    trigger="face",
    record_s=0.0,
    asr_s=0.0,
    asr_error=None,
    ser_s=0.0,
    ser_error=None,
    wake_wait_s=0.0,
    vision_s=0.0,
    move=True,
):
    """Face + ASR text + audio emotion -> edge decide -> one of 778 phrases."""
    global _BUSY, _STATUS, _TRIAL
    try:
        _ensure_paths()
        from pipeline.actions import normalize_emotion
        from pipeline.bench_pi_latency import append_latency_row, finish_round
        from pipeline.schemas import Perception

        emo = normalize_emotion(emotion or "neutral")
        face_found = bool(emotion) and emo != "neutral"
        with _LOCK:
            _TRIAL += 1
            trial = _TRIAL
        perception = Perception(
            session_id="wonderpi",
            ts_ms=int(time.time() * 1000),
            vision_emotion=emo if emotion else None,
            vision_conf=float(confidence or 0.0),
            face_found=face_found or bool(emotion),
            transcript=transcript or "",
            audio_emotion=audio_emotion,
            audio_conf=float(audio_conf or 0.0),
            ok=True,
            extras={"keyword": keyword} if keyword else {},
        )
        row = finish_round(
            perception,
            trial=trial,
            trigger=trigger,
            requested_backend=_decide_backend(),
            wake_wait_s=wake_wait_s,
            record_s=record_s,
            asr_s=asr_s,
            asr_error=asr_error,
            ser_s=ser_s,
            ser_error=ser_error,
            vision_s=vision_s,
            move=move,
            selector=_selector(),
            scheduler=_scheduler(),
        )
        append_latency_row(latency_csv_path(), row)
        _STATUS = "%s %s %.2fs" % (row["人脸情绪"] or emo, row["短语"], row["识别到决策_s"])
        print("EmotionPhrase", _STATUS, "csv", latency_csv_path())
    except Exception as exc:  # noqa: BLE001
        _STATUS = "play failed"
        print("EmotionPhrase play:", exc)
    finally:
        with _LOCK:
            _BUSY = False


def face_should_move(stable, face_held):
    """A held face moves even when the expression stays neutral."""
    if stable:
        return stable
    if face_held >= 8:
        return "neutral"
    return None


def _maybe_launch(emotion, confidence, keyword):
    if not emotion and not keyword:
        return
    if not _claim_busy():
        return
    threading.Thread(
        target=play_and_log,
        kwargs={
            "emotion": emotion,
            "confidence": confidence,
            "keyword": keyword,
            "trigger": "keyword" if keyword else "face",
            "vision_s": _LAST_VISION_S,
            "move": True,
        },
        daemon=True,
    ).start()


def _play_wakeup(heard_at):
    """Record after the wake word, run local ASR, then the phrase loop."""
    try:
        _ensure_paths()
        from pipeline.audio_pi import handle_echo_keyword

        wav_dir = os.path.join(_repo_root(), "pipeline_runs", "latency")
        os.makedirs(wav_dir, exist_ok=True)
        wav = os.path.join(wav_dir, "wakeup_%d.wav" % int(time.time()))
        duration = float(os.environ.get("AUDIO_SEC", "3") or "3")
        before_record = time.perf_counter()
        event = handle_echo_keyword("wakeup", wav, duration_sec=duration)
        play_and_log(
            _LAST_EMOTION,
            _LAST_CONF,
            "wakeup",
            transcript=event.transcript or "",
            audio_emotion=event.audio_emotion,
            audio_conf=event.audio_conf,
            trigger="wakeup",
            record_s=event.record_s,
            asr_s=event.asr_s,
            asr_error=event.error,
            ser_s=event.ser_s,
            ser_error=event.ser_error,
            wake_wait_s=max(0.0, before_record - heard_at),
            vision_s=_LAST_VISION_S,
            move=True,
        )
    except Exception as exc:  # noqa: BLE001
        global _BUSY, _STATUS
        _STATUS = "asr failed"
        print("EmotionPhrase asr:", exc)
        with _LOCK:
            _BUSY = False


def _echo_loop(gen):
    global _PENDING_KEYWORD, _ECHO
    _ensure_paths()
    try:
        from pipeline.audio_pi import WonderEchoListener
    except Exception as exc:  # noqa: BLE001
        print("EmotionPhrase echo import:", exc)
        return
    listener = WonderEchoListener()
    if not listener.open():
        print("EmotionPhrase echo:", listener.error or "unavailable")
        return
    _ECHO = listener
    while _RUNNING and _ECHO_GEN == gen:
        try:
            word = listener.poll()
        except Exception:
            word = None
        if word == "wakeup":
            if _claim_busy():
                print("EmotionPhrase wakeup", flush=True)
                threading.Thread(
                    target=_play_wakeup, args=(time.perf_counter(),), daemon=True,
                ).start()
        elif word:
            _PENDING_KEYWORD = word
            print("EmotionPhrase keyword", word)
        time.sleep(0.05)
    try:
        listener.close()
    except Exception:
        pass


def stable_launch_emotion():
    """Return a non-neutral stable label. Neutral must not start the 12s lock."""
    voter = _scheduler().voter
    stable, conf = voter.vote()
    if not stable or stable == "neutral":
        voter._last_label = None
        voter._last_time = 0.0
        return None, 0.0
    return stable, conf


def _apply_vision(result, vision_s):
    """Update emotion from a frame already processed off the camera thread."""
    global _STATUS, _LAST_EMOTION, _LAST_CONF, _LAST_VISION_S, _FACE_HELD
    _LAST_VISION_S = vision_s
    emotion = None
    confidence = 0.0
    if result is not None and getattr(result, "found", False):
        if not getattr(result, "calibrating", False) and getattr(result, "emotion", None):
            emotion = result.emotion
            confidence = float(getattr(result, "emotion_score", 0.0) or 0.0)
            _LAST_EMOTION = emotion
            _LAST_CONF = confidence
            _FACE_HELD += 1
            _scheduler().observe(emotion, confidence)
            _STATUS = "%s %.2f" % (emotion, confidence)
        else:
            _FACE_HELD = 0
            _STATUS = "calibrating"
    else:
        _FACE_HELD = 0
        _STATUS = "no face"
    stable, conf = stable_launch_emotion()
    launch_emotion = face_should_move(stable, _FACE_HELD)
    if launch_emotion:
        _maybe_launch(launch_emotion, conf or confidence, None)


def _vision_loop(gen):
    global _STATUS
    analyzer = _load_analyzer()
    while _RUNNING and _VISION_GEN == gen:
        frame = _LATEST
        if analyzer is None or frame is None:
            time.sleep(0.05)
            continue
        try:
            t0 = time.perf_counter()
            result = analyzer.process(frame)
            _apply_vision(result, time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001
            _STATUS = "vision %s" % str(exc)[:40]
            print("EmotionPhrase vision:", exc)
        time.sleep(0.02)


def init():
    global _STATUS
    print("EmotionPhrase Init")
    _scheduler().reset() if hasattr(_scheduler(), "reset") else None
    _STATUS = "camera"


def _enter_motion():
    """Wave as soon as the WonderPi button opens, then one greeting phrase."""
    print("EmotionPhrase enter", flush=True)
    if not _claim_busy():
        return
    try:
        import hiwonder.ActionGroupControl as AGC  # type: ignore

        AGC.runActionGroup("wave")
    except Exception as exc:  # noqa: BLE001
        print("EmotionPhrase wave:", exc)
    play_and_log("happy", 0.66, None, trigger="enter", move=True)


def start():
    global _RUNNING, _STATUS, _ECHO_GEN, _FACE_HELD, _VISION_GEN
    _ECHO_GEN += 1
    _VISION_GEN += 1
    gen = _ECHO_GEN
    vision_gen = _VISION_GEN
    _FACE_HELD = 0
    _RUNNING = True
    _STATUS = "camera"
    print("EmotionPhrase Start")
    threading.Thread(target=_echo_loop, args=(gen,), daemon=True).start()
    threading.Thread(target=_vision_loop, args=(vision_gen,), daemon=True).start()
    if _ENTER_ON_START:
        threading.Thread(target=_enter_motion, daemon=True).start()


def stop():
    global _RUNNING, _STATUS
    _RUNNING = False
    _STATUS = "stopped"
    print("EmotionPhrase Stop")


def exit():
    global _RUNNING
    _RUNNING = False
    print("EmotionPhrase Exit")
    try:
        import hiwonder.ActionGroupControl as AGC  # type: ignore

        AGC.runActionGroup("stand_slow")
    except Exception as exc:  # noqa: BLE001
        print("EmotionPhrase stand:", exc)


def run(img):
    """Return the camera frame immediately so WonderPi never stays on loading."""
    global _PENDING_KEYWORD, _LATEST
    if img is None:
        return img
    if not _RUNNING:
        return img
    _LATEST = img
    keyword = _PENDING_KEYWORD
    _PENDING_KEYWORD = None
    if keyword:
        _maybe_launch(_LAST_EMOTION or "neutral", _LAST_CONF, keyword)
    if _ANALYZER_ERROR and _ANALYZER is None:
        return _overlay(img, _ANALYZER_ERROR)
    return _overlay(img, "phrase " + _STATUS)
