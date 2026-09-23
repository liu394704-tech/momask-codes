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


def _play_phrase(emotion, confidence, keyword):
    global _BUSY, _STATUS
    try:
        _ensure_paths()
        from pipeline.actions import normalize_emotion
        from pipeline.decide_edge import edge_rule_decide
        from pipeline.schemas import Perception
        from pipeline.track_a import run_track_a

        emo = normalize_emotion(emotion)
        perception = Perception(
            session_id="wonderpi",
            ts_ms=int(time.time() * 1000),
            vision_emotion=emo,
            vision_conf=float(confidence or 0.0),
            face_found=True,
            transcript="",
            ok=True,
            extras={"keyword": keyword} if keyword else {},
        )
        decision = edge_rule_decide(perception)
        result = run_track_a(
            decision,
            simulate=False,
            execute_robot=True,
            keyword=keyword,
            scheduler=_scheduler(),
            perception=perception,
            selector=_selector(),
        )
        _STATUS = "%s %s" % (emo, result.action or "")
        print("EmotionPhrase", _STATUS, result.detail)
    except Exception as exc:  # noqa: BLE001
        _STATUS = "play failed"
        print("EmotionPhrase play:", exc)
    finally:
        with _LOCK:
            _BUSY = False


def _maybe_launch(emotion, confidence, keyword):
    global _BUSY
    if not emotion or emotion == "neutral":
        if not keyword:
            return
    scheduler = _scheduler()
    if not scheduler.can_schedule():
        return
    with _LOCK:
        if _BUSY:
            return
        _BUSY = True
    threading.Thread(
        target=_play_phrase,
        args=(emotion, confidence, keyword),
        daemon=True,
    ).start()


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
        if word:
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


def init():
    print("EmotionPhrase Init")
    _load_analyzer()
    _scheduler().reset() if hasattr(_scheduler(), "reset") else None
    global _STATUS
    _STATUS = "ready"


def start():
    global _RUNNING, _STATUS, _ECHO_GEN
    _ECHO_GEN += 1
    gen = _ECHO_GEN
    _RUNNING = True
    _STATUS = "look at the camera"
    print("EmotionPhrase Start")
    threading.Thread(target=_echo_loop, args=(gen,), daemon=True).start()


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
    global _PENDING_KEYWORD, _STATUS
    if img is None or not _RUNNING:
        return img
    analyzer = _load_analyzer()
    keyword = _PENDING_KEYWORD
    _PENDING_KEYWORD = None
    if analyzer is None:
        return _overlay(img, _ANALYZER_ERROR or "no FaceExpression")
    try:
        result = analyzer.process(img)
    except Exception as exc:  # noqa: BLE001
        return _overlay(img, "vision %s" % str(exc)[:40])
    emotion = None
    confidence = 0.0
    if result is not None and getattr(result, "found", False):
        if not getattr(result, "calibrating", False) and getattr(result, "emotion", None):
            emotion = result.emotion
            confidence = float(getattr(result, "emotion_score", 0.0) or 0.0)
            _scheduler().observe(emotion, confidence)
            _STATUS = "%s %.2f" % (emotion, confidence)
        else:
            _STATUS = "calibrating"
    else:
        _STATUS = "no face"
    stable, conf = stable_launch_emotion()
    launch_emotion = stable
    if keyword or launch_emotion:
        _maybe_launch(launch_emotion or emotion or "neutral", conf or confidence, keyword)
    return _overlay(img, "phrase " + _STATUS)
