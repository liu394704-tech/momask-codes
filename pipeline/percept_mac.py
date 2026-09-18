#!/usr/bin/env python3
"""Local geometry emotion. Camera stays open; calibrate once per process."""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Optional

from .schemas import Perception

_FUNCTIONS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "源码",
    "TonyPi",
    "Functions",
)

_SESSION: Optional["VisionSession"] = None


def _open_camera(index: int = 0):
    import cv2

    backends = []
    if hasattr(cv2, "CAP_AVFOUNDATION"):
        backends.append(cv2.CAP_AVFOUNDATION)
    backends.append(cv2.CAP_ANY)
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
        cap.release()
    return None


def mock_perception(session_id: str, emotion: str = "happy", conf: float = 0.7) -> Perception:
    intensity = "strong" if conf >= 0.55 else "mild"
    return Perception(
        session_id=session_id,
        ts_ms=int(time.time() * 1000),
        vision_emotion=emotion,
        vision_conf=conf,
        vision_intensity=intensity,
        face_found=True,
        transcript="",
        ok=True,
    )


class VisionSession:
    def __init__(self, camera_index: int, calibrate_frames: int, mirror: bool):
        import FaceExpression as FX  # type: ignore

        self.camera_index = camera_index
        self.mirror = mirror
        self.analyzer = FX.ExpressionAnalyzer(
            calibration_frames=calibrate_frames,
        )
        self.cap = _open_camera(camera_index)
        self.calib_s = 0.0

    @property
    def ready(self) -> bool:
        return self.cap is not None and bool(getattr(self.analyzer, "calibrated", False))

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        try:
            self.analyzer.close()
        except Exception:
            pass


def close_vision_session() -> None:
    global _SESSION
    if _SESSION is not None:
        _SESSION.close()
        _SESSION = None


def _ensure_session(camera_index: int, calibrate_frames: int, mirror: bool) -> VisionSession:
    global _SESSION
    if (
        _SESSION is not None
        and _SESSION.camera_index == camera_index
        and _SESSION.cap is not None
    ):
        return _SESSION
    close_vision_session()
    if _FUNCTIONS not in sys.path:
        sys.path.insert(0, _FUNCTIONS)
    _SESSION = VisionSession(camera_index, calibrate_frames, mirror)
    return _SESSION


def _vote(emotions):
    labels = [e for e, _ in emotions]
    best = max(set(labels), key=labels.count)
    confs = [c for e, c in emotions if e == best]
    conf = sum(confs) / max(len(confs), 1)
    intensity = "strong" if conf >= 0.55 else "mild"
    return best, float(conf), intensity


def capture_perception_mac(
    session_id: str,
    camera_index: int = 0,
    calibrate_frames: int = 25,
    sample_frames: int = 12,
    mirror: bool = True,
    transcript: str = "",
    keep_open: bool = True,
) -> Perception:
    """Read webcam frames via FaceExpression (FaceMesh + geometry_emotion_model).

    Calibrate the personal neutral baseline once per process, then sample a short
    window so the 4-class emotion, score vector, and AU-like action tags are kept.
    """
    t0 = time.perf_counter()
    extras: dict[str, Any] = {"vision_backend": "FaceExpression.geometry_emotion"}
    try:
        import cv2
    except ImportError as exc:
        return Perception(
            session_id=session_id,
            ts_ms=int(time.time() * 1000),
            ok=False,
            error="cv2 missing: %s" % exc,
        )

    try:
        sess = _ensure_session(camera_index, calibrate_frames, mirror)
    except Exception as exc:  # noqa: BLE001
        return Perception(
            session_id=session_id,
            ts_ms=int(time.time() * 1000),
            ok=False,
            error="ExpressionAnalyzer init failed: %s" % exc,
        )

    if sess.cap is None:
        if not keep_open:
            close_vision_session()
        return Perception(
            session_id=session_id,
            ts_ms=int(time.time() * 1000),
            ok=False,
            error="camera_unavailable",
        )

    emotions = []
    score_acc: dict[str, list] = {}
    actions_last: list = []
    face_found = False
    try:
        if not sess.analyzer.calibrated:
            t_cal = time.perf_counter()
            deadline = t_cal + 6.0
            while time.perf_counter() < deadline:
                ok, frame = sess.cap.read()
                if not ok or frame is None:
                    continue
                if mirror:
                    frame = cv2.flip(frame, 1)
                result = sess.analyzer.process(frame)
                if result.found:
                    face_found = True
                if result.found and not result.calibrating:
                    break
            sess.calib_s = time.perf_counter() - t_cal
        extras["t_calib_s"] = round(sess.calib_s, 3)
        extras["calibrated"] = bool(sess.analyzer.calibrated)

        for _ in range(sample_frames):
            ok, frame = sess.cap.read()
            if not ok or frame is None:
                continue
            if mirror:
                frame = cv2.flip(frame, 1)
            result = sess.analyzer.process(frame)
            if result.found:
                face_found = True
            if result.found and not result.calibrating and result.emotion:
                emotions.append((result.emotion, float(result.emotion_score or 0.0)))
                if result.emotion_scores:
                    for k, v in result.emotion_scores.items():
                        score_acc.setdefault(str(k), []).append(float(v))
                if result.actions:
                    actions_last = list(result.actions)
    finally:
        if not keep_open:
            close_vision_session()

    extras["t_vision_s"] = round(time.perf_counter() - t0, 3)
    extras["n_emotion_frames"] = len(emotions)
    mean_scores = {
        k: round(sum(vs) / max(len(vs), 1), 4) for k, vs in score_acc.items()
    }

    if not emotions:
        return Perception(
            session_id=session_id,
            ts_ms=int(time.time() * 1000),
            face_found=face_found,
            face_actions=actions_last,
            emotion_scores=mean_scores,
            transcript=transcript,
            ok=face_found,
            error=None if face_found else "no_stable_emotion",
            vision_emotion="neutral" if face_found else None,
            vision_conf=0.2 if face_found else 0.0,
            vision_intensity="mild",
            extras=extras,
        )

    best, conf, intensity = _vote(emotions)
    extras["raw_vote"] = best
    return Perception(
        session_id=session_id,
        ts_ms=int(time.time() * 1000),
        vision_emotion=best,
        vision_conf=conf,
        vision_intensity=intensity,
        face_found=True,
        face_actions=actions_last,
        emotion_scores=mean_scores,
        transcript=transcript,
        ok=True,
        extras=extras,
    )
