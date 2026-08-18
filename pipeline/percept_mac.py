#!/usr/bin/env python3
"""Mac perception: local geometry emotion (+ optional cloud/mic later)."""
from __future__ import annotations

import os
import sys
import time
from typing import Optional, Tuple

from .schemas import Perception

_FUNCTIONS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "源码",
    "TonyPi",
    "Functions",
)


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


def capture_perception_mac(
    session_id: str,
    camera_index: int = 0,
    calibrate_frames: int = 20,
    sample_frames: int = 12,
    mirror: bool = True,
    transcript: str = "",
) -> Perception:
    """Capture webcam frames and run local geometry emotion model."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        return Perception(
            session_id=session_id,
            ts_ms=int(time.time() * 1000),
            ok=False,
            error="cv2/numpy missing: %s" % exc,
        )

    if _FUNCTIONS not in sys.path:
        sys.path.insert(0, _FUNCTIONS)
    try:
        import FaceExpression as FX  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return Perception(
            session_id=session_id,
            ts_ms=int(time.time() * 1000),
            ok=False,
            error="FaceExpression import failed: %s" % exc,
        )

    try:
        analyzer = FX.ExpressionAnalyzer(calibration_frames=calibrate_frames)
    except Exception as exc:  # noqa: BLE001
        return Perception(
            session_id=session_id,
            ts_ms=int(time.time() * 1000),
            ok=False,
            error="ExpressionAnalyzer init failed: %s" % exc,
        )

    cap = _open_camera(camera_index)
    if cap is None:
        analyzer.close()
        return Perception(
            session_id=session_id,
            ts_ms=int(time.time() * 1000),
            ok=False,
            error="camera_unavailable",
        )

    emotions = []
    face_found = False
    try:
        # Warmup / calibration
        deadline = time.time() + 8.0
        while time.time() < deadline:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if mirror:
                frame = cv2.flip(frame, 1)
            result = analyzer.process(frame)
            if result.found:
                face_found = True
            if result.found and not result.calibrating and result.emotion:
                break
            time.sleep(0.03)

        for _ in range(sample_frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if mirror:
                frame = cv2.flip(frame, 1)
            result = analyzer.process(frame)
            if result.found:
                face_found = True
            if result.found and not result.calibrating and result.emotion:
                emotions.append((result.emotion, float(result.emotion_score or 0.0)))
            time.sleep(0.03)
    finally:
        cap.release()
        analyzer.close()

    if not emotions:
        return Perception(
            session_id=session_id,
            ts_ms=int(time.time() * 1000),
            face_found=face_found,
            transcript=transcript,
            ok=face_found,
            error=None if face_found else "no_stable_emotion",
            vision_emotion="neutral" if face_found else None,
            vision_conf=0.2 if face_found else 0.0,
            vision_intensity="mild",
        )

    # Majority emotion
    labels = [e for e, _ in emotions]
    best = max(set(labels), key=labels.count)
    confs = [c for e, c in emotions if e == best]
    conf = sum(confs) / max(len(confs), 1)
    intensity = "strong" if conf >= 0.55 else "mild"
    return Perception(
        session_id=session_id,
        ts_ms=int(time.time() * 1000),
        vision_emotion=best,
        vision_conf=float(conf),
        vision_intensity=intensity,
        face_found=True,
        transcript=transcript,
        ok=True,
    )
