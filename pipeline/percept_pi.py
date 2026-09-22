#!/usr/bin/env python3
"""Raspberry Pi camera + FaceExpression geometry emotion -> Perception."""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Optional, Tuple

from .actions import merge_keyword_transcript
from .percept_mac import _FUNCTIONS, _vote, close_vision_session
from .schemas import Perception

_SESSION: Optional["PiVisionSession"] = None

_HIWONDER_CANDIDATES = (
    "/home/pi/TonyPi",
    "/home/cat/TonyPi",
    os.path.join(os.path.dirname(_FUNCTIONS), "HiwonderSDK"),
    os.path.join(os.path.dirname(os.path.dirname(_FUNCTIONS)), "HiwonderSDK"),
)


def _ensure_robot_paths() -> None:
    if _FUNCTIONS not in sys.path:
        sys.path.insert(0, _FUNCTIONS)
    for root in _HIWONDER_CANDIDATES:
        if not root:
            continue
        if os.path.isdir(root) and root not in sys.path:
            sys.path.append(root)
        sdk = os.path.join(root, "HiwonderSDK") if not root.endswith("HiwonderSDK") else root
        if os.path.isdir(sdk) and sdk not in sys.path:
            sys.path.append(sdk)


def _open_cv2_camera(index: int):
    import cv2

    order = [index]
    for extra in (-1, 0, 1, 2):
        if extra not in order:
            order.append(extra)
    for idx in order:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap
        cap.release()
    return None


def _open_pi_camera(index: int = -1) -> Tuple[Optional[str], object]:
    """Prefer hiwonder.Camera on the robot, else OpenCV V4L2."""
    _ensure_robot_paths()
    try:
        import hiwonder.Camera as Camera  # type: ignore

        cam = Camera.Camera()
        cam.camera_open()
        time.sleep(0.25)
        ok, frame = cam.read()
        if ok and frame is not None:
            return "hiwonder", cam
        try:
            cam.camera_close()
        except Exception:
            pass
    except Exception:
        pass
    cap = _open_cv2_camera(index)
    if cap is not None:
        return "cv2", cap
    return None, None


def _read_frame(backend: str, cap) -> Tuple[bool, object]:
    if cap is None:
        return False, None
    if backend == "hiwonder":
        return cap.read()
    return cap.read()


def _release_camera(backend: Optional[str], cap) -> None:
    if cap is None:
        return
    try:
        if backend == "hiwonder":
            cap.camera_close()
        else:
            cap.release()
    except Exception:
        pass


class PiVisionSession:
    def __init__(self, camera_index: int, calibrate_frames: int, mirror: bool):
        import FaceExpression as FX  # type: ignore

        self.camera_index = camera_index
        self.mirror = mirror
        self.analyzer = FX.ExpressionAnalyzer(calibration_frames=calibrate_frames)
        self.backend, self.cap = _open_pi_camera(camera_index)
        self.calib_s = 0.0
        self.last_result = None

    @property
    def ready(self) -> bool:
        return self.cap is not None and bool(getattr(self.analyzer, "calibrated", False))

    def read_processed(self):
        import cv2

        ok, frame = _read_frame(self.backend or "", self.cap)
        if not ok or frame is None:
            return None, None
        if self.mirror:
            frame = cv2.flip(frame, 1)
        result = self.analyzer.process(frame)
        self.last_result = result
        return frame, result

    def close(self) -> None:
        _release_camera(self.backend, self.cap)
        self.cap = None
        try:
            self.analyzer.close()
        except Exception:
            pass


def close_pi_vision_session() -> None:
    global _SESSION
    if _SESSION is not None:
        _SESSION.close()
        _SESSION = None
    close_vision_session()


def _ensure_session(camera_index: int, calibrate_frames: int, mirror: bool) -> PiVisionSession:
    global _SESSION
    if (
        _SESSION is not None
        and _SESSION.camera_index == camera_index
        and _SESSION.cap is not None
    ):
        return _SESSION
    close_pi_vision_session()
    _ensure_robot_paths()
    _SESSION = PiVisionSession(camera_index, calibrate_frames, mirror)
    return _SESSION


def get_pi_vision_session(
    camera_index: int = -1, calibrate_frames: int = 25, mirror: bool = False
) -> PiVisionSession:
    return _ensure_session(camera_index, calibrate_frames, mirror)


def capture_perception_pi(
    session_id: str,
    camera_index: int = -1,
    calibrate_frames: int = 25,
    sample_frames: int = 12,
    mirror: bool = False,
    transcript: str = "",
    keyword: Optional[str] = None,
    keep_open: bool = True,
) -> Perception:
    """Sample FaceExpression on the Pi camera into a Perception object."""
    t0 = time.perf_counter()
    extras: dict[str, Any] = {
        "vision_backend": "FaceExpression.geometry_emotion",
        "platform": "pi",
    }
    if keyword:
        extras["keyword"] = keyword
    transcript = merge_keyword_transcript(keyword, transcript)

    try:
        import cv2  # noqa: F401
    except ImportError as exc:
        extras["keyword_unavailable"] = extras.get("keyword")
        return Perception(
            session_id=session_id,
            ts_ms=int(time.time() * 1000),
            transcript=transcript,
            ok=False,
            error="cv2 missing: %s" % exc,
            extras=extras,
        )

    try:
        sess = _ensure_session(camera_index, calibrate_frames, mirror)
    except Exception as exc:  # noqa: BLE001
        return Perception(
            session_id=session_id,
            ts_ms=int(time.time() * 1000),
            transcript=transcript,
            ok=False,
            error="ExpressionAnalyzer init failed: %s" % exc,
            extras=extras,
        )

    if sess.cap is None:
        extras["camera"] = "unavailable"
        if not keep_open:
            close_pi_vision_session()
        return Perception(
            session_id=session_id,
            ts_ms=int(time.time() * 1000),
            transcript=transcript,
            ok=False,
            error="camera_unavailable",
            extras=extras,
        )

    extras["camera"] = sess.backend
    emotions = []
    score_acc: dict[str, list] = {}
    actions_last: list = []
    face_found = False
    try:
        if not sess.analyzer.calibrated:
            t_cal = time.perf_counter()
            deadline = t_cal + 8.0
            while time.perf_counter() < deadline:
                _frame, result = sess.read_processed()
                if result is None:
                    continue
                if result.found:
                    face_found = True
                if result.found and not result.calibrating:
                    break
            sess.calib_s = time.perf_counter() - t_cal
        extras["t_calib_s"] = round(sess.calib_s, 3)
        extras["calibrated"] = bool(sess.analyzer.calibrated)

        for _ in range(sample_frames):
            _frame, result = sess.read_processed()
            if result is None:
                continue
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
            close_pi_vision_session()

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
