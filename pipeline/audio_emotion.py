#!/usr/bin/env python3
"""On-device speech emotion (SER). Hiwonder has no audio-emotion channel.

Default backend: Alibaba FunASR / ModelScope ``iic/emotion2vec_plus_seed``
(smallest emotion2vec+, Chinese + English, CPU). Not the cloud 中转站.

If funasr/weights are missing, infer() returns unavailable and vision/ASR
rules keep working.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# emotion2vec+ / common SER names -> TonyPi 4-class geometry space
_TO_ROBOT = {
    "happy": "happy",
    "excited": "happy",
    "joy": "happy",
    "sad": "unhappy",
    "sadness": "unhappy",
    "angry": "unhappy",
    "anger": "unhappy",
    "disgust": "unhappy",
    "disgusted": "unhappy",
    "fear": "unhappy",
    "fearful": "unhappy",
    "surprised": "surprised",
    "surprise": "surprised",
    "neutral": "neutral",
    "other": "neutral",
    "unknown": "neutral",
}


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def ser_enabled() -> bool:
    raw = _env("ENABLE_AUDIO_SER", "1").lower()
    return raw not in ("0", "false", "no", "off")


def default_model_id() -> str:
    return _env("AUDIO_SER_MODEL") or "iic/emotion2vec_plus_seed"


def default_cache_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return _env("AUDIO_SER_CACHE") or os.path.join(root, "models", "audio_ser")


def to_robot_emotion(label: Optional[str]) -> Optional[str]:
    """Map an open-source SER label onto happy/unhappy/surprised/neutral."""
    if not label:
        return None
    raw = str(label).strip().lower()
    if "/" in raw:
        raw = raw.split("/")[-1].strip()
    if raw in _TO_ROBOT:
        return _TO_ROBOT[raw]
    if raw in ("happy", "unhappy", "surprised", "neutral"):
        return raw
    return None


@dataclass
class SerResult:
    ok: bool
    label: Optional[str] = None
    robot_emotion: Optional[str] = None
    confidence: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)
    elapsed_s: float = 0.0
    backend: str = "emotion2vec_plus_seed"
    error: Optional[str] = None

    def to_extras(self) -> Dict[str, Any]:
        return {
            "ser_ok": self.ok,
            "ser_label": self.label,
            "ser_backend": self.backend,
            "ser_s": round(self.elapsed_s, 3),
            "ser_error": self.error,
            "ser_scores": self.scores,
        }


_MODEL = None
_MODEL_ERROR: Optional[str] = None


def _load_funasr():
    global _MODEL, _MODEL_ERROR
    if _MODEL is not None:
        return _MODEL
    if _MODEL_ERROR:
        raise RuntimeError(_MODEL_ERROR)
    model_id = default_model_id()
    cache = default_cache_dir()
    os.makedirs(cache, exist_ok=True)
    try:
        from funasr import AutoModel  # type: ignore
    except Exception as exc:  # noqa: BLE001
        _MODEL_ERROR = "funasr_missing:%s" % exc
        raise RuntimeError(_MODEL_ERROR)
    try:
        _MODEL = AutoModel(
            model=model_id,
            hub="ms",
            cache_dir=cache,
            disable_update=True,
            device="cpu",
        )
    except Exception as exc:  # noqa: BLE001
        _MODEL_ERROR = "ser_load_failed:%s" % str(exc)[:220]
        raise RuntimeError(_MODEL_ERROR)
    return _MODEL


def _parse_funasr(raw: Any) -> Tuple[Optional[str], float, Dict[str, float]]:
    payload = raw
    if isinstance(raw, list) and raw:
        payload = raw[0]
    if not isinstance(payload, dict):
        return None, 0.0, {}
    labels: List[str] = list(payload.get("labels") or payload.get("label") or [])
    scores_raw = payload.get("scores") or payload.get("score") or []
    if isinstance(labels, str):
        labels = [labels]
    if isinstance(scores_raw, (int, float)):
        scores_raw = [float(scores_raw)]
    scores: Dict[str, float] = {}
    for index, name in enumerate(labels):
        key = str(name)
        value = float(scores_raw[index]) if index < len(scores_raw) else 0.0
        scores[key] = value
    if not scores and payload.get("text"):
        return str(payload.get("text")), 0.5, {}
    if not scores:
        return None, 0.0, {}
    best = max(scores, key=scores.get)
    return best, float(scores[best]), scores


def infer_wav(wav_path: str) -> SerResult:
    """Run local SER on a 16 kHz wav. Never calls the cloud relay."""
    t0 = time.perf_counter()
    if not ser_enabled():
        return SerResult(ok=False, error="ser_disabled", elapsed_s=0.0)
    if not wav_path or not os.path.isfile(wav_path):
        return SerResult(ok=False, error="wav_missing", elapsed_s=0.0)
    if os.path.getsize(wav_path) < 256:
        return SerResult(ok=False, error="wav_too_small", elapsed_s=0.0)
    try:
        model = _load_funasr()
        raw = model.generate(wav_path, granularity="utterance", extract_embedding=False)
        label, conf, scores = _parse_funasr(raw)
    except Exception as exc:  # noqa: BLE001
        return SerResult(
            ok=False,
            error=str(exc)[:220],
            elapsed_s=time.perf_counter() - t0,
            backend=default_model_id(),
        )
    robot = to_robot_emotion(label)
    return SerResult(
        ok=bool(robot or label),
        label=label,
        robot_emotion=robot,
        confidence=float(conf or 0.0),
        scores=scores,
        elapsed_s=time.perf_counter() - t0,
        backend=default_model_id(),
    )
