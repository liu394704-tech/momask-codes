#!/usr/bin/env python3
"""Pi audio: WonderEcho keyword listener + wakeup-then local Whisper ASR.

No cloud ASR. If the serial dongle is missing, keyword_unavailable is recorded
and the vision loop can still run.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .actions import WONDERECHO_CMDS, keyword_phrase
from .audio_mac import local_whisper_transcribe, record_mic_wav


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def default_echo_port() -> str:
    return _env("WONDERECHO_PORT") or "/dev/ttyUSB0"


@dataclass
class AudioEvent:
    kind: str  # wakeup | keyword | asr | unavailable
    keyword: Optional[str] = None
    transcript: str = ""
    wav_path: Optional[str] = None
    asr_s: float = 0.0
    record_s: float = 0.0
    error: Optional[str] = None


@dataclass
class WonderEchoListener:
    """Non-blocking WonderEcho poller. Safe if pyserial / device is absent."""

    port: str = field(default_factory=default_echo_port)
    available: bool = False
    error: Optional[str] = None
    _handle: object = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def open(self) -> bool:
        with self._lock:
            if self._handle is not None:
                return True
            try:
                import serial  # type: ignore
            except ImportError:
                self.error = "pyserial_missing"
                self.available = False
                return False
            try:
                handle = serial.Serial(
                    None,
                    115200,
                    serial.EIGHTBITS,
                    serial.PARITY_NONE,
                    serial.STOPBITS_ONE,
                    timeout=0.02,
                )
                handle.rts = False
                handle.dtr = False
                handle.setPort(self.port)
                handle.open()
                handle.reset_input_buffer()
            except Exception as exc:  # noqa: BLE001
                self.error = "echo_open_failed:%s" % str(exc)[:160]
                self.available = False
                return False
            self._handle = handle
            self.available = True
            self.error = None
            return True

    def close(self) -> None:
        with self._lock:
            handle = self._handle
            self._handle = None
            self.available = False
            if handle is None:
                return
            try:
                handle.close()
            except Exception:
                pass

    def poll(self) -> Optional[str]:
        """Return one mapped keyword, or None. Does not block beyond serial timeout."""
        if not self.open():
            return None
        with self._lock:
            handle = self._handle
            if handle is None:
                return None
            try:
                raw = handle.read(5)
            except Exception as exc:  # noqa: BLE001
                self.error = "echo_read_failed:%s" % str(exc)[:160]
                return None
        if not raw:
            return None
        return WONDERECHO_CMDS.get(bytes(raw))


def record_and_transcribe(
    wav_path: str,
    duration_sec: float = 3.0,
    whisper_size: str = "tiny",
    language: str = "zh",
) -> Tuple[str, Optional[str], float, float]:
    """Record from the default mic, then local Whisper. Returns transcript, err, record_s, asr_s."""
    t0 = time.perf_counter()
    ok, rec_err = record_mic_wav(wav_path, duration_sec)
    record_s = time.perf_counter() - t0
    if not ok:
        return "", rec_err or "record_failed", record_s, 0.0
    text, asr_err, asr_s = local_whisper_transcribe(
        wav_path, model_size=whisper_size, language=language or None
    )
    return (text or "").strip(), asr_err, record_s, asr_s


def handle_echo_keyword(
    keyword: str,
    wav_path: str,
    duration_sec: float = 3.0,
    whisper_size: str = "tiny",
    language: str = "zh",
) -> AudioEvent:
    """Map a WonderEcho keyword. wakeup triggers ASR; others are immediate commands."""
    name = (keyword or "").strip().lower()
    if name == "wakeup":
        text, err, record_s, asr_s = record_and_transcribe(
            wav_path, duration_sec=duration_sec, whisper_size=whisper_size, language=language
        )
        return AudioEvent(
            kind="asr" if text or not err else "wakeup",
            keyword="wakeup",
            transcript=text,
            wav_path=wav_path,
            asr_s=asr_s,
            record_s=record_s,
            error=err,
        )
    phrase = keyword_phrase(name)
    return AudioEvent(kind="keyword", keyword=name, transcript=phrase)
