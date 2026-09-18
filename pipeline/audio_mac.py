#!/usr/bin/env python3
"""Mac microphone record + cloud Whisper ASR helpers."""
from __future__ import annotations

import os
import platform
import shutil
import struct
import subprocess
import threading
import time
import wave
from collections import deque
from typing import Optional, Tuple


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def record_wav_pyaudio(out_path: str, duration_sec: float, sample_rate: int = 16000) -> Tuple[bool, str]:
    try:
        import pyaudio  # type: ignore
    except ImportError:
        return False, "pyaudio not installed"

    dev_raw = _env("LIVE_MONITOR_PYAUDIO_DEVICE_INDEX", "")
    input_device_index = int(dev_raw) if dev_raw.isdigit() else None
    chunk = 1024
    frames = []
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=chunk,
        )
        n_read = max(1, int(round(sample_rate * duration_sec / float(chunk))))
        for _ in range(n_read):
            frames.append(stream.read(chunk, exception_on_overflow=False))
        stream.stop_stream()
        stream.close()
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:220]
    finally:
        pa.terminate()

    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
    if not os.path.isfile(out_path) or os.path.getsize(out_path) < 256:
        return False, "wav too small"
    return True, ""


def record_wav_ffmpeg(out_path: str, duration_sec: float, sample_rate: int = 16000) -> Tuple[bool, str]:
    ff = shutil.which("ffmpeg")
    if not ff:
        return False, "ffmpeg not found"
    dur = str(max(1, int(round(duration_sec))))
    system = platform.system()
    if system == "Darwin":
        av = _env("LIVE_MONITOR_AVFOUNDATION", "none:0") or "none:0"
        cmd = [
            ff, "-loglevel", "error", "-nostdin", "-y",
            "-f", "avfoundation", "-i", av,
            "-t", dur, "-ac", "1", "-ar", str(sample_rate), "-f", "wav", out_path,
        ]
    elif system == "Linux":
        alsa = _env("LIVE_MONITOR_ALSA_DEVICE", "default") or "default"
        cmd = [
            ff, "-loglevel", "error", "-nostdin", "-y",
            "-f", "alsa", "-i", alsa,
            "-t", dur, "-ac", "1", "-ar", str(sample_rate), "-f", "wav", out_path,
        ]
    else:
        return False, "unsupported OS for ffmpeg mic"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=float(duration_sec) + 25)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:220]
    if r.returncode != 0:
        return False, ((r.stderr or r.stdout or "")[:220] or "ffmpeg failed")
    if not os.path.isfile(out_path) or os.path.getsize(out_path) < 256:
        return False, "ffmpeg wav too small"
    return True, ""


def record_mic_wav(out_path: str, duration_sec: float) -> Tuple[bool, str]:
    ok, err = record_wav_pyaudio(out_path, duration_sec)
    if ok:
        return True, ""
    ok2, err2 = record_wav_ffmpeg(out_path, duration_sec)
    if ok2:
        return True, ""
    return False, "PyAudio: %s | ffmpeg: %s" % (err, err2)


def wav_rms_int16(wav_path: str) -> float:
    """Mean RMS of a 16-bit wav. Silence in the last live run was ~60."""
    if not os.path.isfile(wav_path):
        return 0.0
    with wave.open(wav_path, "rb") as wf:
        n, sw, ch = wf.getnframes(), wf.getsampwidth(), wf.getnchannels()
        raw = wf.readframes(n)
    if sw != 2 or not raw:
        return 0.0
    n_samp = len(raw) // 2
    samples = struct.unpack("<%dh" % n_samp, raw)
    if not samples:
        return 0.0
    acc = 0.0
    for s in samples:
        acc += float(s) * float(s)
    return (acc / float(len(samples))) ** 0.5


class MicRingBuffer:
    """Keep the last few seconds of mic audio so capture wait can be ~0."""

    def __init__(
        self,
        sample_rate: int = 16000,
        ring_sec: float = 3.0,
        chunk: int = 1024,
    ):
        self.sample_rate = int(sample_rate)
        self.ring_sec = float(ring_sec)
        self.chunk = int(chunk)
        self._max_samples = int(self.sample_rate * self.ring_sec)
        self._lock = threading.Lock()
        self._chunks = deque()
        self._n_samples = 0
        self._stop = threading.Event()
        self._thread = None
        self._err = ""
        self._ready_samples = 0

    @property
    def error(self) -> str:
        return self._err

    def filled_sec(self) -> float:
        with self._lock:
            n = self._n_samples
        return n / float(self.sample_rate) if self.sample_rate else 0.0

    def start(self) -> bool:
        if self._thread is not None:
            return True
        try:
            import pyaudio  # type: ignore
        except ImportError:
            self._err = "pyaudio not installed"
            return False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, args=(pyaudio,), name="mic-ring", daemon=True
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def snapshot_wav(self, out_path: str, duration_sec: float = 1.2) -> Tuple[bool, str, float, float]:
        """Write the last duration_sec. Returns (ok, err, rms, actual_sec)."""
        n_want = max(1, int(self.sample_rate * float(duration_sec)))
        with self._lock:
            chunks = list(self._chunks)
            n_have = self._n_samples
        if n_have < int(self.sample_rate * 0.25):
            return False, "mic_buffer_too_short", 0.0, 0.0
        raw = b"".join(chunks)
        n_samp = len(raw) // 2
        if n_samp > n_want:
            raw = raw[-(n_want * 2):]
            n_samp = n_want
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(raw)
        rms = wav_rms_int16(out_path)
        actual = n_samp / float(self.sample_rate)
        return True, "", rms, actual

    def _loop(self, pyaudio_mod) -> None:
        pa = pyaudio_mod.PyAudio()
        stream = None
        dev_raw = _env("LIVE_MONITOR_PYAUDIO_DEVICE_INDEX", "")
        input_device_index = int(dev_raw) if dev_raw.isdigit() else None
        try:
            stream = pa.open(
                format=pyaudio_mod.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=self.chunk,
            )
            while not self._stop.is_set():
                data = stream.read(self.chunk, exception_on_overflow=False)
                n = len(data) // 2
                with self._lock:
                    self._chunks.append(data)
                    self._n_samples += n
                    while self._n_samples > self._max_samples and self._chunks:
                        old = self._chunks.popleft()
                        self._n_samples -= len(old) // 2
        except Exception as exc:  # noqa: BLE001
            self._err = str(exc)[:220]
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            pa.terminate()


def local_whisper_transcribe(
    wav_path: str,
    model_size: str = "tiny",
    language: Optional[str] = None,
) -> Tuple[str, Optional[str], float]:
    """On-device ASR via openai-whisper. Returns (text, error, elapsed_s)."""
    t0 = time.perf_counter()
    try:
        model = _whisper_model(model_size)
        kwargs = {"fp16": False}
        lang = language if language is not None else (_env("WHISPER_LANGUAGE") or None)
        if lang:
            kwargs["language"] = lang
        out = model.transcribe(wav_path, **kwargs)
        text = (out.get("text") or "").strip()
        return text, None, time.perf_counter() - t0
    except ImportError:
        return "", "openai-whisper not installed", time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        return "", str(exc)[:300], time.perf_counter() - t0


_WHISPER = {}


def _whisper_model(model_size: str):
    import whisper  # type: ignore

    key = model_size or "tiny"
    if key not in _WHISPER:
        _WHISPER[key] = whisper.load_model(key)
    return _WHISPER[key]


def whisper_transcribe(
    wav_path: str,
    model: Optional[str] = None,
    timeout_s: float = 60.0,
) -> Tuple[str, Optional[str]]:
    """Return (transcript, error). error is None on success."""
    api_key = _env("OPENAI_API_KEY")
    if not api_key:
        return "", "OPENAI_API_KEY missing"
    base_url = _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = model or _env("WHISPER_MODEL", "gpt-4o-transcribe")
    try:
        from openai import OpenAI
    except ImportError:
        return "", "pip install openai"

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
    try:
        with open(wav_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
            )
        text = (transcription.text or "").strip()
        return text, None
    except Exception as exc:  # noqa: BLE001
        return "", str(exc)[:300]
