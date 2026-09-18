#!/usr/bin/env python3
"""Long-lived Mac backend: local Whisper + on-device Qwen + resident MoMask.

Run with momask_env. One JSON per stdin line, one JSON per stdout line.

  {"cmd":"ready"}
  {"cmd":"infer","wav":"...","perception":{...},"out_npy":"..."}
  {"cmd":"quit"}
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("EDGE_LLM_N_THREADS", "6")
os.environ.setdefault("EDGE_LLM_N_CTX", "2048")
os.environ.setdefault(
    "EDGE_LLM_GGUF",
    str(ROOT / "models" / "edge_llm" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
)


def _print(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    t_boot = time.perf_counter()
    real_out = sys.stdout
    sys.stdout = sys.stderr
    try:
        from pipeline.audio_mac import local_whisper_transcribe
        from pipeline.decide_edge import edge_llm_decide, _load_llama
        from pipeline.momask_runtime import ensure_loaded, generate
        from pipeline.schemas import Perception

        whisper_size = os.environ.get("WHISPER_LOCAL_MODEL", "tiny")
        t0 = time.perf_counter()
        try:
            from pipeline.audio_mac import _whisper_model

            _whisper_model(whisper_size)
            t_whisper_load = time.perf_counter() - t0
            whisper_load_err = None
        except Exception as exc:  # noqa: BLE001
            t_whisper_load = time.perf_counter() - t0
            whisper_load_err = str(exc)[:200]

        try:
            momask_load_s = ensure_loaded()
            momask_load_err = None
        except Exception as exc:  # noqa: BLE001
            momask_load_s = 0.0
            momask_load_err = str(exc)[:200]

        t2 = time.perf_counter()
        qwen_load_s = 0.0
        qwen_load_err = None
        try:
            _load_llama()
            qwen_load_s = time.perf_counter() - t2
        except Exception as exc:  # noqa: BLE001
            qwen_load_s = time.perf_counter() - t2
            qwen_load_err = str(exc)[:200]
    finally:
        sys.stdout = real_out

    _print(
        {
            "cmd": "booted",
            "boot_s": round(time.perf_counter() - t_boot, 3),
            "whisper_load_s": round(t_whisper_load, 3),
            "whisper_load_err": whisper_load_err,
            "qwen_load_s": round(qwen_load_s, 3),
            "qwen_load_err": qwen_load_err,
            "momask_load_s": round(float(momask_load_s), 3),
            "momask_load_err": momask_load_err,
        }
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        cmd = req.get("cmd")
        if cmd == "quit":
            _print({"cmd": "bye"})
            return 0
        if cmd != "infer":
            _print({"ok": False, "error": "unknown_cmd"})
            continue

        perc = Perception(
            **{
                k: v
                for k, v in (req.get("perception") or {}).items()
                if k in Perception.__dataclass_fields__
            }
        )
        wav = req.get("wav") or ""
        t_asr = 0.0
        asr_err = None
        real_out = sys.stdout
        sys.stdout = sys.stderr
        try:
            if wav and os.path.isfile(wav):
                text, asr_err, t_asr = local_whisper_transcribe(wav, model_size=whisper_size)
                if text:
                    perc.transcript = text
            elif wav:
                asr_err = "wav_missing"

            t_dec0 = time.perf_counter()
            decision = edge_llm_decide(perc)
            t_decide = time.perf_counter() - t_dec0

            joints_path = ""
            t_momask = 0.0
            load_s = 0.0
            gen_s = 0.0
            momask_err = None
            npy = req.get("out_npy") or ""
            if decision.action_prompt and npy:
                try:
                    path, load_s, gen_s, _t = generate(decision.action_prompt, Path(npy))
                    joints_path = str(path)
                    t_momask = gen_s
                except Exception as exc:  # noqa: BLE001
                    momask_err = str(exc)[:300]
        finally:
            sys.stdout = real_out

        _print(
            {
                "ok": bool(decision.ok) and not momask_err,
                "transcript": perc.transcript,
                "t_asr_s": round(t_asr, 3),
                "asr_error": asr_err,
                "t_decide_s": round(t_decide, 3),
                "emotion": decision.emotion,
                "intent": decision.intent,
                "confidence": decision.confidence,
                "action_prompt": decision.action_prompt,
                "action_group": decision.action_group,
                "fallback": decision.fallback,
                "decide_error": decision.error,
                "reason": decision.reason,
                "t_momask_s": round(t_momask, 3),
                "momask_load_s": round(load_s, 3),
                "momask_gen_s": round(gen_s, 3),
                "joints_path": joints_path,
                "momask_error": momask_err,
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
