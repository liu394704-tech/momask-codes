#!/usr/bin/env python3
"""Fast-path worker: local Whisper tiny + resident MoMask. No Qwen on this path.

Run with momask_env. One JSON per stdin line, one JSON per stdout line.

  {"cmd":"infer","wav":"...","skip_asr":true,"perception":{...},"out_npy":"..."}
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


def _print(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    t_boot = time.perf_counter()
    real_out = sys.stdout
    sys.stdout = sys.stderr
    try:
        from pipeline.audio_mac import local_whisper_transcribe
        from pipeline.decide_edge import edge_rule_decide
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
    finally:
        sys.stdout = real_out

    _print(
        {
            "cmd": "booted",
            "boot_s": round(time.perf_counter() - t_boot, 3),
            "whisper_load_s": round(t_whisper_load, 3),
            "whisper_load_err": whisper_load_err,
            "momask_load_s": round(float(momask_load_s), 3),
            "momask_load_err": momask_load_err,
            "decide": "rule",
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
        skip_asr = bool(req.get("skip_asr"))
        t_asr = 0.0
        asr_err = None
        real_out = sys.stdout
        sys.stdout = sys.stderr
        try:
            if not skip_asr and wav and os.path.isfile(wav):
                text, asr_err, t_asr = local_whisper_transcribe(
                    wav, model_size=whisper_size
                )
                if text:
                    perc.transcript = text
            elif not skip_asr and wav:
                asr_err = "wav_missing"

            t_dec0 = time.perf_counter()
            decision = edge_rule_decide(perc)
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
                "skip_asr": skip_asr,
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
