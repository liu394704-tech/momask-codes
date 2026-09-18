#!/usr/bin/env python3
"""Benchmark decision-LLM latency from a (bare) Raspberry Pi via an
OpenAI-compatible relay/proxy. Stdlib only (no pip installs needed).

What it measures, per model, over synthetic multimodal perception samples:
  - TTFT   : time to first streamed content token (network + queue + prefill)
  - total  : full response latency
  - success/timeout/error counts (this is the "unstable network" risk you care about)
  - json_ok: whether the model returned a parseable Decision JSON
  - out_chars: response size (proxy for output tokens = latency)

Usage (on the Pi):
  export RELAY_API_KEY=sk-xxxx
  python3 scripts/bench_pi_decision.py \
      --base-url https://your-relay.example.com/v1 \
      --models "gpt-4o-mini,deepseek-chat,qwen-flash" \
      --rounds 3 --timeout 8 --out /tmp/bench_decision.csv

Notes:
  * --base-url may be ".../v1", ".../v1/chat/completions", or a bare host;
    the script normalizes to the chat/completions endpoint.
  * Streaming is used by default to measure TTFT. Use --no-stream to test
    plain latency (some relays don't forward SSE well).
  * The API key is read from --api-key or env RELAY_API_KEY. It is never
    printed or written to the CSV.
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Prompt: mirror the on-device SYSTEM_PROMPT so results transfer to production.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are the decision module for a social robot on-device.\n"
    "Given multimodal perception JSON, output ONLY one compact JSON object with fields:\n"
    "  emotion: one of neutral,happy,sad,angry,fearful,disgust,surprised,unhappy\n"
    "  intent: one of greeting,comfort_request,play,stop,help,unknown\n"
    "  confidence: number 0..1\n"
    "  action_group: array from [stand,wave,bow,jugong,squat,chest,twist,stepping,back_fast]\n"
    "  action_prompt: ONE English HumanML3D-style motion sentence starting with \"a person\"\n"
    "  motion_length_hint: int, 0 means auto\n"
    "  fallback: boolean\n"
    "  reason: short English note\n"
    "Output JSON only, no markdown, no explanation.\n"
)

FEWSHOT: List[Tuple[Dict[str, Any], Dict[str, Any]]] = [
    (
        {"vision_emotion": "happy", "vision_conf": 0.82, "face_actions": ["big_smile"],
         "transcript": "你好呀"},
        {"emotion": "happy", "intent": "greeting", "confidence": 0.85,
         "action_group": ["wave"],
         "action_prompt": "a person cheerfully waves with the right hand and stands with an open posture",
         "motion_length_hint": 0, "fallback": False, "reason": "smile+greeting"},
    ),
    (
        {"vision_emotion": "unhappy", "vision_conf": 0.61, "face_actions": ["frown"],
         "transcript": "我今天好累"},
        {"emotion": "sad", "intent": "comfort_request", "confidence": 0.7,
         "action_group": ["bow"],
         "action_prompt": "a person gently bows the head and then slowly waves in a caring way",
         "motion_length_hint": 0, "fallback": False, "reason": "tired->comfort"},
    ),
]

# Synthetic multimodal perception samples (aligned with pipeline.schemas.Perception).
SAMPLES: List[Dict[str, Any]] = [
    {"vision_emotion": "happy", "vision_conf": 0.88, "vision_intensity": "strong",
     "face_found": True, "face_actions": ["big_smile", "brow_raise"],
     "emotion_scores": {"happy": 0.88, "neutral": 0.08, "surprised": 0.03, "unhappy": 0.01},
     "transcript": "你好，今天天气真好"},
    {"vision_emotion": "neutral", "vision_conf": 0.44, "vision_intensity": "mild",
     "face_found": True, "face_actions": [],
     "emotion_scores": {"neutral": 0.6, "happy": 0.2, "unhappy": 0.1, "surprised": 0.1},
     "transcript": "陪我玩一会儿好不好"},
    {"vision_emotion": "unhappy", "vision_conf": 0.72, "vision_intensity": "strong",
     "face_found": True, "face_actions": ["frown", "brow_furrow"],
     "emotion_scores": {"unhappy": 0.72, "neutral": 0.2, "happy": 0.05, "surprised": 0.03},
     "transcript": "我有点难过"},
    {"vision_emotion": "surprised", "vision_conf": 0.66, "vision_intensity": "mild",
     "face_found": True, "face_actions": ["brow_raise", "mouth_open"],
     "emotion_scores": {"surprised": 0.66, "neutral": 0.2, "happy": 0.1, "unhappy": 0.04},
     "transcript": "哇，你会动啦"},
    {"vision_emotion": "neutral", "vision_conf": 0.5, "vision_intensity": "mild",
     "face_found": True, "face_actions": [],
     "emotion_scores": {"neutral": 0.7, "happy": 0.15, "unhappy": 0.1, "surprised": 0.05},
     "transcript": "停一下，别动了"},
    {"vision_emotion": "happy", "vision_conf": 0.79, "vision_intensity": "strong",
     "face_found": True, "face_actions": ["laugh_combo"],
     "emotion_scores": {"happy": 0.79, "neutral": 0.15, "surprised": 0.04, "unhappy": 0.02},
     "transcript": "我们一起跳个舞吧"},
    {"vision_emotion": "neutral", "vision_conf": 0.2, "vision_intensity": "mild",
     "face_found": False, "face_actions": [],
     "emotion_scores": {}, "transcript": ""},
    {"vision_emotion": "unhappy", "vision_conf": 0.55, "vision_intensity": "mild",
     "face_found": True, "face_actions": ["frown"],
     "emotion_scores": {"unhappy": 0.55, "neutral": 0.3, "happy": 0.1, "surprised": 0.05},
     "transcript": "帮我一下好吗"},
]

REQUIRED_KEYS = ("emotion", "intent", "confidence", "action_prompt")


def normalize_endpoint(base_url: str) -> str:
    b = base_url.strip().rstrip("/")
    if b.endswith("/chat/completions"):
        return b
    if b.endswith("/v1"):
        return b + "/chat/completions"
    return b + "/v1/chat/completions"


def build_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for pin, pout in FEWSHOT:
        msgs.append({"role": "user", "content": json.dumps({"perception": pin}, ensure_ascii=False)})
        msgs.append({"role": "assistant", "content": json.dumps(pout, ensure_ascii=False)})
    payload = {
        "perception": sample,
        "hint": "Write a precise English action_prompt for MoMask text-to-motion.",
    }
    msgs.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})
    return msgs


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    import re

    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    if not t.startswith("{"):
        m = re.search(r"\{.*\}", t, flags=re.S)
        if m:
            t = m.group(0)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


class Result:
    __slots__ = ("ok", "ttft", "total", "out_chars", "json_ok", "err", "content")

    def __init__(self) -> None:
        self.ok = False
        self.ttft: Optional[float] = None
        self.total: Optional[float] = None
        self.out_chars = 0
        self.json_ok = False
        self.err = ""
        self.content = ""


def call_once(
    endpoint: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    *,
    stream: bool,
    timeout: float,
    max_tokens: int,
    temperature: float,
    ctx: ssl.SSLContext,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Result:
    res = Result()
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if extra_body:
        body.update(extra_body)
    data = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
        "Accept": "text/event-stream" if stream else "application/json",
    }
    req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout, context=ctx)
    except urllib.error.HTTPError as e:
        res.err = "HTTP %s: %s" % (e.code, (e.read()[:180].decode("utf-8", "ignore")))
        res.total = time.perf_counter() - t0
        return res
    except Exception as e:  # noqa: BLE001
        res.err = "%s: %s" % (type(e).__name__, str(e)[:160])
        res.total = time.perf_counter() - t0
        return res

    parts: List[str] = []
    try:
        if stream:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                    delta = obj["choices"][0].get("delta", {})
                    piece = delta.get("content")
                except Exception:
                    piece = None
                if piece:
                    if res.ttft is None:
                        res.ttft = time.perf_counter() - t0
                    parts.append(piece)
            res.content = "".join(parts)
        else:
            payload = json.loads(resp.read().decode("utf-8", "ignore"))
            res.content = payload["choices"][0]["message"]["content"] or ""
            res.ttft = None
    except Exception as e:  # noqa: BLE001
        res.err = "stream_parse: %s" % (str(e)[:160])
        res.total = time.perf_counter() - t0
        return res

    res.total = time.perf_counter() - t0
    res.out_chars = len(res.content)
    res.ok = True
    res.json_ok = extract_json(res.content) is not None
    if res.json_ok:
        obj = extract_json(res.content) or {}
        if not all(k in obj for k in REQUIRED_KEYS):
            res.json_ok = False
    return res


def pct(vals: List[float], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def summarize(model: str, results: List[Result]) -> Dict[str, Any]:
    ok = [r for r in results if r.ok]
    totals = [r.total for r in ok if r.total is not None]
    ttfts = [r.ttft for r in ok if r.ttft is not None]
    return {
        "model": model,
        "n": len(results),
        "success": len(ok),
        "timeout_err": len(results) - len(ok),
        "json_ok": sum(1 for r in ok if r.json_ok),
        "ttft_p50": pct(ttfts, 50),
        "ttft_p95": pct(ttfts, 95),
        "total_p50": pct(totals, 50),
        "total_p95": pct(totals, 95),
        "total_max": max(totals) if totals else float("nan"),
        "total_mean": statistics.fmean(totals) if totals else float("nan"),
        "out_chars_mean": statistics.fmean([r.out_chars for r in ok]) if ok else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True, help="Relay base URL, e.g. https://relay/v1")
    ap.add_argument("--api-key", default=os.environ.get("RELAY_API_KEY", ""), help="or env RELAY_API_KEY")
    ap.add_argument("--models", required=True, help="comma-separated model ids on the relay")
    ap.add_argument("--rounds", type=int, default=3, help="repeats per sample")
    ap.add_argument("--timeout", type=float, default=8.0, help="per-request timeout seconds")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--no-stream", action="store_true", help="disable SSE (no TTFT measurement)")
    ap.add_argument("--insecure", action="store_true", help="skip TLS verify (relay with self-signed cert)")
    ap.add_argument("--limit-samples", type=int, default=0, help="use only first N synthetic samples (0=all)")
    ap.add_argument("--extra-body", default="", help='extra JSON merged into request body, '
                    'e.g. \'{"thinking":{"type":"disabled"}}\' to turn off reasoning')
    ap.add_argument("--out", default="", help="write per-request CSV to this path")
    args = ap.parse_args()

    extra_body: Optional[Dict[str, Any]] = None
    if args.extra_body.strip():
        try:
            extra_body = json.loads(args.extra_body)
        except Exception as e:  # noqa: BLE001
            print("ERROR: --extra-body is not valid JSON: %s" % e, file=sys.stderr)
            return 2

    if not args.api_key:
        print("ERROR: no API key (pass --api-key or set RELAY_API_KEY)", file=sys.stderr)
        return 2

    endpoint = normalize_endpoint(args.base_url)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    stream = not args.no_stream
    samples = SAMPLES if args.limit_samples <= 0 else SAMPLES[: args.limit_samples]
    ctx = ssl.create_default_context()
    if args.insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    print("endpoint  : %s" % endpoint)
    print("models    : %s" % ", ".join(models))
    print("stream    : %s | timeout=%.1fs | max_tokens=%d | rounds=%d | samples=%d"
          % (stream, args.timeout, args.max_tokens, args.rounds, len(samples)))
    if extra_body:
        print("extra_body: %s" % json.dumps(extra_body))
    print("-" * 78)

    csv_rows: List[List[Any]] = []
    summaries: List[Dict[str, Any]] = []

    for model in models:
        # warmup (excluded): open TLS + prefill cache
        try:
            call_once(endpoint, args.api_key, model, build_messages(samples[0]),
                      stream=stream, timeout=args.timeout, max_tokens=args.max_tokens,
                      temperature=args.temperature, ctx=ctx, extra_body=extra_body)
        except Exception:
            pass

        results: List[Result] = []
        for si, sample in enumerate(samples):
            msgs = build_messages(sample)
            for r in range(args.rounds):
                res = call_once(endpoint, args.api_key, model, msgs,
                                stream=stream, timeout=args.timeout,
                                max_tokens=args.max_tokens, temperature=args.temperature,
                                ctx=ctx, extra_body=extra_body)
                results.append(res)
                tag = "ok" if res.ok else "ERR"
                ttft_s = ("%.3f" % res.ttft) if res.ttft is not None else "-"
                tot_s = ("%.3f" % res.total) if res.total is not None else "-"
                print("  [%s] s%d r%d  ttft=%s total=%s json=%s %s"
                      % (model, si, r, ttft_s, tot_s, res.json_ok,
                         ("" if res.ok else "<%s>" % res.err)))
                csv_rows.append([model, si, r, int(res.ok), ttft_s, tot_s,
                                 int(res.json_ok), res.out_chars, res.err])
        summaries.append(summarize(model, results))
        print("-" * 78)

    print("\n=== SUMMARY (seconds) ===")
    hdr = ("%-22s %4s %4s %4s %6s | ttft50 ttft95 | tot50  tot95  totmax totmean | outc"
           % ("model", "n", "ok", "to", "json"))
    print(hdr)
    for s in summaries:
        print("%-22s %4d %4d %4d %6d | %6.3f %6.3f | %6.3f %6.3f %6.3f %6.3f | %4.0f"
              % (s["model"], s["n"], s["success"], s["timeout_err"], s["json_ok"],
                 s["ttft_p50"], s["ttft_p95"],
                 s["total_p50"], s["total_p95"], s["total_max"], s["total_mean"],
                 s["out_chars_mean"]))

    if args.out:
        import csv

        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "sample", "round", "ok", "ttft_s", "total_s",
                        "json_ok", "out_chars", "error"])
            w.writerows(csv_rows)
        print("\nper-request CSV -> %s" % args.out)

    print("\nHint: total_p95 是你真正要卡的数；配合 A 轨并行，只要 total_p95 + MoMask 生成"
          " 落在预算内即可。timeout/err 计数反映你担心的网络不稳。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
