#!/usr/bin/env python3
"""On-device (Raspberry Pi CPU) decision-LLM benchmark via local ollama.
Talks ONLY to 127.0.0.1:11434 -> proves inference runs locally (works offline).
Reports per-call total latency, tokens/s, and p50/p95, plus stage timings.
"""
import json, time, urllib.request, statistics as st, socket, platform, sys

HOST = "http://127.0.0.1:11434/api/chat"
SYS = ('Output ONLY one compact JSON object: {"emotion":..,"intent":..,'
       '"confidence":..,"action_group":..,"action_prompt":"a person ..."}. '
       'action_prompt is ONE short English sentence. No markdown.')
SAMPLES = [
    'happy face big_smile, transcript 你好一起玩吧',
    'unhappy face frown, transcript 我今天好累',
    'surprised face brow_raise, transcript 哇你会动啦',
    'neutral face, transcript 停一下别动了',
    'happy face laugh, transcript 我们跳个舞',
]


def call(model, text):
    body = json.dumps({
        "model": model, "stream": False,
        "options": {"temperature": 0.2, "num_predict": 80},
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": "perception: " + text}],
    }).encode()
    req = urllib.request.Request(HOST, data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t = time.perf_counter()
    r = urllib.request.urlopen(req, timeout=300)
    o = json.loads(r.read())
    dt = time.perf_counter() - t
    ev = o.get("eval_count") or 0
    evd = (o.get("eval_duration") or 1) / 1e9
    ld = (o.get("load_duration") or 0) / 1e9
    tps = ev / evd if evd else 0.0
    c = o["message"]["content"]
    ok = False
    try:
        j = json.loads(c[c.find("{"):c.rfind("}") + 1])
        ok = all(k in j for k in ("emotion", "intent", "action_prompt"))
    except Exception:
        ok = False
    return dt, tps, ev, ld, ok, c


def bench(model, rounds=3):
    print("\n=== %s (local CPU) ===" % model, flush=True)
    lat = []
    jok = 0
    for tx in SAMPLES:
        for _ in range(rounds):
            try:
                dt, tps, ev, ld, ok, c = call(model, tx)
                lat.append(dt)
                jok += int(ok)
                print("  %.2fs  %.1f tok/s  out=%d json=%s  %s"
                      % (dt, tps, ev, ok, repr(c)[:80]), flush=True)
            except Exception as e:
                print("  ERR", str(e)[:100], flush=True)
    if lat:
        s = sorted(lat)
        p = lambda q: s[min(len(s) - 1, int(round(q * (len(s) - 1))))]
        print("  >> %s  n=%d  p50=%.2f p95=%.2f max=%.2f mean=%.2f  json_ok=%d/%d"
              % (model, len(s), p(.5), p(.95), max(s), st.fmean(s), jok, len(s)), flush=True)


def main():
    print("machine:", platform.machine(), "host:", socket.gethostname(), flush=True)
    models = sys.argv[1:] or ["qwen2.5:1.5b", "qwen2.5:0.5b"]
    try:
        call(models[0], "warmup")  # load weights into RAM (excluded)
    except Exception as e:
        print("warmup err:", str(e)[:100], flush=True)
    for m in models:
        bench(m)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
