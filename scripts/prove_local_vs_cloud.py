#!/usr/bin/env python3
"""Proof helper: same Pi, one call to CLOUD relay, one to LOCAL ollama.
Used to demonstrate that the local model keeps working when the relay is
unreachable (proves on-device inference). Mode via argv[1]: cloud | local.
"""
import json, os, ssl, sys, time, urllib.request

RELAY = "https://www.dmxapi.cn/v1/chat/completions"
LOCAL = "http://127.0.0.1:11434/api/chat"
SYS = 'Output one JSON: {"emotion":..,"intent":..,"action_prompt":"a person ..."}. No markdown.'
USR = "perception: happy face, transcript 你好"


def cloud():
    key = os.environ.get("RELAY_API_KEY", "")
    body = json.dumps({"model": "qwen-flash", "stream": False, "max_tokens": 60,
                       "temperature": 0.2,
                       "messages": [{"role": "system", "content": SYS},
                                    {"role": "user", "content": USR}]}).encode()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(RELAY, data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key}, method="POST")
    t = time.perf_counter()
    r = urllib.request.urlopen(req, timeout=12, context=ctx)
    o = json.loads(r.read())
    return time.perf_counter() - t, o["choices"][0]["message"]["content"]


def local():
    body = json.dumps({"model": "qwen2.5:1.5b", "stream": False,
                       "options": {"temperature": 0.2, "num_predict": 60},
                       "messages": [{"role": "system", "content": SYS},
                                    {"role": "user", "content": USR}]}).encode()
    req = urllib.request.Request(LOCAL, data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t = time.perf_counter()
    r = urllib.request.urlopen(req, timeout=120)
    o = json.loads(r.read())
    return time.perf_counter() - t, o["message"]["content"]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "local"
    fn = cloud if mode == "cloud" else local
    try:
        dt, content = fn()
        print("[%s] OK  %.2fs  %s" % (mode.upper(), dt, repr(content)[:120]))
    except Exception as e:
        print("[%s] FAIL  %s: %s" % (mode.upper(), type(e).__name__, str(e)[:140]))


if __name__ == "__main__":
    main()
