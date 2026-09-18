#!/usr/bin/env python3
"""Quality eval of local qwen2.5 as the fusion->action_prompt decision model.
Tightened prompt (enum-constrained + few-shot). Runs on Pi via local ollama.
Shows, per scenario, the model's emotion/intent/action_group/action_prompt and
whether fields are valid, so you can judge if it's good enough for MoMask.
"""
import json, sys, time, urllib.request

HOST = "http://127.0.0.1:11434/api/chat"

EMOTIONS = {"neutral", "happy", "sad", "angry", "surprised", "unhappy"}
GROUPS = {"stand", "wave", "bow", "squat", "chest", "twist", "stepping", "back_fast"}
INTENTS = {"greeting", "comfort_request", "play", "stop", "help", "unknown"}

SYSTEM = (
    "You are the decision module of a social robot. You receive already-extracted "
    "multimodal perception (facial emotion from vision + speech transcript from audio) "
    "as text, and you FUSE them into one motion decision.\n"
    "Output ONLY one compact JSON object, no markdown, with EXACTLY these fields:\n"
    '  "emotion": one of [neutral,happy,sad,angry,surprised,unhappy]\n'
    '  "intent": one of [greeting,comfort_request,play,stop,help,unknown]\n'
    '  "confidence": number between 0 and 1\n'
    '  "action_group": one of [stand,wave,bow,squat,chest,twist,stepping,back_fast]\n'
    '  "action_prompt": ONE vivid English HumanML3D-style motion sentence, 8-20 words, '
    'MUST start with "a person" and describe a full-body/arm motion (not a facial expression).\n'
    "Rules: transcript intent overrides face when they conflict. If no face and no "
    "transcript, use intent=unknown, action_group=stand."
)

FEWSHOT = [
    ('face=happy(big_smile), transcript="你好，一起玩吧"',
     '{"emotion":"happy","intent":"play","confidence":0.9,"action_group":"chest",'
     '"action_prompt":"a person excitedly pumps both fists to the chest then waves both arms high in celebration"}'),
    ('face=unhappy(frown), transcript="我今天好累"',
     '{"emotion":"sad","intent":"comfort_request","confidence":0.8,"action_group":"bow",'
     '"action_prompt":"a person slowly bows the head then reaches out one hand gently as if to console someone"}'),
    ('face=neutral, transcript="停一下"',
     '{"emotion":"neutral","intent":"stop","confidence":0.85,"action_group":"stand",'
     '"action_prompt":"a person halts abruptly and stands still with both arms held down at the sides"}'),
]

SCENARIOS = [
    'face=happy(big_smile), transcript="你好呀"',
    'face=unhappy(frown,brow_furrow), transcript="我心情不太好"',
    'face=surprised(brow_raise,mouth_open), transcript="哇你会动啦"',
    'face=angry(brow_furrow), transcript="别烦我"',
    'face=neutral, transcript="过来帮我一下"',
    'face=happy(laugh), transcript="我们跳个舞吧"',
    'face=unhappy(frown), transcript=""            # 只有视频，无语音',
    'face=none, transcript="你好，在吗"             # 只有音频，无人脸',
]


def decide(model, user):
    msgs = [{"role": "system", "content": SYSTEM}]
    for u, a in FEWSHOT:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user})
    body = json.dumps({"model": model, "stream": False,
                       "options": {"temperature": 0.2, "num_predict": 160},
                       "messages": msgs}).encode()
    req = urllib.request.Request(HOST, data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t = time.perf_counter()
    r = urllib.request.urlopen(req, timeout=120)
    o = json.loads(r.read())
    return time.perf_counter() - t, o["message"]["content"]


def validate(txt):
    try:
        j = json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
    except Exception:
        return None, ["JSON parse fail"]
    issues = []
    if j.get("emotion") not in EMOTIONS:
        issues.append("emotion∉enum(%s)" % j.get("emotion"))
    if j.get("intent") not in INTENTS:
        issues.append("intent∉enum(%s)" % j.get("intent"))
    if j.get("action_group") not in GROUPS:
        issues.append("action_group∉enum(%s)" % j.get("action_group"))
    ap = str(j.get("action_prompt") or "")
    if not ap.lower().startswith("a person"):
        issues.append("prompt不以'a person'开头")
    if len(ap.split()) < 6:
        issues.append("prompt过短(%d词)" % len(ap.split()))
    return j, issues


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:1.5b"
    print("== 质量评测 model=%s ==" % model)
    try:
        decide(model, 'face=neutral, transcript="hi"')  # warmup
    except Exception:
        pass
    ok = 0
    lat = []
    for i, sc in enumerate(SCENARIOS, 1):
        user = sc.split("#")[0].strip()
        try:
            dt, out = decide(model, user)
            lat.append(dt)
            j, issues = validate(out)
            print("\n[%d] 输入: %s" % (i, sc))
            if j:
                print("    emotion=%-9s intent=%-15s group=%-16s (%.2fs)"
                      % (j.get("emotion"), j.get("intent"), j.get("action_group"), dt))
                print("    action_prompt: %s" % j.get("action_prompt"))
            else:
                print("    RAW:", out[:160])
            if not issues:
                ok += 1
                print("    ✅ 字段全部合法")
            else:
                print("    ⚠️  " + "; ".join(issues))
        except Exception as e:
            print("\n[%d] 输入: %s\n    ERR %s" % (i, sc, str(e)[:100]))
    n = len(SCENARIOS)
    avg = sum(lat) / len(lat) if lat else 0
    print("\n== 汇总: %d/%d 完全合法 | 平均 %.2fs ==" % (ok, n, avg))


if __name__ == "__main__":
    main()
