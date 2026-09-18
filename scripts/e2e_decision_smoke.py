#!/usr/bin/env python3
"""端到端联调：只测「决策层」JSON（不改主工程、不跑 MoMask / TonyPi）。

用法：
  # 1) 无 API：规则兜底，验证样例与评测脚本本身
  python scripts/e2e_decision_smoke.py --mode mock

  # 2) 小 LLM 决策（需已 export OPENAI_API_KEY / OPENAI_BASE_URL）
  export OPENAI_API_KEY=sk-...
  export OPENAI_BASE_URL=https://你的中转台/v1
  export OPENAI_MODEL=gpt-4o-mini   # 或中转台上的 qwen-turbo 等
  python scripts/e2e_decision_smoke.py --mode llm

输出：
  experiment/e2e_spec/decision_smoke_results.csv
  experiment/e2e_spec/decision_smoke_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "experiment/e2e_spec/e2e_cases_20.json"
OUT_DIR = ROOT / "experiment/e2e_spec"

ALLOWED_ACTIONS = {
    "stand", "go_forward", "back_fast", "left_move_fast", "right_move_fast",
    "turn_left", "turn_right", "wave", "bow", "squat", "chest", "twist",
    "left_shot_fast", "right_shot_fast", "sit_ups", "wing_chun", "stepping",
    "push_ups", "left_uppercut", "right_uppercut", "left_kick", "right_kick",
    "stand_up_front", "stand_up_back", "stand_slow", "jugong", "weightlifting",
}
ATTACK = {
    "left_shot_fast", "right_shot_fast", "left_uppercut", "right_uppercut",
    "left_kick", "right_kick", "wing_chun",
}

PLANNER_SYSTEM = """You are a robot action planner.
Given a perception JSON, output ONE JSON object only (no markdown):
{
  "emotion": "neutral|happy|sad|angry|fearful|disgust|surprised|unknown",
  "intent": "greeting|comfort_request|play|stop|help|unknown",
  "action_prompt": "English HumanML3D-style single-person motion sentence",
  "action_group": ["wave"],
  "response_text": "short reply",
  "motion_length_hint": 0,
  "confidence": 0.0,
  "fallback": false,
  "reason": "short"
}
Rules:
1) action_prompt MUST be concrete body motion English (not abstract advice).
2) action_group values must be from this whitelist only:
stand,go_forward,back_fast,left_move_fast,right_move_fast,turn_left,turn_right,wave,bow,squat,chest,twist,left_shot_fast,right_shot_fast,sit_ups,wing_chun,stepping
3) If fused.confidence < 0.45 OR emotion/intent unknown: fallback=true, action_group=["stand"], calm stand prompt.
4) If emotion is angry/fearful: NEVER use attack actions (shot/kick/punch/wing_chun); prefer stand/bow.
5) Use at most 3 action_group items.
"""


def load_cases(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list) and len(data) >= 1
    return data


def perception_payload(case: dict) -> dict:
    return {
        "session_id": case["id"],
        "vision": case.get("vision", {}),
        "audio": case.get("audio", {}),
        "fused": case.get("fused", {}),
    }


def mock_decide(case: dict) -> dict:
    fused = case.get("fused", {})
    emotion = fused.get("emotion") or "unknown"
    intent = fused.get("intent") or "unknown"
    conf = float(fused.get("confidence") or 0.0)
    fallback = conf < 0.45 or emotion == "unknown" or intent == "unknown" and conf < 0.5

    mapping = {
        ("happy", "greeting"): (["wave"], "a person waves cheerfully with the right hand"),
        ("happy", "play"): (["chest", "twist"], "a person celebrates by raising both arms and twists happily"),
        ("sad", "comfort_request"): (["bow"], "a person bows gently and slowly in a caring way"),
        ("angry", "help"): (["stand"], "a person stands still calmly with open relaxed arms"),
        ("angry", "unknown"): (["stand"], "a person stands still and keeps a calm posture"),
        ("fearful", "help"): (["stand"], "a person stands still and slowly raises both hands in a reassuring way"),
        ("fearful", "comfort_request"): (["bow"], "a person bows gently and steps slightly back"),
        ("disgust", "stop"): (["stand", "back_fast"], "a person steps back and waves a hand to refuse"),
        ("surprised", "unknown"): (["wave"], "a person raises both hands briefly in surprise then waves"),
        ("neutral", "unknown"): (["stand"], "a person stands still and nods once"),
        ("neutral", "play"): (["go_forward"], "a person walks forward a few steps"),
    }

    # intent-specific overrides from transcript keywords
    text = (case.get("audio") or {}).get("transcript") or ""
    if "前进" in text or "向前" in text:
        groups, prompt = ["go_forward"], "a person walks forward a few steps"
    elif "后退" in text:
        groups, prompt = ["back_fast"], "a person steps backward carefully"
    elif "左转" in text:
        groups, prompt = ["turn_left"], "a person turns left in place"
    elif "下蹲" in text:
        groups, prompt = ["squat"], "a person performs a squat and stands back up"
    elif "再见" in text:
        groups, prompt = ["wave"], "a person waves goodbye with the right hand"
    elif "扭" in text:
        groups, prompt = ["twist"], "a person twists the torso side to side playfully"
    else:
        groups, prompt = mapping.get((emotion, intent), (["stand"], "a person stands still naturally"))

    if fallback:
        groups, prompt = ["stand"], "a person stands still safely and waits"
    # safety
    if emotion in {"angry", "fearful"}:
        groups = [g for g in groups if g not in ATTACK] or ["stand"]

    return {
        "emotion": emotion,
        "intent": intent,
        "action_prompt": prompt,
        "action_group": groups,
        "response_text": "好的。",
        "motion_length_hint": 0,
        "confidence": conf,
        "fallback": fallback,
        "reason": "mock_rule_based",
    }


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def llm_decide(case: dict, model: str) -> tuple[dict, float]:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise SystemExit("llm 模式需要 OPENAI_API_KEY 与 OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url)
    user = json.dumps(perception_payload(case), ensure_ascii=False)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    dt = time.perf_counter() - t0
    content = resp.choices[0].message.content or ""
    return extract_json(content), dt


def evaluate(case: dict, decision: dict) -> dict:
    ok_parse = isinstance(decision, dict)
    groups = decision.get("action_group") if ok_parse else None
    prompt = (decision.get("action_prompt") or "") if ok_parse else ""
    checks = {
        "json_ok": ok_parse,
        "prompt_nonempty": bool(str(prompt).strip()),
        "prompt_is_englishish": bool(re.search(r"[A-Za-z]{3,}", prompt)),
        "groups_list": isinstance(groups, list),
        "groups_whitelist": isinstance(groups, list) and all(g in ALLOWED_ACTIONS for g in groups),
        "groups_len_ok": isinstance(groups, list) and len(groups) <= 3,
        "no_attack_when_forbidden": True,
        "expect_group_hit": True,
        "expect_fallback": True,
        "keyword_hit": True,
    }

    if not case.get("allow_attack", False) and isinstance(groups, list):
        checks["no_attack_when_forbidden"] = not any(g in ATTACK for g in groups)

    expect_any = case.get("expect_action_group_any") or []
    if expect_any and isinstance(groups, list):
        checks["expect_group_hit"] = any(g in groups for g in expect_any)

    if "expect_fallback" in case:
        checks["expect_fallback"] = bool(decision.get("fallback")) == bool(case["expect_fallback"])

    kws = [k.lower() for k in (case.get("expect_prompt_keywords") or [])]
    if kws and prompt:
        p = prompt.lower()
        checks["keyword_hit"] = any(k in p for k in kws)

    passed = all(checks.values())
    return {"passed": passed, **checks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(DEFAULT_CASES))
    ap.add_argument("--mode", choices=["mock", "llm"], default="mock")
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cases = load_cases(Path(args.cases))
    if args.limit > 0:
        cases = cases[: args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    latencies = []

    for case in cases:
        cid = case["id"]
        err = ""
        t_s = 0.0
        try:
            if args.mode == "mock":
                t0 = time.perf_counter()
                decision = mock_decide(case)
                t_s = time.perf_counter() - t0
            else:
                decision, t_s = llm_decide(case, args.model)
            ev = evaluate(case, decision)
        except Exception as e:
            decision = {}
            ev = {"passed": False, "json_ok": False}
            err = f"{type(e).__name__}: {e}"

        latencies.append(t_s)
        row = {
            "id": cid,
            "mode": args.mode,
            "passed": ev.get("passed", False),
            "decision_s": round(t_s, 4),
            "emotion": decision.get("emotion"),
            "intent": decision.get("intent"),
            "action_group": json.dumps(decision.get("action_group", []), ensure_ascii=False),
            "action_prompt": decision.get("action_prompt", ""),
            "fallback": decision.get("fallback"),
            "confidence": decision.get("confidence"),
            "error": err,
            **{f"chk_{k}": v for k, v in ev.items() if k != "passed"},
        }
        rows.append(row)
        flag = "PASS" if row["passed"] else "FAIL"
        print(f"[{flag}] {cid} {t_s:.3f}s group={row['action_group']} prompt={str(row['action_prompt'])[:48]}")

    csv_path = OUT_DIR / "decision_smoke_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    n_pass = sum(1 for r in rows if r["passed"])
    summary = {
        "mode": args.mode,
        "model": args.model if args.mode == "llm" else "mock_rules",
        "n": n,
        "n_pass": n_pass,
        "pass_rate": round(n_pass / n, 4) if n else 0.0,
        "decision_mean_s": round(statistics.mean(latencies), 4) if latencies else None,
        "decision_p95_s": round(sorted(latencies)[int(round(0.95 * (len(latencies) - 1)))], 4) if latencies else None,
        "results_csv": str(csv_path.relative_to(ROOT)),
    }
    summary_path = OUT_DIR / "decision_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n==== summary ====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
