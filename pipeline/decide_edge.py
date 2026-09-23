#!/usr/bin/env python3
"""Edge-side Decide: offline rules (+ optional local GGUF LLM). No OpenAI required.

Backends used via pipeline.decide.run_decide:
  - edge / rule  : deterministic Emotion+transcript -> action_prompt (default on Pi)
  - edge_llm     : llama-cpp-python + EDGE_LLM_GGUF (Qwen2.5-0.5B/1.5B Instruct Q4)
  - mock         : alias of edge rule (CI)
  - cloud        : Mac联调 only (OpenAI-compatible); not for Pi deployment
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .actions import merge_keyword_transcript, sanitize_action_group
from .schemas import Decision, Perception

SYSTEM_PROMPT = """You are the decision module for a social robot on-device.
Given multimodal perception JSON, output ONLY one JSON object with fields:
  emotion: one of neutral,happy,sad,angry,fearful,disgust,surprised,unhappy
  intent: one of greeting,comfort_request,play,stop,help,unknown
  confidence: number 0..1
  action_group: array of 1-3 hint names from the social allow-list
    (wave,bow,jugong,squat,chest,twist,stepping,hand poses, small steps).
    Track A phrase selector may compose a longer sequence from these hints.
  action_prompt: ONE English HumanML3D-style motion sentence starting with "a person"
  motion_length_hint: int, 0 means auto
  fallback: boolean
  reason: short English note
Priority: extras.keyword and spoken transcript override face emotion.
If intent is stop, action_group must be ["stand"].
Locomotion keywords map to go_forward / back_fast / turn_left / turn_right (one or two steps).
Never output kicks, punches, or wing_chun.
"""

# transcript keyword -> (intent, action_group, action_prompt)
_TRANSCRIPT_RULES: List[Tuple[Tuple[str, ...], str, List[str], str]] = [
    (
        ("停", "不要", "停止", "stop", "enough", "别动"),
        "stop",
        ["stand"],
        "a person stands still with both arms relaxed at the sides",
    ),
    (
        ("前进", "往前", "forward"),
        "play",
        ["go_forward"],
        "a person takes two small steps forward then stands still",
    ),
    (
        ("后退", "往后", "back"),
        "play",
        ["back_fast"],
        "a person takes two small steps backward then stands still",
    ),
    (
        ("左转", "turn left", "turn_left"),
        "play",
        ["turn_left"],
        "a person turns left in place then stands still",
    ),
    (
        ("右转", "turn right", "turn_right"),
        "play",
        ["turn_right"],
        "a person turns right in place then stands still",
    ),
    (
        ("你好", "hello", "hi", "hey", "早上好", "晚上好"),
        "greeting",
        ["wave"],
        "a person cheerfully waves with the right hand and then stands with an open posture",
    ),
    (
        ("累", "陪", "安慰", "难过", "伤心", "tired", "sad", "comfort", "lonely"),
        "comfort_request",
        ["bow"],
        "a person gently bows and then waves slowly with one hand as if offering comfort",
    ),
    (
        ("玩", "开心", "高兴", "play", "fun", "happy", "dance"),
        "play",
        ["chest"],
        "a person excitedly pumps the chest and waves both hands in celebration",
    ),
    (
        ("帮", "帮助", "help", "救"),
        "help",
        ["wave"],
        "a person raises one hand to wave for attention then stands ready to help",
    ),
]


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # tolerate leading prose: take first {...}
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            text = m.group(0)
    return json.loads(text)


def _match_transcript(transcript: str) -> Optional[Tuple[str, List[str], str]]:
    t = (transcript or "").strip().lower()
    if not t:
        return None
    for keys, intent, group, prompt in _TRANSCRIPT_RULES:
        for k in keys:
            if k.lower() in t:
                return intent, group, prompt
    return None


def _emotion_from_actions(actions: List[str]) -> Optional[str]:
    """Map FaceExpression AU-like tags onto the 4-class geometry labels."""
    s = set(actions or [])
    if "laugh_combo" in s or "big_smile" in s or "smile" in s:
        return "happy"
    if "surprise_combo" in s or ("brow_raise" in s and "mouth_open" in s):
        return "surprised"
    if (
        "unhappy_furrow" in s
        or "unhappy_brow_up" in s
        or "frown" in s
        or "brow_furrow" in s
        or "brow_inner_up" in s
    ):
        return "unhappy"
    return None


def edge_rule_decide(perception: Perception) -> Decision:
    """On-device analyzer: vision emotion + transcript keywords -> MoMask prompt.

    Always available on Pi. Does not call any cloud API.
    """
    emo = (perception.vision_emotion or "neutral").strip().lower()
    conf = float(perception.vision_conf or 0.0)
    intensity = (perception.vision_intensity or "mild").strip().lower()
    extras = getattr(perception, "extras", None) or {}
    keyword = extras.get("keyword")
    transcript = merge_keyword_transcript(keyword, perception.transcript or "")
    actions = list(getattr(perception, "face_actions", None) or [])
    if not actions:
        actions = list(extras.get("face_actions") or [])
    action_emo = _emotion_from_actions(actions)
    if action_emo and (emo in ("", "neutral") or conf < 0.50):
        emo = action_emo
        conf = max(conf, 0.55)
        intensity = "strong" if (
            "laugh_combo" in actions or "big_smile" in actions or "surprise_combo" in actions
        ) else intensity
    audio_emo = (perception.audio_emotion or extras.get("audio_emotion") or "").strip().lower()
    audio_conf = float(perception.audio_conf or extras.get("audio_conf") or 0.0)
    if audio_emo in ("sad", "angry"):
        audio_emo = "unhappy"
    used_audio = False
    if audio_emo in ("happy", "unhappy", "surprised", "neutral") and audio_conf >= 0.35:
        if not perception.face_found or emo in ("", "neutral") or conf < 0.50:
            emo = audio_emo
            conf = max(conf, audio_conf)
            intensity = "strong" if audio_conf >= 0.55 and audio_emo != "neutral" else intensity
            used_audio = True
    mild = intensity != "strong"

    intent = "unknown"
    group: List[str] = ["stand"]
    prompt = "a person stands still with a relaxed neutral posture"
    fallback = False
    reason = "edge_rule"

    hit = _match_transcript(transcript)
    if hit is not None:
        intent, group, prompt = hit
        conf = max(conf, 0.65)
        reason = "edge_rule:transcript"
    elif not perception.face_found and not transcript and not used_audio:
        fallback = True
        group = ["stand"]
        prompt = ""
        conf = min(conf, 0.3)
        reason = "edge_rule:no_signal"
    elif emo == "happy":
        intent = "greeting" if not transcript else "play"
        group = ["wave"] if mild else ["chest"]
        prompt = (
            "a person happily waves with the right hand and then stands with an open posture"
            if group[0] == "wave"
            else "a person excitedly pumps the chest and waves both hands in celebration"
        )
        conf = max(conf, 0.55)
        reason = "edge_rule:happy"
    elif emo in ("unhappy", "sad", "angry"):
        intent = "comfort_request"
        group = ["bow"] if mild else ["squat"]
        prompt = (
            "a person gently bows the head and then slowly waves in a caring way"
            if group[0] == "bow"
            else "a person squats slightly with a subdued posture then gives a small reassuring wave"
        )
        conf = max(conf, 0.5)
        reason = "edge_rule:%s" % emo
    elif emo == "surprised":
        intent = "unknown"
        group = ["twist"] if mild else ["back_fast"]
        prompt = (
            "a person twists the torso in mild surprise then steps in place"
            if group[0] == "twist"
            else "a person steps back quickly in surprise then stands still"
        )
        conf = max(conf, 0.5)
        reason = "edge_rule:surprised"
    else:
        # neutral / low conf: still emit a mild stand prompt so Track B can be tested;
        # Arbiter may degrade on low confidence — that is intentional safety.
        intent = "unknown"
        group = ["stand"]
        prompt = "a person stands still with a relaxed neutral posture"
        if conf < 0.35 and not transcript:
            fallback = True
            # keep a non-empty prompt only if face exists (A can still run)
            if not perception.face_found:
                prompt = ""
            reason = "edge_rule:weak_neutral"
        else:
            reason = "edge_rule:neutral"

    if used_audio:
        reason = reason + "|audio_ser"
    if actions:
        reason = reason + "|actions:" + ",".join(actions[:6])

    return Decision(
        session_id=perception.session_id,
        emotion=("sad" if emo == "unhappy" else emo) or "neutral",
        intent=intent,
        confidence=float(conf),
        action_prompt=prompt,
        action_group=group,
        motion_length_hint=0,
        fallback=fallback,
        reason=reason,
        ok=True,
        extras={"phrase_hint": intent},
    )


_LLM = None  # lazy singleton


def default_gguf_path() -> str:
    """Preferred on-device weight: Qwen2.5-1.5B-Instruct Q4_K_M GGUF."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(
        root, "models", "edge_llm", "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    )


def resolve_gguf_path() -> str:
    gguf = _env("EDGE_LLM_GGUF") or default_gguf_path()
    return gguf


def _load_llama():
    global _LLM
    if _LLM is not None:
        return _LLM
    gguf = resolve_gguf_path()
    if not os.path.isfile(gguf):
        raise FileNotFoundError(
            "Qwen GGUF missing: %s (set EDGE_LLM_GGUF or run scripts/pi_download_qwen_gguf.sh)"
            % gguf
        )
    from llama_cpp import Llama  # type: ignore

    n_ctx = int(_env("EDGE_LLM_N_CTX", "2048") or "2048")
    n_threads = int(_env("EDGE_LLM_N_THREADS", "4") or "4")
    _LLM = Llama(
        model_path=gguf,
        n_ctx=n_ctx,
        n_threads=n_threads,
        chat_format="chatml",
        verbose=False,
    )
    return _LLM


def edge_llm_decide(perception: Perception, timeout_s: float = 120.0) -> Decision:
    """On-device Qwen2.5 Instruct (GGUF) via llama-cpp-python.

    Official recommendation for this project: Qwen2.5-1.5B-Instruct Q4_K_M.
    Falls back to edge_rule if weight/runtime unavailable.
    """
    t0 = time.time()
    try:
        llm = _load_llama()
        user_payload = {
            "perception": perception.to_dict(),
            "hint": "Write a precise English action_prompt for MoMask text-to-motion.",
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        out = llm.create_chat_completion(
            messages=messages,
            max_tokens=350,
            temperature=0.2,
        )
        content = out["choices"][0]["message"]["content"] or ""
        data = _extract_json(content)
    except Exception as exc:  # noqa: BLE001
        fb = edge_rule_decide(perception)
        fb.reason = "edge_llm_fallback:%s" % fb.reason
        fb.error = str(exc)[:220]
        return fb

    group = data.get("action_group") or []
    if isinstance(group, str):
        group = [group]
    group = sanitize_action_group(group)
    conf = float(data.get("confidence") or 0.0)
    prompt = str(data.get("action_prompt") or "").strip()
    if prompt and not prompt.lower().startswith("a person"):
        prompt = "a person " + prompt.lstrip()
    intent = str(data.get("intent") or "unknown")
    if not group:
        fb = edge_rule_decide(perception)
        fb.reason = "edge_llm_empty_group:%s" % fb.reason
        return fb

    return Decision(
        session_id=perception.session_id,
        emotion=str(data.get("emotion") or perception.vision_emotion or "neutral"),
        intent=intent,
        confidence=conf,
        action_prompt=prompt,
        action_group=group,
        motion_length_hint=int(data.get("motion_length_hint") or 0),
        fallback=bool(data.get("fallback", False)),
        reason=str(data.get("reason") or ("edge_llm %.2fs" % (time.time() - t0))),
        ok=True,
        extras={"phrase_hint": intent},
    )


def edge_decide(perception: Perception) -> Decision:
    """Prefer on-device Qwen GGUF if weight exists; otherwise rules."""
    gguf = resolve_gguf_path()
    if os.path.isfile(gguf):
        os.environ.setdefault("EDGE_LLM_GGUF", gguf)
        return edge_llm_decide(perception)
    return edge_rule_decide(perception)
