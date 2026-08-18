#!/usr/bin/env python3
"""Cloud OpenAI-compatible Decide: perception JSON -> decision JSON + action_prompt."""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Optional

from .schemas import Decision, Perception

SYSTEM_PROMPT = """You are the decision module for a social robot.
Given multimodal perception JSON, output ONLY one JSON object with fields:
  emotion: one of neutral,happy,sad,angry,fearful,disgust,surprised,unhappy
  intent: one of greeting,comfort_request,play,stop,help,unknown
  confidence: number 0..1
  action_group: array of TonyPi action names from
    [stand,wave,bow,jugong,squat,chest,twist,stepping,back_fast]
    Prefer non-aggressive actions. Empty array allowed only with fallback=true.
  action_prompt: ONE English HumanML3D-style motion sentence (visible body actions,
    with manner adverbs like gently/slowly/excitedly). Never abstract only ("feels sad").
  motion_length_hint: int, 0 means auto
  fallback: boolean, true if unsure / low signal / unsafe
  reason: short English note

Rules:
- If user speech expresses a clear need, let intent follow the speech.
- Use facial emotion as tone; intensity mild -> softer motion wording.
- If face missing and no transcript, set fallback=true, action_group=["stand"], action_prompt="".
- Output JSON only, no markdown.
"""


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def mock_decide(perception: Perception) -> Decision:
    """Deterministic offline decide for baseline / CI without API key."""
    emo = perception.vision_emotion or "neutral"
    conf = float(perception.vision_conf or 0.0)
    transcript = (perception.transcript or "").strip()
    intent = "unknown"
    prompt = ""
    group = []
    fallback = False

    if not perception.face_found and not transcript:
        fallback = True
        group = ["stand"]
        conf = min(conf, 0.3)
    elif emo == "happy":
        intent = "greeting" if not transcript else "play"
        group = ["wave"] if (perception.vision_intensity or "mild") == "mild" else ["chest"]
        prompt = (
            "a person happily waves with the right hand and then stands with an open posture"
            if group[0] == "wave"
            else "a person excitedly pumps the chest and waves both hands in celebration"
        )
        conf = max(conf, 0.55)
    elif emo in ("unhappy", "sad", "angry"):
        intent = "comfort_request"
        group = ["bow"] if (perception.vision_intensity or "mild") == "mild" else ["squat"]
        prompt = (
            "a person gently bows the head and then slowly waves in a caring way"
            if group[0] == "bow"
            else "a person squats slightly with a subdued posture then gives a small reassuring wave"
        )
        conf = max(conf, 0.5)
    elif emo == "surprised":
        intent = "unknown"
        group = ["twist"] if (perception.vision_intensity or "mild") == "mild" else ["back_fast"]
        prompt = (
            "a person twists the torso in mild surprise then steps in place"
            if group[0] == "twist"
            else "a person steps back quickly in surprise then stands still"
        )
        conf = max(conf, 0.5)
    else:
        intent = "unknown"
        group = ["stand"]
        prompt = "a person stands still with a relaxed neutral posture"
        if conf < 0.45:
            fallback = True
            prompt = ""

    if transcript and ("累" in transcript or "陪" in transcript):
        intent = "comfort_request"
        group = ["bow"]
        prompt = (
            "a person gently bows and then waves slowly with one hand as if offering comfort"
        )
        conf = max(conf, 0.7)
        fallback = False

    return Decision(
        session_id=perception.session_id,
        emotion=emo if emo != "unhappy" else "sad",
        intent=intent,
        confidence=float(conf),
        action_prompt=prompt,
        action_group=group,
        motion_length_hint=0,
        fallback=fallback,
        reason="mock_decide",
        ok=True,
    )


def cloud_decide(
    perception: Perception,
    model: Optional[str] = None,
    timeout_s: float = 30.0,
) -> Decision:
    """Call OpenAI-compatible chat API. Requires OPENAI_API_KEY (+ optional BASE_URL)."""
    api_key = _env("OPENAI_API_KEY")
    if not api_key:
        return Decision(
            session_id=perception.session_id,
            emotion="neutral",
            intent="unknown",
            confidence=0.0,
            action_prompt="",
            action_group=["stand"],
            fallback=True,
            reason="missing_api_key",
            ok=False,
            error="OPENAI_API_KEY not set",
        )

    base_url = _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = model or _env("OPENAI_MODEL", "gpt-4o-mini")

    try:
        from openai import OpenAI
    except ImportError:
        return Decision(
            session_id=perception.session_id,
            emotion="neutral",
            intent="unknown",
            confidence=0.0,
            action_prompt="",
            action_group=["stand"],
            fallback=True,
            reason="openai_pkg_missing",
            ok=False,
            error="pip install openai",
        )

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
    user_payload = {
        "perception": perception.to_dict(),
        "hint": "Write a precise English action_prompt for MoMask text-to-motion.",
    }
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        content = resp.choices[0].message.content or ""
        data = _extract_json(content)
    except Exception as exc:  # noqa: BLE001
        return Decision(
            session_id=perception.session_id,
            emotion=perception.vision_emotion or "neutral",
            intent="unknown",
            confidence=0.0,
            action_prompt="",
            action_group=["stand"],
            fallback=True,
            reason="cloud_decide_error",
            ok=False,
            error=str(exc),
        )

    group = data.get("action_group") or []
    if isinstance(group, str):
        group = [group]
    conf = float(data.get("confidence") or 0.0)
    return Decision(
        session_id=perception.session_id,
        emotion=str(data.get("emotion") or "neutral"),
        intent=str(data.get("intent") or "unknown"),
        confidence=conf,
        action_prompt=str(data.get("action_prompt") or "").strip(),
        action_group=[str(x) for x in group],
        motion_length_hint=int(data.get("motion_length_hint") or 0),
        fallback=bool(data.get("fallback", False)),
        reason=str(data.get("reason") or ("cloud_decide %.2fs" % (time.time() - t0))),
        ok=True,
    )
