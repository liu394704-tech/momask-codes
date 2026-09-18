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
    """Deterministic offline decide (alias of edge rules). No cloud API."""
    from .decide_edge import edge_rule_decide

    d = edge_rule_decide(perception)
    # keep historical reason tag for tests that may look at it
    if d.reason.startswith("edge_rule"):
        d.reason = "mock_decide:" + d.reason
    return d


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
