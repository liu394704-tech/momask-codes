import json
import re
from typing import Any, Dict, Tuple


def extract_json_object(text: str) -> Dict[str, Any]:
    """Parse model output: strip fences, find outermost {...}."""
    s = text.strip()
    if "```" in s:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
        if m:
            s = m.group(1).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output:\n" + text[:2000])
    return json.loads(s[start : end + 1])


def validate_momask_plan(d: Dict[str, Any]) -> Tuple[str, int]:
    if "text_prompt" not in d:
        raise KeyError("JSON missing text_prompt")
    prompt = str(d["text_prompt"]).strip()
    if not prompt:
        raise ValueError("text_prompt is empty")
    ml = d.get("motion_length", 0)
    try:
        motion_length = int(ml)
    except (TypeError, ValueError):
        motion_length = 0
    if motion_length < 0:
        motion_length = 0
    if motion_length > 196:
        motion_length = 196
    return prompt, motion_length


def validate_vision_momask_plan(d: Dict[str, Any]) -> Tuple[str, int, str]:
    """Same as MoMask plan plus emotion label for session memory."""
    prompt, motion_length = validate_momask_plan(d)
    em = d.get("emotion", "unknown")
    emotion = str(em).strip() if em is not None else "unknown"
    if not emotion:
        emotion = "unknown"
    return prompt, motion_length, emotion
