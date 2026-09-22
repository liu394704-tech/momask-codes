#!/usr/bin/env python3
"""Safe ActionGroup allow-list and WonderEcho keyword mapping.

Used by Track A, edge Decide, and the Pi live loop. Kick / punch groups are
never admitted.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

# Social allow-list: original groups plus unused safe clips for phrase composition.
ACTION_ALLOWLIST = (
    "stand",
    "stand_slow",
    "wave",
    "bow",
    "jugong",
    "squat",
    "squat_down",
    "squat_up",
    "chest",
    "twist",
    "stepping",
    "left_hand",
    "right_hand",
    "lift_left_hand",
    "go_hand_up",
    "go_hand_up1",
    "back_fast",
    "back_one_step",
    "go_forward",
    "go_forward_one_small_step",
    "go_forward_one_step",
    "turn_left",
    "turn_right",
    "turn_left_small_step",
    "turn_right_small_step",
    "left_move_10",
    "right_move_10",
)

# Full-step locomotion used only for explicit voice commands, never chained in phrases.
LOCOMOTION_ACTIONS = ("go_forward", "back_fast", "turn_left", "turn_right")
LOCOMOTION_TIMES = 2

# Phrase playback uses these short clips once (no times=2).
SMALL_LOCOMOTION_ACTIONS = (
    "go_forward_one_small_step",
    "go_forward_one_step",
    "back_one_step",
    "turn_left_small_step",
    "turn_right_small_step",
    "left_move_10",
    "right_move_10",
)

LOCOMOTION_HINTS = {
    "go_forward": ("go_forward_one_small_step", "go_forward_one_step"),
    "back_fast": ("back_one_step",),
    "turn_left": ("turn_left_small_step",),
    "turn_right": ("turn_right_small_step",),
}

BLOCKED_ACTIONS = (
    "left_shot_fast",
    "right_shot_fast",
    "left_uppercut",
    "right_uppercut",
    "left_kick",
    "right_kick",
    "wing_chun",
)

# WonderEcho serial payloads from TonyPi voice_control_move.py
WONDERECHO_CMDS = {
    b"\xaa\x55\x03\x00\xfb": "wakeup",
    b"\xaa\x55\x02\x00\xfb": "sleep",
    b"\xaa\x55\x00\x01\xfb": "forward",
    b"\xaa\x55\x00\x02\xfb": "back",
    b"\xaa\x55\x00\x03\xfb": "turn_left",
    b"\xaa\x55\x00\x04\xfb": "turn_right",
}

# Hardware keyword -> ActionGroup (wakeup is not an action).
KEYWORD_TO_ACTION = {
    "wakeup": None,
    "sleep": "stand",
    "forward": "go_forward",
    "back": "back_fast",
    "turn_left": "turn_left",
    "turn_right": "turn_right",
}

# Injected into transcript so rule / LLM fusion sees Chinese intent.
KEYWORD_TO_PHRASE = {
    "wakeup": "",
    "sleep": "停",
    "forward": "前进",
    "back": "后退",
    "turn_left": "左转",
    "turn_right": "右转",
}

STOP_INTENTS = ("stop",)
STOP_KEYWORDS = ("sleep", "stop")
STOP_PHRASES = ("停", "不要", "停止", "stop", "enough", "别动")

LOCO_KEYWORDS = {
    "forward": "go_forward",
    "back": "back_fast",
    "turn_left": "turn_left",
    "turn_right": "turn_right",
}


def sanitize_action_group(names: Optional[Sequence[str]]) -> List[str]:
    """Keep only allow-listed names; drop blocked / unknown groups."""
    out: List[str] = []
    blocked = set(BLOCKED_ACTIONS)
    allow = set(ACTION_ALLOWLIST)
    for raw in names or []:
        name = str(raw or "").strip()
        if not name or name in blocked or name not in allow:
            continue
        if name not in out:
            out.append(name)
    return out


def first_allowed_action(names: Optional[Sequence[str]]) -> Optional[str]:
    cleaned = sanitize_action_group(names)
    return cleaned[0] if cleaned else None


def is_stop_signal(
    keyword: Optional[str] = None,
    transcript: str = "",
    intent: str = "",
) -> bool:
    if (keyword or "").strip().lower() in STOP_KEYWORDS:
        return True
    if (intent or "").strip().lower() in STOP_INTENTS:
        return True
    text = (transcript or "").strip().lower()
    return any(p.lower() in text for p in STOP_PHRASES)


def keyword_phrase(keyword: Optional[str]) -> str:
    if not keyword:
        return ""
    return KEYWORD_TO_PHRASE.get(str(keyword), "")


def merge_keyword_transcript(keyword: Optional[str], transcript: str = "") -> str:
    """Keyword / command outranks free ASR: prepend the mapped phrase."""
    phrase = keyword_phrase(keyword)
    text = (transcript or "").strip()
    if phrase and phrase not in text:
        return ("%s %s" % (phrase, text)).strip()
    return text


def locomotion_times(action: Optional[str]) -> int:
    if action in LOCOMOTION_ACTIONS:
        return LOCOMOTION_TIMES
    return 1


def locomotion_hint_from_group(names: Optional[Sequence[str]]) -> Optional[str]:
    """Return a full-step locomotion name if the decision hinted at one."""
    for name in sanitize_action_group(names):
        if name in LOCOMOTION_HINTS or name in LOCOMOTION_ACTIONS:
            return name if name in LOCOMOTION_HINTS else name
    return None


def normalize_emotion(emotion: Optional[str]) -> str:
    emo = (emotion or "neutral").strip().lower()
    if emo in ("sad", "angry"):
        return "unhappy"
    if emo not in ("neutral", "happy", "unhappy", "surprised"):
        return "neutral"
    return emo
