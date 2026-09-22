#!/usr/bin/env python3
"""Hand-authored 2–3 clip social phrases.

Each phrase is a short, recoverable sequence of allow-listed ActionGroups.
Locomotion uses one_step / _10 clips only — never chained go_forward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .actions import ACTION_ALLOWLIST, BLOCKED_ACTIONS
from .preset_catalog import CATALOG


@dataclass(frozen=True)
class Phrase:
    id: str
    clips: Tuple[str, ...]
    intent: str
    emotion_fit: Tuple[str, ...]
    energy: str
    tags: Tuple[str, ...]
    laterality: str


def _p(
    pid: str,
    clips: Tuple[str, ...],
    intent: str,
    emotion_fit: Tuple[str, ...],
    energy: str,
    tags: Tuple[str, ...],
    laterality: str,
) -> Phrase:
    return Phrase(pid, clips, intent, emotion_fit, energy, tags, laterality)


PHRASES: Tuple[Phrase, ...] = (
    # --- greeting (20) ---
    _p("greet_wave_step", ("wave", "stepping"), "greeting",
       ("happy", "neutral"), "mid", ("greet",), "right"),
    _p("greet_right_twist", ("right_hand", "twist"), "greeting",
       ("happy", "neutral"), "mid", ("greet", "torso"), "right"),
    _p("greet_left_sidestep", ("left_hand", "left_move_10"), "greeting",
       ("happy", "neutral"), "mid", ("greet", "locomote"), "left"),
    _p("greet_handup_turnr", ("go_hand_up", "turn_right_small_step"), "greeting",
       ("happy", "neutral"), "mid", ("greet", "locomote"), "right"),
    _p("greet_handup1_hold", ("go_hand_up1", "stand_slow"), "greeting",
       ("happy", "neutral"), "low", ("greet", "idle"), "both"),
    _p("greet_lift_sider", ("lift_left_hand", "right_move_10"), "greeting",
       ("happy", "neutral"), "mid", ("greet", "locomote"), "left"),
    _p("greet_chest_turnl", ("chest", "turn_left_small_step"), "greeting",
       ("happy",), "high", ("greet", "celebrate"), "both"),
    _p("greet_twist_fwd", ("twist", "go_forward_one_small_step"), "greeting",
       ("happy", "surprised"), "mid", ("greet", "torso"), "none"),
    _p("greet_step_righthand", ("stepping", "right_hand"), "greeting",
       ("happy", "neutral"), "mid", ("greet",), "right"),
    _p("greet_sidel_wave", ("left_move_10", "wave"), "greeting",
       ("happy", "neutral"), "mid", ("greet", "locomote"), "left"),
    _p("greet_turnl_lefthand", ("turn_left_small_step", "left_hand"), "greeting",
       ("happy", "neutral"), "mid", ("greet",), "left"),
    _p("greet_righthand_step", ("right_hand", "stepping"), "greeting",
       ("happy", "neutral"), "mid", ("greet",), "right"),
    _p("greet_wave_sider", ("wave", "right_move_10"), "greeting",
       ("happy",), "mid", ("greet", "locomote"), "right"),
    _p("greet_lefthand_twist", ("left_hand", "twist"), "greeting",
       ("happy", "neutral"), "mid", ("greet", "torso"), "left"),
    _p("greet_handup_step", ("go_hand_up", "stepping"), "greeting",
       ("happy", "neutral"), "mid", ("greet",), "both"),
    _p("greet_lift_twist", ("lift_left_hand", "twist"), "greeting",
       ("happy", "neutral"), "mid", ("greet", "torso"), "left"),
    _p("greet_wave_turnl", ("wave", "turn_left_small_step"), "greeting",
       ("happy",), "mid", ("greet", "locomote"), "right"),
    _p("greet_handup1_twist", ("go_hand_up1", "twist"), "greeting",
       ("happy", "surprised"), "mid", ("greet", "torso"), "both"),
    _p("greet_right_fwd", ("right_hand", "go_forward_one_small_step"), "greeting",
       ("happy", "neutral"), "mid", ("greet", "locomote"), "right"),
    _p("greet_step_handup", ("stepping", "go_hand_up"), "greeting",
       ("happy", "neutral"), "mid", ("greet",), "both"),
    # --- comfort (16) ---
    _p("comfort_bow_hold", ("bow", "stand_slow"), "comfort_request",
       ("unhappy", "neutral"), "low", ("comfort",), "none"),
    _p("comfort_jugong_squat", ("jugong", "squat_down", "squat_up"), "comfort_request",
       ("unhappy",), "low", ("comfort", "torso"), "none"),
    _p("comfort_bow_squat", ("bow", "squat"), "comfort_request",
       ("unhappy",), "low", ("comfort",), "none"),
    _p("comfort_jugong_hold", ("jugong", "stand_slow"), "comfort_request",
       ("unhappy", "neutral"), "low", ("comfort",), "none"),
    _p("comfort_squat_bow", ("squat_down", "squat_up", "bow"), "comfort_request",
       ("unhappy",), "low", ("comfort", "torso"), "none"),
    _p("comfort_bow_step", ("bow", "stepping"), "comfort_request",
       ("unhappy", "neutral"), "low", ("comfort",), "none"),
    _p("comfort_jugong_step", ("jugong", "stepping"), "comfort_request",
       ("unhappy", "neutral"), "low", ("comfort",), "none"),
    _p("comfort_squat_hold", ("squat", "stand_slow"), "comfort_request",
       ("unhappy",), "low", ("comfort",), "none"),
    _p("comfort_bow_sidel", ("bow", "left_move_10"), "comfort_request",
       ("unhappy",), "low", ("comfort", "locomote"), "left"),
    _p("comfort_jugong_sider", ("jugong", "right_move_10"), "comfort_request",
       ("unhappy",), "low", ("comfort", "locomote"), "right"),
    _p("comfort_lefthand_bow", ("left_hand", "bow"), "comfort_request",
       ("unhappy", "neutral"), "low", ("comfort", "hand_left"), "left"),
    _p("comfort_righthand_jugong", ("right_hand", "jugong"), "comfort_request",
       ("unhappy", "neutral"), "low", ("comfort", "hand_right"), "right"),
    _p("comfort_hold_bow", ("stand_slow", "bow"), "comfort_request",
       ("unhappy", "neutral"), "low", ("comfort", "idle"), "none"),
    _p("comfort_squatdown_up", ("squat_down", "squat_up"), "comfort_request",
       ("unhappy", "neutral"), "low", ("comfort", "torso"), "none"),
    _p("comfort_twist_bow", ("twist", "bow"), "comfort_request",
       ("unhappy", "surprised"), "mid", ("comfort", "torso"), "none"),
    _p("comfort_step_jugong", ("stepping", "jugong"), "comfort_request",
       ("unhappy", "neutral"), "low", ("comfort",), "none"),
    # --- play (16) ---
    _p("play_chest_sidel", ("chest", "left_move_10"), "play",
       ("happy",), "high", ("celebrate", "locomote"), "left"),
    _p("play_chest_sider", ("chest", "right_move_10"), "play",
       ("happy",), "high", ("celebrate", "locomote"), "right"),
    _p("play_wave_chest", ("wave", "chest"), "play",
       ("happy",), "high", ("celebrate", "greet"), "right"),
    _p("play_twist_chest", ("twist", "chest"), "play",
       ("happy",), "high", ("celebrate", "torso"), "none"),
    _p("play_step_chest", ("stepping", "chest"), "play",
       ("happy",), "high", ("celebrate",), "none"),
    _p("play_handup_chest", ("go_hand_up", "chest"), "play",
       ("happy",), "high", ("celebrate", "greet"), "both"),
    _p("play_chest_twist", ("chest", "twist"), "play",
       ("happy",), "high", ("celebrate", "torso"), "none"),
    _p("play_handup1_sidel", ("go_hand_up1", "left_move_10"), "play",
       ("happy",), "mid", ("celebrate", "locomote"), "left"),
    _p("play_lift_chest", ("lift_left_hand", "chest"), "play",
       ("happy",), "high", ("celebrate",), "left"),
    _p("play_wave_step_twist", ("wave", "stepping", "twist"), "play",
       ("happy",), "high", ("celebrate", "greet"), "right"),
    _p("play_chest_fwd", ("chest", "go_forward_one_small_step"), "play",
       ("happy",), "high", ("celebrate", "locomote"), "none"),
    _p("play_twist_step", ("twist", "stepping"), "play",
       ("happy", "surprised"), "mid", ("celebrate", "torso"), "none"),
    _p("play_righthand_chest", ("right_hand", "chest"), "play",
       ("happy",), "high", ("celebrate",), "right"),
    _p("play_lefthand_step", ("left_hand", "stepping"), "play",
       ("happy",), "mid", ("greet",), "left"),
    _p("play_sidel_chest", ("left_move_10", "chest"), "play",
       ("happy",), "high", ("celebrate", "locomote"), "left"),
    _p("play_turnr_wave", ("turn_right_small_step", "wave"), "play",
       ("happy",), "mid", ("greet", "celebrate"), "right"),
    # --- help (8) ---
    _p("help_handup_hold", ("go_hand_up", "stand_slow"), "help",
       ("happy", "neutral", "unhappy"), "mid", ("greet",), "both"),
    _p("help_handup1_wave", ("go_hand_up1", "wave"), "help",
       ("happy", "neutral"), "mid", ("greet",), "both"),
    _p("help_lift_step", ("lift_left_hand", "stepping"), "help",
       ("happy", "neutral"), "mid", ("greet",), "left"),
    _p("help_righthand_hold", ("right_hand", "stand_slow"), "help",
       ("happy", "neutral"), "mid", ("greet",), "right"),
    _p("help_wave_fwd", ("wave", "go_forward_one_small_step"), "help",
       ("happy", "neutral"), "mid", ("greet", "locomote"), "right"),
    _p("help_lefthand_hold", ("left_hand", "stand_slow"), "help",
       ("unhappy", "neutral"), "low", ("greet", "comfort"), "left"),
    _p("help_handup_turnl", ("go_hand_up", "turn_left_small_step"), "help",
       ("happy", "neutral"), "mid", ("greet", "locomote"), "both"),
    _p("help_step_handup1", ("stepping", "go_hand_up1"), "help",
       ("happy", "neutral"), "mid", ("greet",), "both"),
    # --- startle (10) ---
    _p("startle_twist_back", ("twist", "back_one_step"), "unknown",
       ("surprised",), "high", ("startle",), "none"),
    _p("startle_step_twist", ("stepping", "twist"), "unknown",
       ("surprised", "happy"), "mid", ("startle", "torso"), "none"),
    _p("startle_back_hold", ("back_one_step", "stand_slow"), "unknown",
       ("surprised",), "high", ("startle",), "none"),
    _p("startle_twist_sidel", ("twist", "left_move_10"), "unknown",
       ("surprised",), "mid", ("startle", "locomote"), "left"),
    _p("startle_handup_back", ("go_hand_up", "back_one_step"), "unknown",
       ("surprised",), "high", ("startle",), "both"),
    _p("startle_turnl_twist", ("turn_left_small_step", "twist"), "unknown",
       ("surprised",), "mid", ("startle",), "left"),
    _p("startle_turnr_step", ("turn_right_small_step", "stepping"), "unknown",
       ("surprised", "neutral"), "mid", ("startle",), "right"),
    _p("startle_twist_hold", ("twist", "stand_slow"), "unknown",
       ("surprised", "neutral"), "mid", ("startle", "idle"), "none"),
    _p("startle_back_twist", ("back_one_step", "twist"), "unknown",
       ("surprised",), "high", ("startle",), "none"),
    _p("startle_sider_twist", ("right_move_10", "twist"), "unknown",
       ("surprised",), "mid", ("startle", "locomote"), "right"),
    # --- idle (8) ---
    _p("idle_step_hold", ("stepping", "stand_slow"), "unknown",
       ("neutral",), "low", ("idle",), "none"),
    _p("idle_twist_hold", ("twist", "stand_slow"), "unknown",
       ("neutral",), "low", ("idle", "torso"), "none"),
    _p("idle_sidel_hold", ("left_move_10", "stand_slow"), "unknown",
       ("neutral",), "low", ("idle", "locomote"), "left"),
    _p("idle_sider_hold", ("right_move_10", "stand_slow"), "unknown",
       ("neutral",), "low", ("idle", "locomote"), "right"),
    _p("idle_turnl_hold", ("turn_left_small_step", "stand_slow"), "unknown",
       ("neutral",), "low", ("idle", "locomote"), "left"),
    _p("idle_turnr_hold", ("turn_right_small_step", "stand_slow"), "unknown",
       ("neutral",), "low", ("idle", "locomote"), "right"),
    _p("idle_squat_up", ("squat_down", "squat_up"), "unknown",
       ("neutral", "unhappy"), "low", ("idle", "torso"), "none"),
    _p("idle_step_twist", ("stepping", "twist"), "unknown",
       ("neutral", "happy"), "low", ("idle", "torso"), "none"),
    # --- locomotion / stop (6) ---
    _p("loco_forward_small", ("go_forward_one_small_step",), "play",
       ("happy", "neutral"), "mid", ("locomote",), "none"),
    _p("loco_forward_step", ("go_forward_one_step",), "play",
       ("happy", "neutral"), "mid", ("locomote",), "none"),
    _p("loco_back", ("back_one_step",), "play",
       ("happy", "neutral", "surprised"), "mid", ("locomote",), "none"),
    _p("loco_turn_l", ("turn_left_small_step",), "play",
       ("happy", "neutral"), "mid", ("locomote",), "left"),
    _p("loco_turn_r", ("turn_right_small_step",), "play",
       ("happy", "neutral"), "mid", ("locomote",), "right"),
    _p("stop_stand", ("stand",), "stop",
       ("neutral", "happy", "unhappy", "surprised"), "low", ("recover",), "none"),
)


LOCO_PHRASE_IDS = {
    "go_forward": ("loco_forward_small", "loco_forward_step"),
    "back_fast": ("loco_back",),
    "turn_left": ("loco_turn_l",),
    "turn_right": ("loco_turn_r",),
}

SOCIAL_INTENTS = (
    "greeting",
    "comfort_request",
    "play",
    "help",
    "unknown",
)


def phrases_by_id():
    return {phrase.id: phrase for phrase in PHRASES}


PHRASE_INDEX = phrases_by_id()


def validate_phrases() -> None:
    allow = set(ACTION_ALLOWLIST)
    blocked = set(BLOCKED_ACTIONS)
    seen = set()
    for phrase in PHRASES:
        if phrase.id in seen:
            raise ValueError("duplicate phrase id: %s" % phrase.id)
        seen.add(phrase.id)
        if not (1 <= len(phrase.clips) <= 3):
            raise ValueError("%s must have 1–3 clips" % phrase.id)
        for name in phrase.clips:
            if name in blocked:
                raise ValueError("%s uses blocked clip %s" % (phrase.id, name))
            if name not in allow:
                raise ValueError("%s uses unknown clip %s" % (phrase.id, name))
            if name not in CATALOG:
                raise ValueError("%s clip missing from catalog: %s" % (phrase.id, name))


validate_phrases()
