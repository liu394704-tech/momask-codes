#!/usr/bin/env python3
"""Tagged atomic ActionGroup clips for the social phrase fallback.

Only safe, non-aggressive groups. Kick / punch / wing_chun never appear.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .actions import ACTION_ALLOWLIST, BLOCKED_ACTIONS


@dataclass(frozen=True)
class Clip:
    name: str
    tags: Tuple[str, ...]
    laterality: str
    energy: str
    duration_hint_s: float = 1.5
    recover: bool = False


def _c(
    name: str,
    tags: Tuple[str, ...],
    laterality: str,
    energy: str,
    duration_hint_s: float = 1.5,
    recover: bool = False,
) -> Clip:
    return Clip(name, tags, laterality, energy, duration_hint_s, recover)


CLIPS: Tuple[Clip, ...] = (
    _c("stand", ("recover", "idle"), "none", "low", 0.8, True),
    _c("stand_slow", ("recover", "idle"), "none", "low", 1.4, True),
    _c("wave", ("greet", "hand_right"), "right", "mid", 1.8),
    _c("bow", ("comfort",), "none", "low", 1.6),
    _c("jugong", ("comfort",), "none", "low", 1.8),
    _c("squat", ("comfort", "torso"), "none", "low", 1.6),
    _c("squat_down", ("comfort", "torso"), "none", "low", 1.2),
    _c("squat_up", ("comfort", "torso", "recover"), "none", "low", 1.2),
    _c("chest", ("celebrate",), "both", "high", 1.8),
    _c("twist", ("torso", "startle"), "none", "mid", 1.5),
    _c("stepping", ("idle", "greet"), "none", "mid", 1.6),
    _c("left_hand", ("greet", "hand_left"), "left", "mid", 1.4),
    _c("right_hand", ("greet", "hand_right"), "right", "mid", 1.4),
    _c("lift_left_hand", ("greet", "hand_left"), "left", "mid", 1.5),
    _c("go_hand_up", ("greet", "hand_right"), "both", "mid", 1.5),
    _c("go_hand_up1", ("greet", "hand_right"), "both", "mid", 1.5),
    _c("back_one_step", ("startle", "locomote"), "none", "high", 1.2),
    _c("go_forward_one_small_step", ("locomote", "greet"), "none", "mid", 1.1),
    _c("go_forward_one_step", ("locomote",), "none", "mid", 1.3),
    _c("turn_left_small_step", ("locomote",), "left", "mid", 1.1),
    _c("turn_right_small_step", ("locomote",), "right", "mid", 1.1),
    _c("turn_left_small_step_a", ("locomote",), "left", "mid", 1.0),
    _c("turn_right_small_step_a", ("locomote",), "right", "mid", 1.0),
    _c("left_move", ("locomote",), "left", "mid", 1.1),
    _c("right_move", ("locomote",), "right", "mid", 1.1),
    _c("left_move_10", ("locomote",), "left", "mid", 1.0),
    _c("right_move_10", ("locomote",), "right", "mid", 1.0),
    _c("left_move_20", ("locomote",), "left", "mid", 1.2),
    _c("right_move_20", ("locomote",), "right", "mid", 1.2),
    # Kept in the catalog so keyword hints stay valid; phrases prefer small steps.
    _c("back_fast", ("startle", "locomote"), "none", "high", 1.4),
    _c("go_forward", ("locomote",), "none", "mid", 1.6),
    _c("turn_left", ("locomote",), "left", "mid", 1.4),
    _c("turn_right", ("locomote",), "right", "mid", 1.4),
)


def catalog_by_name() -> Dict[str, Clip]:
    return {clip.name: clip for clip in CLIPS}


CATALOG: Dict[str, Clip] = catalog_by_name()


def validate_catalog() -> None:
    allow = set(ACTION_ALLOWLIST)
    blocked = set(BLOCKED_ACTIONS)
    for clip in CLIPS:
        if clip.name in blocked:
            raise ValueError("blocked clip in catalog: %s" % clip.name)
        if clip.name not in allow:
            raise ValueError("catalog clip not on allow-list: %s" % clip.name)


validate_catalog()
