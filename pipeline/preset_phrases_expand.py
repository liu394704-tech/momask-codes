#!/usr/bin/env python3
"""Systematic extra phrases: mirrors, 3-clip chains, unused small-step variants.

Keeps hand-authored CORE_PHRASES intact and fills combinatorial gaps so the
library is large enough to survive long sessions without looking canned.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Set, Tuple

from .preset_catalog import CATALOG
from .preset_phrases import Phrase, _p


HANDS = (
    "wave",
    "left_hand",
    "right_hand",
    "lift_left_hand",
    "go_hand_up",
    "go_hand_up1",
)
STEPS = (
    "left_move",
    "right_move",
    "left_move_10",
    "right_move_10",
    "left_move_20",
    "right_move_20",
    "turn_left_small_step",
    "turn_right_small_step",
    "turn_left_small_step_a",
    "turn_right_small_step_a",
    "go_forward_one_small_step",
)
TURNS = (
    "turn_left_small_step",
    "turn_right_small_step",
    "turn_left_small_step_a",
    "turn_right_small_step_a",
)
SIDES = (
    "left_move",
    "right_move",
    "left_move_10",
    "right_move_10",
    "left_move_20",
    "right_move_20",
)
HOLDS = ("stand_slow",)
BODIES = ("twist", "stepping")
COMFORT_CORE = ("bow", "jugong", "squat")


def _laterality(clips: Sequence[str]) -> str:
    left = right = both = False
    for name in clips:
        clip = CATALOG.get(name)
        if clip is None:
            continue
        if clip.laterality == "left":
            left = True
        elif clip.laterality == "right":
            right = True
        elif clip.laterality == "both":
            both = True
    if left and right:
        return "both"
    if left:
        return "left"
    if right:
        return "right"
    if both:
        return "both"
    return "none"


def _energy(clips: Sequence[str], default: str = "mid") -> str:
    if any(name in ("chest", "back_one_step") for name in clips):
        return "high"
    if any(name in ("bow", "jugong", "squat", "squat_down", "stand_slow") for name in clips):
        return "low"
    return default


def _slug(clips: Sequence[str]) -> str:
    return "_".join(clips)


def _add(
    out: List[Phrase],
    seen: Set[Tuple[str, ...]],
    pid: str,
    clips: Sequence[str],
    intent: str,
    emotion_fit: Tuple[str, ...],
    tags: Tuple[str, ...],
    energy: str = "",
) -> None:
    names = tuple(clips)
    if len(names) < 2 or len(names) > 3:
        return
    if len(set(names)) != len(names):
        return
    if names in seen:
        return
    for name in names:
        if name not in CATALOG:
            return
    seen.add(names)
    out.append(_p(
        pid,
        names,
        intent,
        emotion_fit,
        energy or _energy(names),
        tags,
        _laterality(names),
    ))


def expand_phrases(existing: Iterable[Phrase]) -> Tuple[Phrase, ...]:
    seen: Set[Tuple[str, ...]] = {tuple(p.clips) for p in existing}
    extra: List[Phrase] = []

    # Greeting: every unused hand + body / step / hold pair.
    for hand in HANDS:
        for body in BODIES:
            _add(extra, seen, "greetX_%s" % _slug((hand, body)),
                 (hand, body), "greeting", ("happy", "neutral"), ("greet", "torso"))
            _add(extra, seen, "greetX_%s" % _slug((body, hand)),
                 (body, hand), "greeting", ("happy", "neutral"), ("greet",))
        for step in STEPS:
            _add(extra, seen, "greetX_%s" % _slug((hand, step)),
                 (hand, step), "greeting", ("happy", "neutral"), ("greet", "locomote"))
            _add(extra, seen, "greetX_%s" % _slug((step, hand)),
                 (step, hand), "greeting", ("happy", "neutral"), ("greet", "locomote"))
        for hold in HOLDS:
            _add(extra, seen, "greetX_%s" % _slug((hand, hold)),
                 (hand, hold), "greeting", ("happy", "neutral"), ("greet", "idle"), "low")

    # Greeting 3-clips: hand -> small step -> torso/hold.
    for hand in HANDS:
        for step in SIDES + TURNS:
            for tail in BODIES + HOLDS:
                _add(
                    extra, seen, "greet3_%s" % _slug((hand, step, tail)),
                    (hand, step, tail), "greeting", ("happy", "neutral"),
                    ("greet", "locomote"),
                )
        for body in BODIES:
            for hold in HOLDS:
                _add(
                    extra, seen, "greet3_%s" % _slug((hand, body, hold)),
                    (hand, body, hold), "greeting", ("happy", "neutral"),
                    ("greet", "idle"), "low",
                )

    # Play / celebrate: chest with unused partners and 3-clip bursts.
    for partner in HANDS + BODIES + SIDES + TURNS + ("go_forward_one_small_step",):
        _add(extra, seen, "playX_%s" % _slug(("chest", partner)),
             ("chest", partner), "play", ("happy",), ("celebrate",))
        _add(extra, seen, "playX_%s" % _slug((partner, "chest")),
             (partner, "chest"), "play", ("happy",), ("celebrate",))
    for hand in HANDS:
        for step in SIDES[:4]:
            _add(
                extra, seen, "play3_%s" % _slug((hand, "chest", step)),
                (hand, "chest", step), "play", ("happy",),
                ("celebrate", "greet", "locomote"), "high",
            )
            _add(
                extra, seen, "play3_%s" % _slug(("chest", hand, step)),
                ("chest", hand, step), "play", ("happy",),
                ("celebrate", "locomote"), "high",
            )
    for body in BODIES:
        _add(
            extra, seen, "play3_%s" % _slug(("chest", body, "stand_slow")),
            ("chest", body, "stand_slow"), "play", ("happy",),
            ("celebrate", "idle"),
        )

    # Comfort: bow / jugong / squat with unused hands, sides, holds, 3-clips.
    for core in COMFORT_CORE:
        for partner in HANDS + HOLDS + BODIES + SIDES + TURNS:
            _add(extra, seen, "comfortX_%s" % _slug((core, partner)),
                 (core, partner), "comfort_request",
                 ("unhappy", "neutral"), ("comfort",), "low")
            _add(extra, seen, "comfortX_%s" % _slug((partner, core)),
                 (partner, core), "comfort_request",
                 ("unhappy", "neutral"), ("comfort",), "low")
        _add(
            extra, seen, "comfort3_%s" % _slug((core, "squat_down", "squat_up")),
            (core, "squat_down", "squat_up"), "comfort_request",
            ("unhappy",), ("comfort", "torso"), "low",
        )
        for hand in ("left_hand", "right_hand", "wave"):
            _add(
                extra, seen, "comfort3_%s" % _slug((hand, core, "stand_slow")),
                (hand, core, "stand_slow"), "comfort_request",
                ("unhappy", "neutral"), ("comfort",), "low",
            )
            _add(
                extra, seen, "comfort3_%s" % _slug((core, hand, "stepping")),
                (core, hand, "stepping"), "comfort_request",
                ("unhappy", "neutral"), ("comfort",), "low",
            )

    # Help: two-hand + approach sequences (single-hand pairs already used by greet).
    help_hands = ("wave", "go_hand_up", "go_hand_up1", "lift_left_hand", "right_hand", "left_hand")
    for i, hand_a in enumerate(help_hands):
        for hand_b in help_hands[i + 1:]:
            _add(extra, seen, "helpX_%s" % _slug((hand_a, hand_b)),
                 (hand_a, hand_b), "help", ("happy", "neutral"), ("greet",))
            _add(
                extra, seen, "help3_%s" % _slug((hand_a, hand_b, "stepping")),
                (hand_a, hand_b, "stepping"), "help",
                ("happy", "neutral"), ("greet",),
            )
            _add(
                extra, seen, "help3_%s" % _slug((hand_a, hand_b, "stand_slow")),
                (hand_a, hand_b, "stand_slow"), "help",
                ("happy", "neutral", "unhappy"), ("greet", "idle"),
            )
            _add(
                extra, seen, "help3_%s" % _slug((hand_a, hand_b, "go_forward_one_small_step")),
                (hand_a, hand_b, "go_forward_one_small_step"), "help",
                ("happy", "neutral"), ("greet", "locomote"),
            )
    for hand in help_hands:
        for turn in TURNS:
            _add(
                extra, seen, "help3_%s" % _slug((hand, "wave", turn)) if hand != "wave"
                else "help3_%s" % _slug((hand, "go_hand_up", turn)),
                (hand, "wave" if hand != "wave" else "go_hand_up", turn),
                "help", ("happy", "neutral"), ("greet", "locomote"),
            )

    # Startle: twist / back / small dodge, plus 3-clip recoveries.
    for head in ("twist", "back_one_step", "stepping"):
        for step in SIDES + TURNS + ("back_one_step",):
            _add(extra, seen, "startleX_%s" % _slug((head, step)),
                 (head, step), "unknown", ("surprised",), ("startle", "locomote"))
    for step in SIDES + TURNS:
        _add(
            extra, seen, "startle3_%s" % _slug(("twist", step, "stand_slow")),
            ("twist", step, "stand_slow"), "unknown",
            ("surprised", "neutral"), ("startle", "idle"),
        )
        _add(
            extra, seen, "startle3_%s" % _slug((step, "twist", "stand_slow")),
            (step, "twist", "stand_slow"), "unknown",
            ("surprised",), ("startle",),
        )
    for hand in ("go_hand_up", "go_hand_up1", "left_hand"):
        _add(
            extra, seen, "startle3_%s" % _slug((hand, "twist", "back_one_step")),
            (hand, "twist", "back_one_step"), "unknown",
            ("surprised",), ("startle",), "high",
        )

    # Idle / acknowledge: small weight shifts that do not look like a greeting loop.
    for step in STEPS:
        _add(extra, seen, "idleX_%s" % _slug((step, "stand_slow")),
             (step, "stand_slow"), "unknown", ("neutral",), ("idle", "locomote"), "low")
        _add(extra, seen, "idleX_%s" % _slug((step, "twist")),
             (step, "twist"), "unknown", ("neutral", "happy"), ("idle", "torso"), "low")
        _add(extra, seen, "idleX_%s" % _slug(("stepping", step)),
             ("stepping", step), "unknown", ("neutral",), ("idle", "locomote"), "low")
    for left, right in (
        ("left_move_10", "right_move_10"),
        ("left_move_20", "right_move_20"),
        ("left_move", "right_move"),
        ("turn_left_small_step", "turn_right_small_step"),
        ("turn_left_small_step_a", "turn_right_small_step_a"),
    ):
        _add(extra, seen, "idle3_%s" % _slug((left, right, "stand_slow")),
             (left, right, "stand_slow"), "unknown",
             ("neutral",), ("idle", "locomote"), "low")
        _add(extra, seen, "idle3_%s" % _slug((right, left, "stand_slow")),
             (right, left, "stand_slow"), "unknown",
             ("neutral",), ("idle", "locomote"), "low")

    # Extra short locomotion variants for voice commands (still one clip).
    for pid, clips, laterality in (
        ("loco_turn_l_a", ("turn_left_small_step_a",), "left"),
        ("loco_turn_r_a", ("turn_right_small_step_a",), "right"),
        ("loco_sidel", ("left_move",), "left"),
        ("loco_sider", ("right_move",), "right"),
        ("loco_sidel_10", ("left_move_10",), "left"),
        ("loco_sider_10", ("right_move_10",), "right"),
        ("loco_sidel_20", ("left_move_20",), "left"),
        ("loco_sider_20", ("right_move_20",), "right"),
    ):
        names = clips
        if names not in seen:
            seen.add(names)
            extra.append(_p(
                pid, names, "play", ("happy", "neutral"), "mid",
                ("locomote",), laterality,
            ))

    return tuple(extra)
