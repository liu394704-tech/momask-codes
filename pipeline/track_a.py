#!/usr/bin/env python3
"""Track A: preset ActionGroup via EmotionActionScheduler.

Mac default is simulate=True. On the Pi, execute_robot=True runs allow-listed
ActionGroups from the LLM / rule decision, then recovers to stand.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Optional, Tuple

from .actions import (
    first_allowed_action,
    is_stop_signal,
    locomotion_times,
    LOCOMOTION_ACTIONS,
)
from .schemas import Decision, TrackAResult

_FUNCTIONS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "源码",
    "TonyPi",
    "Functions",
)


def _import_scheduler():
    if _FUNCTIONS not in sys.path:
        sys.path.insert(0, _FUNCTIONS)
    from EmotionActionScheduler import (  # type: ignore
        EmotionActionScheduler,
        intensity_from_confidence,
    )
    return EmotionActionScheduler, intensity_from_confidence


def _normalize_emotion(emotion: Optional[str]) -> str:
    emo = (emotion or "neutral").strip().lower()
    if emo in ("sad", "angry"):
        return "unhappy"
    if emo not in ("neutral", "happy", "unhappy", "surprised"):
        return "neutral"
    return emo


def resolve_action(
    decision: Decision,
    scheduler=None,
    intensity: str = "mild",
    keyword: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """Pick an allow-listed ActionGroup. LLM/rule group wins over emotion pools."""
    if is_stop_signal(keyword=keyword, intent=decision.intent):
        return "stand", "stop_signal"

    preferred = first_allowed_action(decision.action_group)
    if preferred:
        return preferred, "decision_action_group"

    emotion = _normalize_emotion(decision.emotion)
    if scheduler is None:
        return None, "no_preset_action"
    action = scheduler.pick_action(emotion, intensity=intensity)
    if action:
        return action, "emotion_pool"
    return None, "no_preset_action"


def _execute_robot(action: str) -> Tuple[bool, str]:
    try:
        import hiwonder.ActionGroupControl as AGC  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return False, "robot_import_failed: %s" % exc

    try:
        if action in LOCOMOTION_ACTIONS:
            AGC.runActionGroup(action, locomotion_times(action), True)
        else:
            AGC.runActionGroup(action)
            if action not in ("stand", "stand_slow"):
                AGC.runActionGroup("stand")
        return True, "robot_executed"
    except Exception as exc:  # noqa: BLE001
        return False, "robot_exec_failed: %s" % exc


def run_track_a(
    decision: Decision,
    simulate: bool = True,
    execute_robot: bool = False,
    keyword: Optional[str] = None,
    scheduler=None,
) -> TrackAResult:
    """Plan (and optionally execute) a preset action from decision.

    On Mac, default is simulate=True (no hiwonder). Robot execution is opt-in.
    `decision.action_group[0]` is used when it is on the safety allow-list.
    """
    try:
        EmotionActionScheduler, intensity_from_confidence = _import_scheduler()
    except Exception as exc:  # noqa: BLE001
        return TrackAResult(
            executed=False,
            action=None,
            simulated=True,
            detail="scheduler_import_failed: %s" % exc,
        )

    intensity = intensity_from_confidence(decision.confidence)
    own_scheduler = scheduler is None
    if scheduler is None:
        scheduler = EmotionActionScheduler()

    action, source = resolve_action(
        decision, scheduler=scheduler, intensity=intensity, keyword=keyword
    )
    emotion = _normalize_emotion(decision.emotion)

    if not action:
        return TrackAResult(
            executed=False,
            action=None,
            intensity=intensity,
            simulated=simulate,
            detail=source,
        )

    if execute_robot and not simulate:
        now = time.time()
        stop = is_stop_signal(keyword=keyword, intent=decision.intent)
        if scheduler is not None:
            if stop:
                scheduler.reset()
            elif not scheduler.can_schedule(now):
                return TrackAResult(
                    executed=False,
                    action=action,
                    intensity=intensity,
                    simulated=True,
                    detail="%s|cooldown_busy" % source,
                )
            scheduler.queue_action(
                action, emotion=emotion, intensity=intensity,
                confidence=decision.confidence, now=time.time(),
            )
            scheduler.mark_action_started()
        ok, detail = _execute_robot(action)
        if scheduler is not None:
            fin = time.time()
            scheduler.mark_action_finished(fin)
            scheduler.mark_recovery_finished(fin)
        return TrackAResult(
            executed=ok,
            action=action,
            intensity=intensity,
            simulated=not ok,
            detail="%s|%s" % (source, detail),
        )

    now = time.time()
    queued = scheduler.queue_action(
        action, emotion=emotion, intensity=intensity,
        confidence=decision.confidence, now=now,
    )
    if queued is None and not own_scheduler:
        return TrackAResult(
            executed=False,
            action=action,
            intensity=intensity,
            simulated=True,
            detail="%s|cooldown_busy" % source,
        )
    scheduler.mark_action_started()
    scheduler.mark_action_finished(now + 0.05)
    scheduler.mark_recovery_finished(now + 0.1)
    return TrackAResult(
        executed=True,
        action=action,
        intensity=intensity,
        simulated=True,
        detail="%s|mac_simulated_action_then_stand" % source,
    )
