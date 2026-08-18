#!/usr/bin/env python3
"""Track A: preset ActionGroup via EmotionActionScheduler (simulate on Mac)."""
from __future__ import annotations

import os
import sys
import time
from typing import Optional

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


def run_track_a(
    decision: Decision,
    simulate: bool = True,
    execute_robot: bool = False,
) -> TrackAResult:
    """Plan (and optionally execute) a preset action from decision.

    On Mac, default is simulate=True (no hiwonder). Robot execution is opt-in.
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

    emotion = decision.emotion
    # Map fine labels back to scheduler's 4-class space.
    if emotion in ("sad", "angry"):
        emotion = "unhappy"
    if emotion not in ("neutral", "happy", "unhappy", "surprised"):
        emotion = "neutral"

    intensity = intensity_from_confidence(decision.confidence)
    scheduler = EmotionActionScheduler()
    # Prefer decision.action_group[0] if present and allowed by pools.
    preferred = (decision.action_group or [None])[0]
    pools = scheduler.action_map.get(emotion) or {"mild": [], "strong": []}
    pool = list(pools.get(intensity) or []) or list(pools.get("mild") or [])
    action = None
    if preferred and preferred in (pools.get("mild") or []) + (pools.get("strong") or []):
        action = preferred
    elif pool:
        action = scheduler.pick_action(emotion, intensity=intensity)
    elif preferred:
        action = preferred

    if emotion == "neutral" or not action:
        return TrackAResult(
            executed=False,
            action=None,
            intensity=intensity,
            simulated=simulate,
            detail="no_preset_action",
        )

    if execute_robot and not simulate:
        try:
            import hiwonder.ActionGroupControl as AGC  # type: ignore

            AGC.runActionGroup(action)
            AGC.runActionGroup("stand")
            return TrackAResult(
                executed=True,
                action=action,
                intensity=intensity,
                simulated=False,
                detail="robot_executed",
            )
        except Exception as exc:  # noqa: BLE001
            return TrackAResult(
                executed=False,
                action=action,
                intensity=intensity,
                simulated=True,
                detail="robot_exec_failed: %s" % exc,
            )

    # Mac simulate: advance scheduler marks for logging realism.
    now = time.time()
    scheduler.queue_action(action, emotion=emotion, intensity=intensity,
                           confidence=decision.confidence, now=now)
    scheduler.mark_action_started()
    scheduler.mark_action_finished(now + 0.05)
    scheduler.mark_recovery_finished(now + 0.1)
    return TrackAResult(
        executed=True,
        action=action,
        intensity=intensity,
        simulated=True,
        detail="mac_simulated_action_then_stand",
    )
