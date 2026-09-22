#!/usr/bin/env python3
"""Track A: multimodal preset phrases via PhraseSelector.

Mac default is simulate=True. On the Pi, execute_robot=True plays the chosen
clip sequence (random pause + stand/stand_slow recovery). Kick / punch groups
never run.
"""
from __future__ import annotations

import os
import random
import sys
import time
from typing import Optional, Tuple

from .actions import (
    LOCOMOTION_ACTIONS,
    is_stop_signal,
    locomotion_times,
    normalize_emotion,
    sanitize_action_group,
)
from .preset_select import PhraseChoice, PhraseSelector, get_default_selector
from .schemas import Decision, Perception, TrackAResult

_FUNCTIONS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "源码",
    "TonyPi",
    "Functions",
)


def _import_scheduler():
    if _FUNCTIONS not in sys.path:
        sys.path.insert(0, _FUNCTIONS)
    try:
        from EmotionActionScheduler import (  # type: ignore
            EmotionActionScheduler,
            intensity_from_confidence,
        )
        return EmotionActionScheduler, intensity_from_confidence
    except Exception:
        from .emotion_scheduler import (  # type: ignore
            EmotionActionScheduler,
            intensity_from_confidence,
        )
        return EmotionActionScheduler, intensity_from_confidence


def resolve_phrase(
    decision: Decision,
    scheduler=None,
    intensity: str = "mild",
    keyword: Optional[str] = None,
    perception: Optional[Perception] = None,
    selector: Optional[PhraseSelector] = None,
    rng: Optional[random.Random] = None,
) -> PhraseChoice:
    """Pick a diversity-aware phrase. Stop and locomotion hard-rules win."""
    picker = selector or get_default_selector()
    return picker.select(
        perception,
        decision,
        keyword=keyword,
        intensity=intensity,
        rng=rng,
    )


def resolve_action(
    decision: Decision,
    scheduler=None,
    intensity: str = "mild",
    keyword: Optional[str] = None,
    perception: Optional[Perception] = None,
    selector: Optional[PhraseSelector] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[Optional[str], str]:
    """Compatibility wrapper: joined clip names + source."""
    choice = resolve_phrase(
        decision,
        scheduler=scheduler,
        intensity=intensity,
        keyword=keyword,
        perception=perception,
        selector=selector,
        rng=rng,
    )
    return choice.action or None, choice.source


def _execute_robot_clips(
    clips: list,
    recovery: Optional[str],
    pause_s: float,
) -> Tuple[bool, str]:
    try:
        import hiwonder.ActionGroupControl as AGC  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return False, "robot_import_failed: %s" % exc

    try:
        for index, clip in enumerate(clips):
            if clip in LOCOMOTION_ACTIONS:
                AGC.runActionGroup(clip, locomotion_times(clip), True)
            else:
                AGC.runActionGroup(clip)
            if index + 1 < len(clips) and pause_s > 0:
                time.sleep(pause_s)
        if recovery and recovery not in clips[-1:]:
            AGC.runActionGroup(recovery)
        return True, "robot_executed"
    except Exception as exc:  # noqa: BLE001
        return False, "robot_exec_failed: %s" % exc


def _result(
    executed: bool,
    choice: Optional[PhraseChoice],
    intensity: str,
    simulated: bool,
    detail: str,
) -> TrackAResult:
    return TrackAResult(
        executed=executed,
        action=choice.action if choice else None,
        intensity=intensity,
        simulated=simulated,
        detail=detail,
        phrase_id=choice.phrase_id if choice else None,
        clips=list(choice.clips) if choice else [],
        recovery=choice.recovery if choice else None,
        bans=list(choice.bans) if choice else [],
    )


def run_track_a(
    decision: Decision,
    simulate: bool = True,
    execute_robot: bool = False,
    keyword: Optional[str] = None,
    scheduler=None,
    perception: Optional[Perception] = None,
    selector: Optional[PhraseSelector] = None,
    rng: Optional[random.Random] = None,
) -> TrackAResult:
    """Plan (and optionally execute) a preset phrase from decision + perception.

    On Mac, default is simulate=True (no hiwonder). Robot execution is opt-in.
    Phrase selection overrides a single `action_group` name except stop / locomotion.
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

    picker = selector or get_default_selector()
    rng = rng or random.Random()
    choice = resolve_phrase(
        decision,
        scheduler=scheduler,
        intensity=intensity,
        keyword=keyword,
        perception=perception,
        selector=picker,
        rng=rng,
    )
    emotion = normalize_emotion(decision.emotion)
    clips = sanitize_action_group(choice.clips)
    if not clips:
        return _result(False, choice, intensity, simulate, choice.source or "no_preset_action")

    choice.clips = clips
    action = choice.action
    source = choice.source

    if execute_robot and not simulate:
        now = time.time()
        stop = is_stop_signal(keyword=keyword, intent=decision.intent)
        if scheduler is not None:
            if stop:
                scheduler.reset()
            elif not scheduler.can_schedule(now):
                return _result(False, choice, intensity, True, "%s|cooldown_busy" % source)
            scheduler.action_cooldown = 6.0 + 4.0 * rng.random()
            scheduler.queue_action(
                action, emotion=emotion, intensity=intensity,
                confidence=decision.confidence, now=time.time(),
            )
            scheduler.mark_action_started()
        pause = 0.08 + 0.17 * rng.random()
        ok, detail = _execute_robot_clips(clips, choice.recovery, pause)
        if scheduler is not None:
            fin = time.time()
            scheduler.mark_action_finished(fin)
            scheduler.mark_recovery_finished(fin)
        if ok:
            picker.commit(choice)
        return _result(ok, choice, intensity, not ok, "%s|%s" % (source, detail))

    now = time.time()
    queued = scheduler.queue_action(
        action, emotion=emotion, intensity=intensity,
        confidence=decision.confidence, now=now,
    )
    if queued is None and not own_scheduler:
        return _result(False, choice, intensity, True, "%s|cooldown_busy" % source)
    scheduler.action_cooldown = 6.0 + 4.0 * rng.random()
    scheduler.mark_action_started()
    scheduler.mark_action_finished(now + 0.05)
    scheduler.mark_recovery_finished(now + 0.1)
    picker.commit(choice)
    return _result(
        True,
        choice,
        intensity,
        True,
        "%s|mac_simulated_phrase" % source,
    )
