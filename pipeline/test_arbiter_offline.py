#!/usr/bin/env python3
"""Offline tests for Arbiter mode switching (no camera, no API)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.arbiter import Arbiter, ArbiterConfig
from pipeline.schemas import Decision, Mode, Perception, TrackAResult, TrackBResult


def _dec(**kwargs):
    base = dict(
        session_id="t",
        emotion="happy",
        intent="greeting",
        confidence=0.8,
        action_prompt="a person waves happily",
        action_group=["wave"],
        fallback=False,
        ok=True,
    )
    base.update(kwargs)
    return Decision(**base)


def test_modes():
    perc = Perception(session_id="t", ts_ms=0, face_found=True, vision_emotion="happy", vision_conf=0.8)

    def run_a(d):
        return TrackAResult(executed=True, action=(d.action_group or ["wave"])[0], simulated=True)

    def run_b(d):
        return TrackBResult(ran=True, ok=True, prompt=d.action_prompt, dry_run=True)

    # A_only
    r = Arbiter(ArbiterConfig(mode=Mode.A_ONLY)).run_round(perc, _dec(), run_a, run_b)
    assert r.effective_mode == "A_only" and r.track_a and r.track_b is None

    # B_only
    r = Arbiter(ArbiterConfig(mode=Mode.B_ONLY)).run_round(perc, _dec(), run_a, run_b)
    assert r.effective_mode == "B_only" and r.track_b and r.track_a is None

    # A_parallel_B
    r = Arbiter(ArbiterConfig(mode=Mode.A_PARALLEL_B)).run_round(perc, _dec(), run_a, run_b)
    assert r.effective_mode == "A_parallel_B" and r.track_a and r.track_b

    # low conf degrade
    r = Arbiter(ArbiterConfig(mode=Mode.A_PARALLEL_B, confidence_min=0.45)).run_round(
        perc, _dec(confidence=0.2, action_prompt="x"), run_a, run_b
    )
    assert r.effective_mode == "A_only" and r.degraded and r.track_b is None

    # fallback degrade
    r = Arbiter(ArbiterConfig(mode=Mode.B_ONLY)).run_round(
        perc, _dec(fallback=True), run_a, run_b
    )
    assert r.effective_mode == "A_only" and r.degraded

    print("ARBITER_TESTS_PASS")


if __name__ == "__main__":
    test_modes()
