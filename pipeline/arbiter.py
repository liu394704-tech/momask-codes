#!/usr/bin/env python3
"""Thin A/B arbiter: mode switch + per-round degrade to Track A."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from .schemas import Decision, Mode, RoundResult, TrackAResult, TrackBResult, Perception


@dataclass
class ArbiterConfig:
    mode: Mode = Mode.A_PARALLEL_B
    confidence_min: float = 0.45
    decide_timeout_s: float = 1.5
    momask_timeout_s: float = 4.0


class Arbiter:
    """Decides A_only / B_only / A_parallel_B and applies safety degrade."""

    def __init__(self, config: Optional[ArbiterConfig] = None):
        self.config = config or ArbiterConfig()

    def resolve_effective_mode(self, decision: Decision) -> Tuple[Mode, bool, Optional[str]]:
        """Return (effective_mode, degraded, reason)."""
        configured = self.config.mode

        if configured == Mode.A_ONLY:
            return Mode.A_ONLY, False, None
        if configured == Mode.B_ONLY:
            # Still allow safety degrade away from B.
            if self._must_skip_b(decision):
                return Mode.A_ONLY, True, self._skip_reason(decision)
            return Mode.B_ONLY, False, None

        # A_parallel_B
        if self._must_skip_b(decision):
            return Mode.A_ONLY, True, self._skip_reason(decision)
        return Mode.A_PARALLEL_B, False, None

    def _must_skip_b(self, decision: Decision) -> bool:
        if not decision.ok:
            return True
        if decision.fallback:
            return True
        if decision.confidence < self.config.confidence_min:
            return True
        if not (decision.action_prompt or "").strip():
            return True
        return False

    def _skip_reason(self, decision: Decision) -> str:
        if not decision.ok:
            return decision.error or "decide_failed"
        if decision.fallback:
            return "fallback_flag"
        if decision.confidence < self.config.confidence_min:
            return "low_confidence"
        if not (decision.action_prompt or "").strip():
            return "empty_action_prompt"
        return "skip_b"

    def run_round(
        self,
        perception: Perception,
        decision: Decision,
        run_a: Callable[[Decision], TrackAResult],
        run_b: Callable[[Decision], TrackBResult],
    ) -> RoundResult:
        effective, degraded, reason = self.resolve_effective_mode(decision)
        track_a = None
        track_b = None

        if effective in (Mode.A_ONLY, Mode.A_PARALLEL_B):
            track_a = run_a(decision)

        if effective in (Mode.B_ONLY, Mode.A_PARALLEL_B):
            track_b = run_b(decision)
            if track_b is not None and track_b.ran and not track_b.ok:
                degraded = True
                reason = reason or (track_b.error or "momask_failed")
                # A already ran in parallel mode; in B_only we cannot invent A here.

        return RoundResult(
            mode=self.config.mode.value,
            effective_mode=effective.value,
            perception=perception,
            decision=decision,
            track_a=track_a,
            track_b=track_b,
            degraded=degraded,
            degrade_reason=reason,
        )
