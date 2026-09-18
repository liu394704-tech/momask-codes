#!/usr/bin/env python3
"""Shared types for A/B arbiter pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Mode(str, Enum):
    A_ONLY = "A_only"
    B_ONLY = "B_only"
    A_PARALLEL_B = "A_parallel_B"


@dataclass
class Perception:
    session_id: str
    ts_ms: int
    vision_emotion: Optional[str] = None
    vision_conf: float = 0.0
    vision_intensity: Optional[str] = None
    face_found: bool = False
    face_actions: List[str] = field(default_factory=list)
    emotion_scores: Dict[str, float] = field(default_factory=dict)
    transcript: str = ""
    audio_emotion: Optional[str] = None
    audio_conf: float = 0.0
    ok: bool = True
    error: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Decision:
    session_id: str
    emotion: str
    intent: str
    confidence: float
    action_prompt: str
    action_group: List[str] = field(default_factory=list)
    motion_length_hint: int = 0
    fallback: bool = False
    reason: str = ""
    ok: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrackAResult:
    executed: bool
    action: Optional[str]
    intensity: Optional[str] = None
    simulated: bool = True
    detail: str = ""


@dataclass
class TrackBResult:
    ran: bool
    ok: bool
    joints_path: Optional[str] = None
    prompt: str = ""
    elapsed_s: float = 0.0
    load_s: float = 0.0
    gen_s: float = 0.0
    error: Optional[str] = None
    dry_run: bool = False


@dataclass
class RoundResult:
    mode: str
    effective_mode: str
    perception: Perception
    decision: Decision
    track_a: Optional[TrackAResult]
    track_b: Optional[TrackBResult]
    degraded: bool = False
    degrade_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "effective_mode": self.effective_mode,
            "degraded": self.degraded,
            "degrade_reason": self.degrade_reason,
            "perception": self.perception.to_dict(),
            "decision": self.decision.to_dict(),
            "track_a": asdict(self.track_a) if self.track_a else None,
            "track_b": asdict(self.track_b) if self.track_b else None,
        }
