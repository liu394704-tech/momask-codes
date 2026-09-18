#!/usr/bin/env python3
"""Unified Decide entry: edge (default) | edge_llm | cloud | mock."""
from __future__ import annotations

import os
from typing import Optional

from .decide_cloud import cloud_decide, mock_decide as _legacy_mock
from .decide_edge import edge_decide, edge_llm_decide, edge_rule_decide
from .schemas import Decision, Perception


def default_backend() -> str:
    """Default: edge_auto = Qwen GGUF if present, else rule fallback.

    Pi deployment must NOT use OpenAI. Set DECIDE_BACKEND=cloud only on Mac联调.
    """
    env = os.environ.get("DECIDE_BACKEND", "").strip().lower()
    if env:
        return env
    return "edge_auto"


def run_decide(
    perception: Perception,
    backend: Optional[str] = None,
) -> Decision:
    b = (backend or default_backend()).strip().lower()
    if b in ("edge", "rule", "local"):
        return edge_rule_decide(perception)
    if b in ("edge_llm", "llm", "gguf"):
        return edge_llm_decide(perception)
    if b in ("edge_auto", "auto"):
        return edge_decide(perception)
    if b in ("cloud", "openai"):
        return cloud_decide(perception)
    if b in ("mock",):
        # Prefer new edge rules (same contract as historical mock_decide).
        return edge_rule_decide(perception)
    # unknown -> safe local
    d = edge_rule_decide(perception)
    d.reason = "unknown_backend:%s->edge_rule" % b
    return d


# re-exports for convenience
__all__ = [
    "run_decide",
    "default_backend",
    "edge_rule_decide",
    "edge_llm_decide",
    "edge_decide",
    "cloud_decide",
    "_legacy_mock",
]
