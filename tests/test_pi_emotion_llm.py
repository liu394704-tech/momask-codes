#!/usr/bin/env python3
"""Unit tests for the Pi emotion -> LLM -> ActionGroup loop (no robot, no MoMask)."""
from __future__ import annotations

import os
import sys
import unittest

from pipeline.actions import (
    ACTION_ALLOWLIST,
    BLOCKED_ACTIONS,
    first_allowed_action,
    is_stop_signal,
    merge_keyword_transcript,
    sanitize_action_group,
    WONDERECHO_CMDS,
)
from pipeline.decide_edge import edge_rule_decide
from pipeline.schemas import Decision, Perception
from pipeline.preset_select import PhraseSelector
from pipeline.track_a import resolve_action, run_track_a


def _perc(**kwargs) -> Perception:
    extras = dict(kwargs.pop("extras", {}) or {})
    return Perception(
        session_id="t1",
        ts_ms=0,
        vision_emotion=kwargs.get("emotion", "happy"),
        vision_conf=kwargs.get("conf", 0.7),
        vision_intensity="strong",
        face_found=kwargs.get("face_found", True),
        face_actions=list(kwargs.get("face_actions") or []),
        transcript=kwargs.get("transcript", ""),
        ok=True,
        extras=extras,
    )


class ActionAllowlistTests(unittest.TestCase):
    def test_strips_kicks(self):
        cleaned = sanitize_action_group(["wave", "left_kick", "wing_chun", "bow"])
        self.assertEqual(cleaned, ["wave", "bow"])

    def test_unknown_dropped(self):
        self.assertIsNone(first_allowed_action(["explode"]))
        self.assertIn("wave", ACTION_ALLOWLIST)
        self.assertIn("left_kick", BLOCKED_ACTIONS)

    def test_keyword_outranks_transcript(self):
        text = merge_keyword_transcript("sleep", "你好一起玩")
        self.assertTrue(text.startswith("停"))
        self.assertTrue(is_stop_signal(keyword="sleep", transcript="你好"))

    def test_wonderecho_payloads(self):
        self.assertEqual(WONDERECHO_CMDS[b"\xaa\x55\x03\x00\xfb"], "wakeup")
        self.assertEqual(WONDERECHO_CMDS[b"\xaa\x55\x00\x01\xfb"], "forward")
        self.assertEqual(WONDERECHO_CMDS[b"\xaa\x55\x00\x03\xfb"], "turn_left")


class RuleDecideTests(unittest.TestCase):
    def test_forward_keyword(self):
        d = edge_rule_decide(_perc(
            emotion="happy",
            extras={"keyword": "forward"},
        ))
        self.assertEqual(d.action_group[0], "go_forward")

    def test_stop_beats_face(self):
        d = edge_rule_decide(_perc(emotion="happy", transcript="停一下"))
        self.assertEqual(d.intent, "stop")
        self.assertEqual(d.action_group, ["stand"])

    def test_tired_bow(self):
        d = edge_rule_decide(_perc(emotion="neutral", transcript="我今天有点累"))
        self.assertEqual(d.intent, "comfort_request")
        self.assertEqual(d.action_group, ["bow"])
        self.assertTrue(d.action_prompt.lower().startswith("a person"))

    def test_hello_wave(self):
        d = edge_rule_decide(_perc(emotion="unhappy", transcript="你好呀"))
        self.assertEqual(d.intent, "greeting")
        self.assertEqual(d.action_group, ["wave"])

    def test_no_signal_fallback(self):
        d = edge_rule_decide(_perc(emotion="neutral", face_found=False, conf=0.1))
        self.assertTrue(d.fallback)
        self.assertEqual(d.action_group, ["stand"])


class TrackAResolveTests(unittest.TestCase):
    def test_llm_group_wins(self):
        d = Decision(
            session_id="t", emotion="happy", intent="play",
            confidence=0.8, action_prompt="a person waves",
            action_group=["go_forward"],
        )
        selector = PhraseSelector()
        action, source = resolve_action(d, selector=selector)
        self.assertIn(action, ("go_forward_one_small_step", "go_forward_one_step"))
        self.assertEqual(source, "locomotion_phrase")

    def test_blocked_group_ignored(self):
        d = Decision(
            session_id="t", emotion="happy", intent="play",
            confidence=0.8, action_prompt="a person kicks",
            action_group=["left_kick"],
        )
        selector = PhraseSelector()
        action, source = resolve_action(d, selector=selector)
        self.assertNotEqual(action, "left_kick")
        self.assertNotIn("left_kick", (action or "").split("+"))
        self.assertIn(source, ("phrase_select", "phrase_fallback", "phrase_stand"))

    def test_stop_stand(self):
        d = Decision(
            session_id="t", emotion="happy", intent="stop",
            confidence=0.9, action_prompt="a person stands",
            action_group=["chest"],
        )
        selector = PhraseSelector()
        action, source = resolve_action(d, selector=selector)
        self.assertEqual(action, "stand")
        self.assertEqual(source, "stop_signal")

    def test_simulate_runs_without_robot(self):
        d = Decision(
            session_id="t", emotion="happy", intent="greeting",
            confidence=0.8, action_prompt="a person waves",
            action_group=["wave"],
        )
        perc = _perc(emotion="happy", transcript="你好呀", face_actions=["smile"])
        result = run_track_a(
            d, simulate=True, execute_robot=False,
            perception=perc, selector=PhraseSelector(),
        )
        self.assertTrue(result.executed)
        self.assertTrue(result.simulated)
        self.assertTrue(result.clips)
        for clip in result.clips:
            self.assertIn(clip, ACTION_ALLOWLIST)
        self.assertIn("phrase_select", result.detail)


class MomaskSwitchTests(unittest.TestCase):
    def test_env_flag(self):
        from pipeline.run_pi_emotion_llm import env_flag

        old = os.environ.get("ENABLE_MOMASK")
        try:
            os.environ["ENABLE_MOMASK"] = "1"
            self.assertTrue(env_flag("ENABLE_MOMASK", False))
            os.environ["ENABLE_MOMASK"] = "off"
            self.assertFalse(env_flag("ENABLE_MOMASK", True))
            del os.environ["ENABLE_MOMASK"]
            self.assertFalse(env_flag("ENABLE_MOMASK", False))
        finally:
            if old is None:
                os.environ.pop("ENABLE_MOMASK", None)
            else:
                os.environ["ENABLE_MOMASK"] = old

    def test_cli_off_skips_track_b(self):
        payload = _run_cli(["--no-momask"])
        self.assertFalse(payload["momask"])
        self.assertIsNone(payload.get("track_b"))
        self.assertEqual(payload["extra"]["effective_mode"], "A_only")

    def test_cli_on_dry_run_writes_marker(self):
        payload = _run_cli(["--momask", "--momask-dry-run"])
        self.assertTrue(payload["momask"])
        self.assertIsNotNone(payload.get("track_b"))
        self.assertTrue(payload["track_b"]["dry_run"])
        self.assertTrue(payload["track_b"]["ok"])
        self.assertTrue(str(payload["track_b"]["joints_path"]).endswith(".txt"))
        self.assertEqual(payload["extra"]["effective_mode"], "A_parallel_B")


def _run_cli(extra):
    import json
    import subprocess
    import tempfile
    from pathlib import Path

    out = tempfile.mkdtemp(prefix="momask_switch_")
    cmd = [
        sys.executable, "-m", "pipeline.run_pi_emotion_llm",
        "--once", "--mock-perception", "--simulate",
        "--decide-backend", "mock",
        "--transcript", "你好呀",
        "--out-dir", out,
    ] + extra
    env = os.environ.copy()
    env.pop("ENABLE_MOMASK", None)
    subprocess.check_call(cmd, cwd=str(Path(__file__).resolve().parents[1]), env=env)
    files = list(Path(out).glob("*_emotion_llm.json"))
    assert files, "missing round json"
    return json.loads(files[0].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
