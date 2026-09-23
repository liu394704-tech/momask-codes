#!/usr/bin/env python3
"""Preset phrase catalog, multimodal pick, and anti-repeat diversity."""
from __future__ import annotations

import random
import unittest
from collections import defaultdict

from pipeline.actions import ACTION_ALLOWLIST, BLOCKED_ACTIONS
from pipeline.preset_catalog import CATALOG, CLIPS
from pipeline.preset_phrases import PHRASES
from pipeline.preset_select import PhraseSelector, clip_jaccard
from pipeline.schemas import Decision, Perception


def _perc(**kwargs) -> Perception:
    extras = dict(kwargs.pop("extras", {}) or {})
    return Perception(
        session_id="div",
        ts_ms=0,
        vision_emotion=kwargs.get("emotion", "happy"),
        vision_conf=kwargs.get("conf", 0.72),
        vision_intensity=kwargs.get("intensity", "strong"),
        face_found=kwargs.get("face_found", True),
        face_actions=list(kwargs.get("face_actions") or ["smile"]),
        transcript=kwargs.get("transcript", "你好"),
        ok=True,
        extras=extras,
    )


def _dec(**kwargs) -> Decision:
    return Decision(
        session_id="div",
        emotion=kwargs.get("emotion", "happy"),
        intent=kwargs.get("intent", "greeting"),
        confidence=kwargs.get("conf", 0.72),
        action_prompt=kwargs.get("prompt", "a person waves"),
        action_group=list(kwargs.get("action_group") or ["wave"]),
        extras={"phrase_hint": kwargs.get("phrase_hint", "greeting")},
    )


class VisionPathTests(unittest.TestCase):
    def test_pi_functions_path_includes_stock_tonypi(self):
        from pipeline.percept_pi import _HIWONDER_CANDIDATES

        self.assertIn("/home/pi/TonyPi/Functions", _HIWONDER_CANDIDATES)
        self.assertIn("/home/cat/TonyPi/Functions", _HIWONDER_CANDIDATES)


class CatalogTests(unittest.TestCase):
    def test_catalog_size_and_safety(self):
        self.assertGreaterEqual(len(CLIPS), 20)
        for clip in CLIPS:
            self.assertIn(clip.name, ACTION_ALLOWLIST)
            self.assertNotIn(clip.name, BLOCKED_ACTIONS)
            self.assertIn(clip.laterality, ("left", "right", "both", "none"))
            self.assertIn(clip.energy, ("low", "mid", "high"))

    def test_phrase_library_coverage(self):
        self.assertGreaterEqual(len(PHRASES), 200)
        clip_sets = [tuple(p.clips) for p in PHRASES]
        self.assertEqual(len(clip_sets), len(set(clip_sets)))
        intents = {p.intent for p in PHRASES}
        for need in ("greeting", "comfort_request", "play", "help", "unknown", "stop"):
            self.assertIn(need, intents)
        emotions = set()
        for phrase in PHRASES:
            emotions.update(phrase.emotion_fit)
            self.assertTrue(1 <= len(phrase.clips) <= 3)
            for name in phrase.clips:
                self.assertIn(name, CATALOG)
                self.assertIn(name, ACTION_ALLOWLIST)
                self.assertNotIn(name, BLOCKED_ACTIONS)
        for need in ("happy", "unhappy", "surprised", "neutral"):
            self.assertIn(need, emotions)


class DiversityTests(unittest.TestCase):
    def test_twenty_happy_hello_rounds(self):
        selector = PhraseSelector()
        perc = _perc(emotion="happy", transcript="你好", face_actions=["smile"])
        dec = _dec(emotion="happy", intent="greeting", action_group=["wave"])
        rng = random.Random(0)
        choices = []
        for _ in range(20):
            choice = selector.select(perc, dec, intensity="strong", rng=rng)
            self.assertTrue(choice.clips)
            for clip in choice.clips:
                self.assertIn(clip, ACTION_ALLOWLIST)
                self.assertNotIn(clip, BLOCKED_ACTIONS)
            selector.commit(choice)
            choices.append(choice)

        for index in range(1, 20):
            jac = clip_jaccard(choices[index - 1].clips, choices[index].clips)
            self.assertLess(
                jac, 0.5,
                "adjacent Jaccard %.3f for %s -> %s" % (
                    jac, choices[index - 1].phrase_id, choices[index].phrase_id,
                ),
            )
            self.assertNotEqual(choices[index].clips[0], choices[index - 1].clips[-1])

        phrase_idx = defaultdict(list)
        for index, choice in enumerate(choices):
            phrase_idx[choice.phrase_id].append(index)
        for phrase_id, hits in phrase_idx.items():
            for prev, nxt in zip(hits, hits[1:]):
                self.assertGreaterEqual(
                    nxt - prev, 10,
                    "phrase %s reused too soon: %s" % (phrase_id, hits),
                )
        for start in range(0, 11):
            window = [c.phrase_id for c in choices[start:start + 10]]
            self.assertEqual(
                len(window), len(set(window)),
                "duplicate inside 10-window %s: %s" % (start, window),
            )
        actions = [c.action for c in choices]
        for start in range(0, 11):
            window = actions[start:start + 10]
            self.assertEqual(len(window), len(set(window)), window)

        clip_idx = defaultdict(list)
        for index, choice in enumerate(choices):
            for clip in set(choice.clips):
                clip_idx[clip].append(index)
        for clip, hits in clip_idx.items():
            for prev, nxt in zip(hits, hits[1:]):
                self.assertGreaterEqual(
                    nxt - prev, 3,
                    "clip %s reused too soon: %s" % (clip, hits),
                )

    def test_stop_clears_history(self):
        selector = PhraseSelector()
        first = selector.select(_perc(), _dec(), rng=random.Random(1))
        selector.commit(first)
        self.assertTrue(selector.phrase_hist)
        stop = selector.select(
            _perc(transcript="停一下"),
            _dec(intent="stop", action_group=["chest"]),
            rng=random.Random(2),
        )
        self.assertEqual(stop.source, "stop_signal")
        self.assertEqual(stop.clips, ["stand"])
        self.assertFalse(selector.phrase_hist)

    def test_forward_uses_small_step(self):
        selector = PhraseSelector()
        choice = selector.select(
            _perc(transcript="前进", emotion="happy", face_actions=[]),
            _dec(intent="play", action_group=["go_forward"], phrase_hint="play"),
            rng=random.Random(3),
        )
        self.assertEqual(choice.source, "locomotion_phrase")
        self.assertTrue(set(choice.clips) <= {
            "go_forward_one_small_step", "go_forward_one_step",
        })


if __name__ == "__main__":
    unittest.main()
