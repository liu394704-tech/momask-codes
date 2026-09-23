#!/usr/bin/env python3
"""WonderPi face-game contract: init/start/stop/exit/run without a robot."""
from __future__ import annotations

import unittest

import numpy as np

from pipeline import wonderpi_face_game as game


class WonderPiContractTests(unittest.TestCase):
    def setUp(self):
        game._ENTER_ON_START = False

    def tearDown(self):
        game.stop()
        game._RUNNING = False
        game._BUSY = False
        game._PENDING_KEYWORD = None
        game._FACE_HELD = 0
        game._ENTER_ON_START = True

    def test_exports_match_facedetect(self):
        for name in ("init", "start", "stop", "exit", "run"):
            self.assertTrue(callable(getattr(game, name)))

    def test_run_returns_frame_when_stopped(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        game.stop()
        out = game.run(img)
        self.assertIs(out, img)

    def test_neutral_vote_does_not_lock_next_emotion(self):
        game._SCHEDULER = None
        scheduler = game._scheduler()
        for _ in range(8):
            scheduler.observe("neutral", 0.9)
        stable, conf = game.stable_launch_emotion()
        self.assertIsNone(stable)
        self.assertEqual(conf, 0.0)
        self.assertIsNone(scheduler.voter._last_label)
        for _ in range(8):
            scheduler.observe("happy", 0.9)
        stable, conf = game.stable_launch_emotion()
        self.assertEqual(stable, "happy")
        self.assertGreater(conf, 0.5)

    def test_held_face_moves_even_when_neutral(self):
        self.assertIsNone(game.face_should_move(None, 3))
        self.assertEqual(game.face_should_move(None, 8), "neutral")
        self.assertEqual(game.face_should_move("happy", 0), "happy")

    def test_play_and_log_writes_latency_without_moving(self):
        import os
        import tempfile
        from pathlib import Path

        game._SCHEDULER = None
        game._SELECTOR = None
        game._BUSY = False
        handle = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        handle.close()
        previous_backend = os.environ.get("DECIDE_BACKEND")
        os.environ["WONDERPI_LATENCY_CSV"] = handle.name
        os.environ["DECIDE_BACKEND"] = "edge"
        try:
            game.play_and_log("happy", 0.8, None, trigger="face", vision_s=0.05, move=False)
            text = Path(handle.name).read_text(encoding="utf-8-sig")
        finally:
            os.environ.pop("WONDERPI_LATENCY_CSV", None)
            if previous_backend is None:
                os.environ.pop("DECIDE_BACKEND", None)
            else:
                os.environ["DECIDE_BACKEND"] = previous_backend
            Path(handle.name).unlink(missing_ok=True)
        self.assertIn("识别到决策_s", text)
        self.assertIn("happy", text)
        self.assertIn("face", text)

    def test_start_run_does_not_raise_without_robot(self):
        img = np.zeros((48, 64, 3), dtype=np.uint8)
        game.init()
        game.start()
        out = game.run(img)
        self.assertEqual(getattr(out, "shape", None), img.shape)


if __name__ == "__main__":
    unittest.main()
