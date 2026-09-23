#!/usr/bin/env python3
"""WonderPi face-game contract: init/start/stop/exit/run without a robot."""
from __future__ import annotations

import unittest

import numpy as np

from pipeline import wonderpi_face_game as game


class WonderPiContractTests(unittest.TestCase):
    def tearDown(self):
        game.stop()
        game._RUNNING = False
        game._BUSY = False
        game._PENDING_KEYWORD = None

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

    def test_start_run_does_not_raise_without_robot(self):
        img = np.zeros((48, 64, 3), dtype=np.uint8)
        game.init()
        game.start()
        out = game.run(img)
        self.assertEqual(getattr(out, "shape", None), img.shape)


if __name__ == "__main__":
    unittest.main()
