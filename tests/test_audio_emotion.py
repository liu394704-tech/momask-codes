#!/usr/bin/env python3
from __future__ import annotations

import unittest

from pipeline.audio_emotion import SerResult, infer_wav, to_robot_emotion


class SerMapTests(unittest.TestCase):
    def test_maps_open_source_labels(self):
        self.assertEqual(to_robot_emotion("happy"), "happy")
        self.assertEqual(to_robot_emotion("生气/angry"), "unhappy")
        self.assertEqual(to_robot_emotion("sad"), "unhappy")
        self.assertEqual(to_robot_emotion("surprised"), "surprised")
        self.assertEqual(to_robot_emotion("neutral"), "neutral")
        self.assertIsNone(to_robot_emotion(""))

    def test_missing_wav_does_not_crash(self):
        result = infer_wav("/tmp/definitely-missing-ser.wav")
        self.assertIsInstance(result, SerResult)
        self.assertFalse(result.ok)


if __name__ == "__main__":
    unittest.main()
