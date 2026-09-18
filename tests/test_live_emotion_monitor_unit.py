"""不依赖摄像头、不调用网络的单元测试（标准库 unittest）。

需在已安装 opencv-python、numpy 等依赖的环境中运行，例如:
  conda activate momask
  pip install opencv-python openai
  python -m unittest discover -s tests -p "test_*.py" -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import numpy as np
    import live_emotion_monitor as lem

    _HAS_LIVE_MONITOR = True
except ImportError:
    _HAS_LIVE_MONITOR = False
    np = None  # type: ignore[assignment]
    lem = None  # type: ignore[assignment]


@unittest.skipUnless(
    _HAS_LIVE_MONITOR,
    "无法导入 live_emotion_monitor（需 opencv-python、numpy 等）。请在 momask conda 环境中运行测试。",
)
class TestLiveEmotionMonitorHelpers(unittest.TestCase):
    def test_env_bool(self) -> None:
        os.environ["_LM_TEST_FLAG"] = "1"
        try:
            self.assertTrue(lem._env_bool("_LM_TEST_FLAG", False))
            os.environ["_LM_TEST_FLAG"] = "0"
            self.assertFalse(lem._env_bool("_LM_TEST_FLAG", True))
        finally:
            os.environ.pop("_LM_TEST_FLAG", None)
        self.assertTrue(lem._env_bool("_LM_MISSING_X", True))
        self.assertFalse(lem._env_bool("_LM_MISSING_X", False))

    def test_resize_noop_and_downscale(self) -> None:
        narrow = np.zeros((120, 400, 3), dtype=np.uint8)
        self.assertEqual(lem.resize_frame_for_vision(narrow, 640).shape, narrow.shape)
        wide = np.zeros((200, 1600, 3), dtype=np.uint8)
        out = lem.resize_frame_for_vision(wide, 640)
        self.assertEqual(out.shape[1], 640)
        self.assertLess(out.shape[0], wide.shape[0])

    def test_resize_max_width_zero(self) -> None:
        frame = np.ones((50, 100, 3), dtype=np.uint8)
        out = lem.resize_frame_for_vision(frame, 0)
        self.assertEqual(out.shape, frame.shape)

    def test_get_audio_path_no_mic_no_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p, diag = lem.get_audio_path_for_cycle(
                use_mic=False,
                demo_path=os.path.join(td, "missing.wav"),
                audio_sec=1.0,
                temp_dir=td,
            )
            self.assertIsNone(p)
            self.assertEqual(diag, "")

    def test_vision_fields(self) -> None:
        d = {"user_action": " 坐 ", "emotion": " 静 ", "robot_reaction": " nod "}
        ua, em, rr = lem._vision_fields(d)
        self.assertEqual(ua, "坐")
        self.assertEqual(em, "静")
        self.assertEqual(rr, "nod")

    def test_default_experiment_csv_relative(self) -> None:
        rel = lem._default_experiment_csv_relative()
        self.assertTrue(rel.startswith("experiment/exp_"))
        self.assertTrue(rel.endswith(".csv"))


if __name__ == "__main__":
    unittest.main()
