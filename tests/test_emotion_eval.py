"""标签解析、中→英映射与端到端评测脚本的单元测试（仅标准库 + 子进程）。

运行:
  python -m unittest discover -s tests -p "test_*.py" -v
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPTS = os.path.join(_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from emotion_label_map import (  # noqa: E402
    CANON_LABELS,
    detect_dataset_from_path,
    map_text_to_label,
    parse_label_from_filename,
    resolve_default_audio_for_batch,
)


class TestEmotionLabelParsing(unittest.TestCase):
    def test_ravdess_codes(self) -> None:
        cases = {
            "03-01-01-01-02-01-12.mp4": "neutral",
            "03-01-02-01-02-01-12.mp4": "neutral",  # calm 归并到 neutral
            "03-01-03-01-02-01-12.mp4": "happy",
            "03-01-04-01-02-01-12.mp4": "sad",
            "03-01-05-01-02-01-12.mp4": "angry",
            "03-01-06-01-02-01-12.mp4": "fearful",
            "03-01-07-01-02-01-12.mp4": "disgust",
            "03-01-08-01-02-01-12.mp4": "surprised",
        }
        for name, expected in cases.items():
            self.assertEqual(parse_label_from_filename(name, "ravdess"), expected)

    def test_ravdess_bad_name(self) -> None:
        self.assertIsNone(parse_label_from_filename("not-ravdess.mp4", "ravdess"))
        self.assertIsNone(parse_label_from_filename("03-01-99-01-02-01-12.mp4", "ravdess"))

    def test_cremad_codes(self) -> None:
        cases = {
            "1001_DFA_ANG_XX.mp4": "angry",
            "1001_DFA_DIS_XX.flv": "disgust",
            "1001_DFA_FEA_XX.mp4": "fearful",
            "1001_DFA_HAP_XX.wav": "happy",
            "1001_DFA_NEU_XX.mp4": "neutral",
            "1001_DFA_SAD_XX.mp4": "sad",
        }
        for name, expected in cases.items():
            self.assertEqual(parse_label_from_filename(name, "cremad"), expected)

    def test_cremad_bad_name(self) -> None:
        self.assertIsNone(parse_label_from_filename("1001_DFA.mp4", "cremad"))
        self.assertIsNone(parse_label_from_filename("1001_DFA_ZZZ_XX.mp4", "cremad"))

    def test_unknown_dataset(self) -> None:
        self.assertIsNone(parse_label_from_filename("x.mp4", "iemocap"))

    def test_detect_dataset(self) -> None:
        self.assertEqual(detect_dataset_from_path("test_data/external/RAVDESS/Actor_01/x.mp4"), "ravdess")
        self.assertEqual(detect_dataset_from_path("test_data/external/CREMA-D/AudioMP4/x.mp4"), "cremad")
        self.assertIsNone(detect_dataset_from_path("test_data/external/MyOwn/x.mp4"))


class TestTextToLabel(unittest.TestCase):
    def test_chinese_keywords(self) -> None:
        self.assertEqual(map_text_to_label("略带紧张的期待，肩膀微微抬起"), "fearful")
        self.assertEqual(map_text_to_label("脸上露出微笑，显得很开心"), "happy")
        self.assertEqual(map_text_to_label("低头难过，眼眶湿润"), "sad")
        self.assertEqual(map_text_to_label("怒气冲冲，眉头紧锁"), "angry")
        self.assertEqual(map_text_to_label("一脸厌恶，皱眉嫌弃"), "disgust")
        self.assertEqual(map_text_to_label("睁大眼睛，十分惊讶"), "surprised")
        self.assertEqual(map_text_to_label("表情平静，无明显情绪波动"), "neutral")

    def test_english_keywords(self) -> None:
        self.assertEqual(map_text_to_label("appears anxious and nervous"), "fearful")
        self.assertEqual(map_text_to_label("a clear smile, looks happy"), "happy")
        self.assertEqual(map_text_to_label("neutral facial expression"), "neutral")

    def test_unmappable(self) -> None:
        self.assertIsNone(map_text_to_label(""))
        self.assertIsNone(map_text_to_label("xyz qqq 1234"))

    def test_labels_within_canon(self) -> None:
        out = map_text_to_label("非常开心")
        self.assertIn(out, CANON_LABELS)


class TestEvaluatorEndToEnd(unittest.TestCase):
    """构造一个最小 CSV，调用 eval_emotion_labels.py，断言输出文件结构与关键指标。"""

    def test_ravdess_minimal_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            csv_path = os.path.join(td, "run.csv")
            out_dir = os.path.join(td, "eval")
            rows = [
                # happy / 预测 happy（命中）
                ("test_data/external/RAVDESS/Actor_01/03-01-03-01-02-01-12.mp4", "脸上露出微笑，很开心", "0.10", "1.20", "1.30"),
                # sad / 预测 sad（命中）
                ("test_data/external/RAVDESS/Actor_01/03-01-04-01-02-01-12.mp4", "难过，眼眶湿润", "0.11", "1.10", "1.21"),
                # angry / 预测 happy（错配）
                ("test_data/external/RAVDESS/Actor_01/03-01-05-01-02-01-12.mp4", "微笑，很开心", "0.09", "1.05", "1.14"),
                # neutral / 文本无法解析（未知）
                ("test_data/external/RAVDESS/Actor_01/03-01-01-01-02-01-12.mp4", "xxx qqq", "0.10", "1.00", "1.10"),
            ]
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow([
                    "ts_iso", "video_relpath", "audio_path", "vision_model", "whisper_model",
                    "num_frames", "whisper_s", "vision_s", "wall_s",
                    "user_action", "emotion", "robot_reaction", "error",
                ])
                for vp, emo, ws, vs, wall in rows:
                    w.writerow([
                        "2026-05-13T03:00:00Z", vp, "", "gpt-test", "whisper-1",
                        4, ws, vs, wall, "", emo, "", "",
                    ])

            cmd = [
                sys.executable,
                os.path.join(_SCRIPTS, "eval_emotion_labels.py"),
                "--csv", csv_path,
                "--dataset", "ravdess",
                "--out-dir", out_dir,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, msg=res.stderr)

            sm = os.path.join(out_dir, "summary.json")
            cm = os.path.join(out_dir, "confusion_matrix.csv")
            pc = os.path.join(out_dir, "per_class.csv")
            sp = os.path.join(out_dir, "sample_predictions.csv")
            for p in (sm, cm, pc, sp):
                self.assertTrue(os.path.isfile(p), "缺少输出: %s" % p)

            with open(sm, "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary["rows_total"], 4)
            self.assertEqual(summary["rows_with_truth"], 4)
            self.assertEqual(summary["predictions_unknown"], 1)
            # 命中 2 / 4（unknown 视为错）
            self.assertAlmostEqual(summary["accuracy_eval_including_unknown_as_wrong"], 0.5, places=6)
            # 命中 2 / 3（仅看可解析）
            self.assertAlmostEqual(summary["accuracy_resolved_only"], 2 / 3, places=6)
            self.assertGreaterEqual(summary["macro_f1"], 0.0)
            self.assertLessEqual(summary["macro_f1"], 1.0)


class TestResolveDefaultAudio(unittest.TestCase):
    def test_crema_layout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            vf = os.path.join(td, "VideoFlash")
            aw = os.path.join(td, "AudioWAV")
            os.makedirs(vf)
            os.makedirs(aw)
            vpath = os.path.join(vf, "1061_TSI_FEA_XX.flv")
            apath = os.path.join(aw, "1061_TSI_FEA_XX.wav")
            open(vpath, "wb").close()
            open(apath, "wb").close()
            self.assertEqual(resolve_default_audio_for_batch(vpath), apath)

    def test_same_dir_wav(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            vpath = os.path.join(td, "clip.mp4")
            apath = os.path.join(td, "clip.wav")
            open(vpath, "wb").close()
            open(apath, "wb").close()
            self.assertEqual(resolve_default_audio_for_batch(vpath), apath)

    def test_real_crema_path_if_present(self) -> None:
        crema = os.path.join(
            _ROOT,
            "test_data/external/CREMA-D/CREMA-D-master/VideoFlash/1061_TSI_FEA_XX.flv",
        )
        if not os.path.isfile(crema):
            self.skipTest("本地未解压 CREMA-D")
        got = resolve_default_audio_for_batch(crema)
        self.assertTrue(got.endswith("1061_TSI_FEA_XX.wav"))
        self.assertTrue(os.path.isfile(got))


if __name__ == "__main__":
    unittest.main()
