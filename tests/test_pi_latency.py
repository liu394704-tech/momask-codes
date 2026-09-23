#!/usr/bin/env python3
"""Latency bench bookkeeping. No camera, mic, or robot."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from pipeline.bench_pi_latency import (
    COLUMNS,
    decide_impl,
    finish_round,
    probe_stack,
    write_report,
)
from pipeline.decide_edge import edge_rule_decide
from pipeline.schemas import Perception


def _perc() -> Perception:
    return Perception(
        session_id="lat",
        ts_ms=0,
        vision_emotion="happy",
        vision_conf=0.8,
        face_found=True,
        transcript="你好",
        ok=True,
        extras={"keyword": "wakeup"},
    )


class LatencyBenchTests(unittest.TestCase):
    def test_probe_reports_asr_and_llm_flags(self):
        probe = probe_stack()
        self.assertIn(probe["asr_deployed"], (True, False))
        self.assertIn(probe["llm_runtime_deployed"], (True, False))
        self.assertIn("gguf_present", probe)

    def test_rule_round_records_decide_time(self):
        row = finish_round(
            _perc(),
            trial=1,
            trigger="mic",
            requested_backend="edge",
            record_s=3.0,
            asr_s=1.25,
            vision_s=0.4,
        )
        self.assertEqual(row["实际决策"], "edge_rule")
        self.assertGreaterEqual(row["决策_s"], 0.0)
        self.assertEqual(row["识别到决策_s"], round(3.0 + 1.25 + 0.4 + row["决策_s"] + row["选动作_s"], 3))
        self.assertTrue(row["短语"])
        self.assertEqual(row["动作播放_s"], 0.0)

    def test_llm_failure_is_labeled_fallback(self):
        decision = edge_rule_decide(_perc())
        decision.reason = "edge_llm_fallback:edge_rule:happy"
        decision.error = "No module named llama_cpp"
        self.assertEqual(decide_impl(decision, "edge_auto"), "edge_rule_fallback")

    def test_csv_has_one_row(self):
        row = finish_round(
            _perc(),
            trial=2,
            trigger="timeout",
            requested_backend="edge",
            asr_error="no_keyword",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_report(Path(tmp), [row], {"asr_deployed": False})
            with path.open(encoding="utf-8-sig") as handle:
                found = list(csv.DictReader(handle))
            self.assertEqual(list(found[0].keys()), COLUMNS)
            self.assertEqual(found[0]["识别错误"], "no_keyword")
            self.assertTrue(path.with_suffix(".json").is_file())


if __name__ == "__main__":
    unittest.main()
