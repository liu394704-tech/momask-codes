"""分层记忆（记忆分层）冒烟单元测试，仅依赖标准库，强制关键词回退路径。

运行:
  python -m unittest discover -s tests -p "test_*.py" -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPTS = os.path.join(_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from memory_hierarchy import MemoryStore, estimate_tokens  # noqa: E402

_FIXTURE = os.path.join(_ROOT, "experiment/memory_smoke/fixtures/u001_session.jsonl")
_EVENTS_BASE = os.path.join(_ROOT, "experiment/offline_runs/ravdess_speech_smoke_json")

PROBE = "用户看起来很紧张害怕，神情警惕不安"


def _load_fixture() -> list[dict]:
    rows = []
    with open(_FIXTURE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class TestMemoryHierarchy(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="mem_smoke_")
        self.records = _load_fixture()
        # 关键词回退路径，保证无 faiss/embedding 环境也能跑（同时覆盖 A7）。
        self.store = MemoryStore(self.tmp, "u001", min_confidence=0.3, use_vector=False)
        self.stats = self.store.ingest(self.records)
        self.portrait = self.store.build_portrait()
        self.result = self.store.retrieve(PROBE, top_k=3, base_dir=_EVENTS_BASE)

    def test_a1_portrait_dominant_is_fearful(self) -> None:
        dominant = self.portrait["dominant_emotions"][0][0]
        self.assertEqual(dominant, "fearful")
        self.assertIn("易紧张", self.portrait["traits"])

    def test_a2_episodes_recall_only_fearful(self) -> None:
        classes = [e["emotion_class"] for e in self.result["episodes"]]
        self.assertGreater(len(classes), 0)
        self.assertTrue(all(c == "fearful" for c in classes))

    def test_a3_lowconf_dropped(self) -> None:
        # cycle 4 置信度 0.15，应被过滤，不入库也不进 Top-k。
        self.assertEqual(self.stats["n_lowconf_dropped"], 1)
        ids = [e["mem_id"] for e in self.result["episodes"]]
        self.assertNotIn("u001-0004", ids)

    def test_a4_audio_only_ingested(self) -> None:
        # cycle 5 为纯音频；入库成功且 face 模态为 False。
        store2 = MemoryStore(self.tmp, "u001", use_vector=False).load()
        atom = store2._by_id.get("u001-0005")  # noqa: SLF001
        self.assertIsNotNone(atom)
        self.assertFalse(atom.modality_present.get("face"))
        self.assertEqual(atom.emotion_class, "sad")

    def test_a5_event_resolves_to_real_json(self) -> None:
        resolved = False
        for ev in self.result["events"]:
            if ev.get("source_ref"):
                self.assertTrue(ev["exists"], ev)
                self.assertIn("payload", ev)
                resolved = True
                break
        self.assertTrue(resolved, "应至少有一条命中记忆能回链到真实离线 JSON")

    def test_a7_keyword_backend_available(self) -> None:
        self.assertEqual(self.store.backend, "keyword")

    def test_a8_prompt_within_budget(self) -> None:
        ctx = self.store.build_prompt_context(PROBE, top_k=3)
        self.assertLess(estimate_tokens(ctx), 800)
        self.assertIn("主导情绪", ctx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
