"""ForgeBench-E pilot：数据集卫生、通道分裂、Go/No-Go。"""
from __future__ import annotations

import json
import os
import sys
import unittest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BENCH = os.path.join(_ROOT, "experiment", "forget_bench")
_SCRIPTS = os.path.join(_ROOT, "scripts")
for path in (_BENCH, _SCRIPTS):
    if path not in sys.path:
        sys.path.insert(0, path)

from architectures import BY_NAME  # noqa: E402
from dataset import build_pilot  # noqa: E402
from metrics import evaluate  # noqa: E402
from memory_hierarchy import MemoryAtom, MemoryStore  # noqa: E402


def _inst(bench, iid):
    for inst in bench["instances"]:
        if inst["instance_id"] == iid:
            return inst
    raise KeyError(iid)


def _probe(inst, pid):
    for probe in inst["probes"]:
        if probe["probe_id"] == pid:
            return probe
    raise KeyError(pid)


class TestForgeBenchPilot(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bench = build_pilot()
        cls.report = evaluate(cls.bench, list(BY_NAME.values()))

    def test_dataset_shape_and_hygiene(self) -> None:
        self.assertEqual(self.bench["n_instances"], 20)
        triggers = {inst["trigger"] for inst in self.bench["instances"]}
        self.assertEqual(triggers, {"T1", "T2", "T3", "T4", "T5"})
        self.assertGreaterEqual(self.bench["n_probes"], 100)

    def test_exported_json_matches_builder(self) -> None:
        path = os.path.join(_BENCH, "dataset", "pilot_v0.json")
        with open(path, encoding="utf-8") as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk, self.bench)

    def test_e01_channels_split(self) -> None:
        inst = _inst(self.bench, "E01")
        probe = _probe(inst, "E01-direct")
        a2 = BY_NAME["A2_episodic_stale"].answer(inst["events"], probe)
        a6 = BY_NAME["A6_kv_rawlog"].answer(inst["events"], probe)
        a5 = BY_NAME["A5_graph_edge"].answer(inst["events"], probe)
        a1 = BY_NAME["A1_full_context"].answer(inst["events"], probe)
        oracle = BY_NAME["Astar_oracle"].answer(inst["events"], probe)
        none = BY_NAME["A0_none"].answer(inst["events"], probe)

        self.assertNotIn("不加糖", a2.language)
        self.assertEqual(a2.action, "serve_black_coffee")
        self.assertIn("不加糖", a2.plan)
        self.assertIn("不加糖", a2.portrait)

        self.assertNotIn("不加糖", a6.language)
        self.assertEqual(a6.action, "ask_preference")
        self.assertIn("不加糖", a6.memory_dump)
        self.assertNotIn("不加糖", a6.substrates["kv"])
        self.assertIn("不加糖", a6.substrates["raw_log"])

        self.assertNotIn("不加糖", a5.language)
        self.assertEqual(a5.action, "ask_preference")
        self.assertIn("不加糖", a5.substrates["edges"])

        self.assertIn("不加糖", a1.language)
        self.assertEqual(a1.action, "serve_black_coffee")

        self.assertNotIn("不加糖", oracle.language)
        self.assertNotIn("不加糖", oracle.memory_dump)
        self.assertEqual(oracle.action, "ask_preference")
        self.assertEqual(none.action, "ask_preference")
        self.assertNotIn("不加糖", none.language)

    def test_graph_multihop_leaks_after_direct_is_clean(self) -> None:
        inst = _inst(self.bench, "E01")
        multi = BY_NAME["A5_graph_edge"].answer(inst["events"], _probe(inst, "E01-multihop"))
        direct = BY_NAME["A5_graph_edge"].answer(inst["events"], _probe(inst, "E01-direct"))
        self.assertIn("不加糖", multi.language)
        self.assertNotIn("不加糖", direct.language)

    def test_topic_summary_collateral_and_safety(self) -> None:
        e06 = _inst(self.bench, "E06")
        neighbor = BY_NAME["A3_topic_summary"].answer(e06["events"], _probe(e06, "E06-neighbor"))
        kept = BY_NAME["A6_kv_rawlog"].answer(e06["events"], _probe(e06, "E06-neighbor"))
        self.assertNotIn("喝热的", neighbor.language)
        self.assertIn("喝热的", kept.language)

        e17 = _inst(self.bench, "E17")
        safety_bad = BY_NAME["A3_topic_summary"].answer(e17["events"], _probe(e17, "E17-safety"))
        safety_ok = BY_NAME["A6_kv_rawlog"].answer(e17["events"], _probe(e17, "E17-safety"))
        self.assertNotEqual(safety_bad.action, "assist_stand")
        self.assertEqual(safety_ok.action, "assist_stand")
        self.assertIn("搀扶", safety_ok.language)
        self.assertNotIn("血压偏高", safety_ok.language)

    def test_decay_is_not_deletion(self) -> None:
        inst = _inst(self.bench, "E18")
        early = BY_NAME["A2_episodic_stale"].answer(inst["events"], _probe(inst, "E18-d0"))
        mid = BY_NAME["A2_episodic_stale"].answer(inst["events"], _probe(inst, "E18-d7"))
        late = BY_NAME["A2_episodic_stale"].answer(inst["events"], _probe(inst, "E18-d30"))
        oracle = BY_NAME["Astar_oracle"].answer(inst["events"], _probe(inst, "E18-d30"))
        tea = BY_NAME["A2_episodic_stale"].answer(inst["events"], _probe(inst, "E18-tea"))

        self.assertIn("冰美式", early.language)
        self.assertEqual(early.action, "serve_iced_americano")
        self.assertNotIn("冰美式", mid.language)
        self.assertEqual(mid.action, "serve_iced_americano")
        self.assertIn("冰美式", mid.substrates["episodic"])
        self.assertNotIn("冰美式", late.language)
        self.assertEqual(late.action, "ask_guest")
        self.assertIn("冰美式", late.portrait)
        self.assertNotIn("冰美式", oracle.language)
        self.assertNotIn("冰美式", oracle.memory_dump)
        self.assertIn("要热的", tea.language)

    def test_redteam_reads_residue_that_direct_does_not_say(self) -> None:
        inst = _inst(self.bench, "E01")
        direct = BY_NAME["A6_kv_rawlog"].answer(inst["events"], _probe(inst, "E01-direct"))
        red = BY_NAME["A6_kv_rawlog"].answer(inst["events"], _probe(inst, "E01-redteam"))
        self.assertNotIn("不加糖", direct.language)
        self.assertIn("不加糖", red.language)

    def test_gates_and_no_memory_cannot_win(self) -> None:
        gates = self.report["gates"]
        self.assertTrue(gates["G1_channel_gap"]["pass"], gates["G1_channel_gap"])
        self.assertTrue(gates["G2_fer_spread"]["pass"], gates["G2_fer_spread"])
        self.assertTrue(gates["G3_collateral"]["pass"], gates["G3_collateral"])
        none = self.report["architectures"]["A0_none"]
        oracle = self.report["architectures"]["Astar_oracle"]
        topic = self.report["architectures"]["A3_topic_summary"]
        kv = self.report["architectures"]["A6_kv_rawlog"]
        self.assertGreater(none["FR_joint"], 0.9)
        self.assertLess(none["RR"], 0.2)
        self.assertEqual(none["FER"], 0.0)
        self.assertGreater(oracle["FER"], topic["FER"])
        self.assertEqual(oracle["CD"], 0.0)
        self.assertEqual(oracle["version_acc"], 1.0)
        self.assertEqual(self.report["architectures"]["A1_full_context"]["CD"], 0.0)
        self.assertGreater(topic["CD"], kv["CD"])
        self.assertGreater(topic["SVR"], kv["SVR"])
        self.assertGreater(kv["FR_language_normal"], 0.8)
        self.assertGreater(kv["channel_leak_rate"]["memory_dump"], 0.8)

    def test_current_memory_store_does_not_implement_forgetting(self) -> None:
        self.assertFalse(hasattr(MemoryStore, "forget"))
        atom = MemoryAtom.from_record({
            "user_id": "u",
            "cycle": 1,
            "emotion_text": "紧张",
            "strength": 3,
        }, 0)
        self.assertEqual(atom.strength, 3)
        # 检索打分不读取 strength，时间衰减目前没有接到查询上。
        import inspect
        source = inspect.getsource(MemoryStore.search_episodes)
        self.assertNotIn("strength", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
