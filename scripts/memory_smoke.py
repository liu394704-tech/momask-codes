"""分层记忆冒烟测试运行器（全离线、零 API）。

单案例：
  python scripts/memory_smoke.py

批量案例（推荐，用于证明落地性）：
  python scripts/memory_smoke.py --batch

  # 强制关键词回退路径
  python scripts/memory_smoke.py --batch --force-keyword
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from memory_hierarchy import MemoryStore, estimate_tokens  # noqa: E402

_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

MIN_CONFIDENCE = 0.3
TOP_K = 3
LATENCY_P95_MS = 50.0
PROMPT_TOKEN_BUDGET = 800

DEFAULT_FIXTURE = os.path.join(_ROOT, "experiment/memory_smoke/fixtures/u001_session.jsonl")
DEFAULT_CASE = {
    "case_id": "u001_fearful",
    "description": "紧张型用户（默认单案例）",
    "fixture": "u001_session.jsonl",
    "probe_query": "用户看起来很紧张害怕，神情警惕不安",
    "expected_dominant": "fearful",
    "expected_retrieval_class": "fearful",
    "lowconf_cycle": 4,
    "audio_only_cycle": 5,
}


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _count_relevant(atoms: list, target_class: str) -> int:
    return sum(1 for a in atoms if a.emotion_class == target_class)


def _compute_metrics(
    store: MemoryStore,
    records: list[dict],
    expected_dominant: str,
    expected_retrieval_class: str,
    lowconf_cycle: Optional[int],
    audio_only_cycle: Optional[int],
    result: dict,
    latencies_ms: list[float],
    prompt_tokens: int,
    ingest_stats: dict,
) -> dict[str, Any]:
    user_id = store.user_id
    episodes = result["episodes"]
    episode_ids = [e["mem_id"] for e in episodes]
    episode_classes = [e["emotion_class"] for e in episodes]

    n_in = len(records)
    n_kept = len(store._atoms)  # noqa: SLF001
    n_relevant = _count_relevant(store._atoms, expected_retrieval_class)  # noqa: SLF001
    n_hits = sum(1 for c in episode_classes if c == expected_retrieval_class)

    lowconf_id = f"{user_id}-{lowconf_cycle:04d}" if lowconf_cycle else None
    audio_id = f"{user_id}-{audio_only_cycle:04d}" if audio_only_cycle else None

    event_ok = False
    event_detail = None
    n_events_with_ref = 0
    n_events_resolved = 0
    for ev in result["events"]:
        if ev.get("source_ref"):
            n_events_with_ref += 1
            if ev.get("exists"):
                n_events_resolved += 1
            if event_detail is None:
                event_detail = ev
                event_ok = bool(ev.get("exists"))

    audio_atom = store._by_id.get(audio_id) if audio_id else None  # noqa: SLF001
    dominant = (result["portrait"].get("dominant_emotions") or [["", 0]])[0][0]
    dominant_share = (result["portrait"].get("dominant_emotions") or [["", 0]])[0][1]

    assertions = {
        "A1_portrait_dominant": dominant == expected_dominant,
        "A2_episodes_recall_target_class": (
            len(episode_ids) > 0 and all(c == expected_retrieval_class for c in episode_classes)
        ),
        "A3_lowconf_filtered": (
            lowconf_id is None
            or (lowconf_id not in store._by_id and lowconf_id not in episode_ids)  # noqa: SLF001
        ),
        "A4_audio_only_ingested": (
            audio_id is None
            or bool(audio_atom and audio_atom.modality_present.get("face") is False)
        ),
        "A5_event_resolved": event_ok or n_events_with_ref == 0,
        "A7_backend_available": store.backend in ("keyword", "vector"),
        "A8_prompt_within_budget": prompt_tokens < PROMPT_TOKEN_BUDGET,
    }

    metrics = {
        "ingest_retention_rate": round(n_kept / n_in, 4) if n_in else 0.0,
        "lowconf_filter_rate": round(ingest_stats.get("n_lowconf_dropped", 0) / max(n_in, 1), 4),
        "dominant_emotion": dominant,
        "dominant_share": dominant_share,
        "recall_at_k": round(n_hits / n_relevant, 4) if n_relevant else 0.0,
        "precision_at_k": round(n_hits / len(episode_classes), 4) if episode_classes else 0.0,
        "n_relevant_in_store": n_relevant,
        "n_hits_in_top_k": n_hits,
        "top_k": TOP_K,
        "event_resolution_rate": (
            round(n_events_resolved / n_events_with_ref, 4) if n_events_with_ref else None
        ),
        "latency_p95_ms": round(_percentile(latencies_ms, 95), 3),
        "prompt_tokens": prompt_tokens,
        "prompt_budget_usage": round(prompt_tokens / PROMPT_TOKEN_BUDGET, 4),
    }

    return {
        "assertions": assertions,
        "metrics": metrics,
        "event_detail": event_detail,
        "all_passed": all(assertions.values()),
    }


def run_case(
    case: dict,
    fixture_dir: str,
    store_root: str,
    events_base: str,
    force_keyword: bool = False,
) -> dict:
    fixture_path = case["fixture"]
    if not os.path.isabs(fixture_path):
        fixture_path = os.path.join(fixture_dir, fixture_path)

    records = _read_jsonl(fixture_path)
    user_id = str(records[0].get("user_id", "anon")) if records else "anon"
    probe_query = case["probe_query"]
    expected_dominant = case["expected_dominant"]
    expected_retrieval_class = case["expected_retrieval_class"]
    lowconf_cycle = case.get("lowconf_cycle")
    audio_only_cycle = case.get("audio_only_cycle")

    store = MemoryStore(
        store_root,
        user_id,
        min_confidence=MIN_CONFIDENCE,
        use_vector=(False if force_keyword else None),
    )
    ingest_stats = store.ingest(records)
    portrait = store.build_portrait()

    latencies_ms: list[float] = []
    result = None
    for _ in range(20):
        t0 = time.perf_counter()
        result = store.retrieve(probe_query, top_k=TOP_K, base_dir=events_base)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    result["portrait"] = portrait

    prompt_ctx = store.build_prompt_context(probe_query, top_k=TOP_K)
    prompt_tokens = estimate_tokens(prompt_ctx)

    eval_out = _compute_metrics(
        store, records, expected_dominant, expected_retrieval_class,
        lowconf_cycle, audio_only_cycle, result, latencies_ms, prompt_tokens, ingest_stats,
    )

    return {
        "case_id": case.get("case_id", user_id),
        "description": case.get("description", ""),
        "user_id": user_id,
        "backend": store.backend,
        "force_keyword": force_keyword,
        "ingest_stats": ingest_stats,
        "portrait": portrait,
        "probe_query": probe_query,
        "expected_dominant": expected_dominant,
        "expected_retrieval_class": expected_retrieval_class,
        "retrieved_episodes": result["episodes"],
        "prompt_preview": prompt_ctx,
        "assertions": eval_out["assertions"],
        "metrics": eval_out["metrics"],
        "event_detail": eval_out["event_detail"],
        "all_passed": eval_out["all_passed"],
    }


def run_smoke(
    fixture: str,
    store_root: str,
    events_base: str,
    force_keyword: bool = False,
) -> dict:
    case = dict(DEFAULT_CASE)
    case["fixture"] = fixture if os.path.isabs(fixture) else os.path.basename(fixture)
    fixture_dir = os.path.dirname(fixture) if os.path.isabs(fixture) else os.path.join(
        _ROOT, "experiment/memory_smoke/fixtures"
    )
    return run_case(case, fixture_dir, store_root, events_base, force_keyword)


def run_batch(
    cases_path: str,
    fixture_dir: str,
    store_root: str,
    events_base: str,
    force_keyword: bool = False,
) -> dict:
    with open(cases_path, encoding="utf-8") as f:
        cases = json.load(f)
    results = [run_case(c, fixture_dir, store_root, events_base, force_keyword) for c in cases]

    n_pass = sum(1 for r in results if r["all_passed"])
    summary = {
        "n_cases": len(results),
        "n_passed": n_pass,
        "pass_rate": round(n_pass / len(results), 4) if results else 0.0,
        "avg_recall_at_k": round(
            sum(r["metrics"]["recall_at_k"] for r in results) / len(results), 4
        ) if results else 0.0,
        "avg_precision_at_k": round(
            sum(r["metrics"]["precision_at_k"] for r in results) / len(results), 4
        ) if results else 0.0,
        "avg_prompt_tokens": round(
            sum(r["metrics"]["prompt_tokens"] for r in results) / len(results), 1
        ) if results else 0.0,
    }
    return {"summary": summary, "cases": results}


def main() -> int:
    ap = argparse.ArgumentParser(description="分层记忆冒烟测试")
    ap.add_argument("--fixture", default=DEFAULT_FIXTURE)
    ap.add_argument(
        "--store-root",
        default=os.path.join(_ROOT, "experiment/memory_smoke/memory_store"),
    )
    ap.add_argument(
        "--events-base",
        default=os.path.join(_ROOT, "experiment/offline_runs/ravdess_speech_smoke_json"),
    )
    ap.add_argument(
        "--out",
        default=os.path.join(_ROOT, "experiment/memory_smoke/results/memory_smoke_report.json"),
    )
    ap.add_argument(
        "--cases",
        default=os.path.join(_ROOT, "experiment/memory_smoke/fixtures/cases.json"),
    )
    ap.add_argument(
        "--fixture-dir",
        default=os.path.join(_ROOT, "experiment/memory_smoke/fixtures"),
    )
    ap.add_argument("--batch", action="store_true", help="批量运行 cases.json 中全部案例")
    ap.add_argument("--force-keyword", action="store_true", help="强制关键词回退路径")
    args = ap.parse_args()

    if args.batch:
        report = run_batch(
            args.cases, args.fixture_dir, args.store_root, args.events_base, args.force_keyword
        )
        out_path = os.path.join(os.path.dirname(args.out), "memory_smoke_batch_report.json")
    else:
        report = run_smoke(args.fixture, args.store_root, args.events_base, args.force_keyword)
        out_path = args.out

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if args.batch:
        s = report["summary"]
        print(f"[memory-smoke batch] {s['n_passed']}/{s['n_cases']} passed  pass_rate={s['pass_rate']}")
        for c in report["cases"]:
            m = c["metrics"]
            mark = "PASS" if c["all_passed"] else "FAIL"
            print(
                f"  [{mark}] {c['case_id']}: dominant={m['dominant_emotion']}({m['dominant_share']}) "
                f"Recall@{m['top_k']}={m['recall_at_k']} P@{m['top_k']}={m['precision_at_k']} "
                f"tokens={m['prompt_tokens']}"
            )
        print(f"[memory-smoke batch] report -> {out_path}")
        return 0 if s["n_passed"] == s["n_cases"] else 1

    print(f"[memory-smoke] backend={report['backend']}  all_passed={report['all_passed']}")
    for name, ok in report["assertions"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    m = report["metrics"]
    print(
        f"  (metrics) Recall@{m['top_k']}={m['recall_at_k']} P@{m['top_k']}={m['precision_at_k']} "
        f"latency_p95={m['latency_p95_ms']}ms tokens={m['prompt_tokens']}"
    )
    print(f"[memory-smoke] report -> {out_path}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
