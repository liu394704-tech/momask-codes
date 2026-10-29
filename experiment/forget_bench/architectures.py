"""ForgeBench-E pilot 的参照架构。

这些不是 Mem0 / RoboMemory 的复现，而是一组确定性策略，用来把
「删除有没有落到各个存放处」从语言模型的理解误差里拆出来。
探针上的 target_keys 只给策略定位该看哪条记忆，不会写进回答模板。

红队探针会把全部 substrate 追加进语言通道，用来表示「被要求把记录原文说出来」。
普通探针只读该架构正常会用来回答和行动的那一层。
"""
from __future__ import annotations

from typing import Optional

ABSTAIN = "我没有这项记录。"
CONFIRM = "我需要再确认一下。"
ACK = "好的。"

BUDGET = 4
DECAY_HIDE_DAYS = 7
DECAY_ACT_DAYS = 30


class Trace:
    def __init__(
        self,
        language: str,
        action: str,
        plan: str,
        memory_dump: str,
        portrait: str,
        substrates: dict[str, str],
    ) -> None:
        self.language = language
        self.action = action
        self.plan = plan
        self.memory_dump = memory_dump
        self.portrait = portrait
        self.substrates = substrates

    def as_dict(self) -> dict:
        return {
            "language": self.language,
            "action": self.action,
            "plan": self.plan,
            "memory_dump": self.memory_dump,
            "portrait": self.portrait,
            "substrates": dict(self.substrates),
        }


def _keys_of(ev: dict) -> list[str]:
    if ev.get("keys"):
        return list(ev["keys"])
    if ev.get("key"):
        return [ev["key"]]
    return []


def _payload(ev: dict, seq: int, prev: Optional[dict] = None) -> dict:
    prev = prev or {}
    return {
        "key": ev["key"],
        "text": ev["text"],
        "topic": ev.get("topic", prev.get("topic", "")),
        "action": ev.get("action", prev.get("action", "")),
        "baseline_action": ev.get("baseline_action", prev.get("baseline_action", "")),
        "priority": int(ev.get("priority", prev.get("priority", 1))),
        "day": int(ev["day"]),
        "seq": seq if ev["op"] == "ADD" else int(prev.get("seq", seq)),
        "links": list(ev.get("links", prev.get("links", []))),
    }


def _evict_fifo(store: dict[str, dict], budget: int) -> dict[str, dict]:
    if len(store) <= budget:
        return store
    ordered = sorted(store.items(), key=lambda kv: (kv[1]["day"], kv[1]["seq"], kv[0]))
    return dict(ordered[-budget:])


def _evict_priority(store: dict[str, dict], budget: int) -> dict[str, dict]:
    if len(store) <= budget:
        return store
    ordered = sorted(store.items(), key=lambda kv: (kv[1]["priority"], kv[1]["day"], kv[1]["seq"], kv[0]))
    drop_n = len(store) - budget
    return dict(ordered[drop_n:])


def _pick_action(probe: dict, available: dict[str, str]) -> str:
    if not probe["target_keys"]:
        return probe["baseline_action"]
    for key in probe["target_keys"]:
        if key in available:
            return available[key]
    return probe["baseline_action"]


def _pick_text(probe: dict, available: dict[str, str]) -> str:
    if not probe["target_keys"]:
        return ACK
    parts = [available[key] for key in probe["target_keys"] if key in available and available[key]]
    if not parts:
        return ABSTAIN
    return " ".join(parts)


def _pack(
    probe: dict,
    language: str,
    action: str,
    plan: str,
    substrates: dict[str, str],
    portrait: str = "",
) -> Trace:
    if probe["family"] == "redteam":
        extra = "\n".join(v for v in substrates.values() if v)
        language = f"{language}\n{extra}" if extra else language
    dump = "\n".join(v for v in substrates.values() if v)
    return Trace(language, action, plan, dump, portrait, substrates)


def _events_until(events: list[dict], sim_day: int) -> list[dict]:
    return [ev for ev in events if int(ev["day"]) <= int(sim_day)]


class _Policy:
    name = "base"
    label = "base"

    def answer(self, events: list[dict], probe: dict) -> Trace:
        raise NotImplementedError


class NoMemory(_Policy):
    """从未写入。遗忘探针上的沉默不计分，除非保留集也还在。"""

    name = "A0_none"
    label = "无记忆对照"

    def answer(self, events: list[dict], probe: dict) -> Trace:
        del events
        if probe["target_keys"]:
            language, plan = CONFIRM, "no_memory"
        else:
            language, plan = ACK, "control"
        return _pack(probe, language, probe["baseline_action"], plan, {})


class FullContext(_Policy):
    """删除指令只是又一条原文，旧值一直留在历史里。"""

    name = "A1_full_context"
    label = "全上下文"

    def answer(self, events: list[dict], probe: dict) -> Trace:
        current: dict[str, dict] = {}
        history: dict[str, list[str]] = {}
        seq = 0
        for ev in _events_until(events, probe["sim_day"]):
            if ev["op"] not in ("ADD", "UPDATE"):
                continue
            seq += 1
            current[ev["key"]] = _payload(ev, seq, current.get(ev["key"]))
            history.setdefault(ev["key"], []).append(ev["text"])
        texts = {k: " ".join(history.get(k, [])) for k in history}
        language = _pick_text(probe, texts)
        action = _pick_action(probe, {k: v["action"] for k, v in current.items()})
        substrates = {
            "history": "\n".join(t for ts in history.values() for t in ts),
        }
        plan = "full_context_replay" if probe["target_keys"] else "control"
        if language not in (ABSTAIN, ACK, CONFIRM):
            plan = "full_context:" + language
        return _pack(probe, language, action, plan, substrates)


class EpisodicStale(_Policy):
    """情节层可以删，行动缓存和画像不失效；衰减只遮住检索，不擦存储。

    对应现在 memory_hierarchy 里「没有 forget()、strength 不参与检索」
    再往下接一个会缓存 robot_reaction 的动作策略。
    """

    name = "A2_episodic_stale"
    label = "情节层+陈旧侧通道"

    def answer(self, events: list[dict], probe: dict) -> Trace:
        episodic: dict[str, dict] = {}
        cache: dict[str, dict] = {}
        portrait: list[str] = []
        raw: list[str] = []
        last_day: dict[str, int] = {}
        seq = 0
        for ev in _events_until(events, probe["sim_day"]):
            if ev["op"] in ("ADD", "UPDATE"):
                seq += 1
                fact = _payload(ev, seq, episodic.get(ev["key"]) or cache.get(ev["key"]))
                episodic[ev["key"]] = fact
                cache[ev["key"]] = {"action": fact["action"], "text": fact["text"], "day": fact["day"], "seq": fact["seq"]}
                portrait.append(fact["text"])
                raw.append(fact["text"])
                last_day[ev["key"]] = int(ev["day"])
            elif ev["op"] == "DELETE":
                for key in _keys_of(ev):
                    episodic.pop(key, None)
                if ev.get("utterance"):
                    raw.append(ev["utterance"])
        episodic = _evict_fifo(episodic, BUDGET)
        sim = int(probe["sim_day"])
        visible = {
            k: v["text"]
            for k, v in episodic.items()
            if sim - last_day.get(k, sim) < DECAY_HIDE_DAYS
        }
        actionable = {}
        for key, rec in cache.items():
            if sim - last_day.get(key, sim) >= DECAY_ACT_DAYS:
                continue
            actionable[key] = rec["action"]
        language = _pick_text(probe, visible)
        action = _pick_action(probe, actionable)
        cache_txt = " ".join(cache[k]["text"] for k in probe["target_keys"] if k in cache)
        plan = f"action_cache:{cache_txt}" if cache_txt else "action_cache_miss"
        portrait_txt = "\n".join(portrait)
        substrates = {
            "episodic": "\n".join(v["text"] for v in episodic.values()),
            "action_cache": "\n".join(v["text"] for v in cache.values()),
            "portrait": portrait_txt,
            "raw_log": "\n".join(raw),
        }
        return _pack(probe, language, action, plan, substrates, portrait=portrait_txt)


class TopicSummary(_Policy):
    """摘要按 topic 分桶。删除一条时整桶丢掉，不做时间衰减。"""

    name = "A3_topic_summary"
    label = "主题桶摘要"

    def answer(self, events: list[dict], probe: dict) -> Trace:
        facts: dict[str, dict] = {}
        seq = 0
        for ev in _events_until(events, probe["sim_day"]):
            if ev["op"] == "ADD":
                seq += 1
                facts[ev["key"]] = _payload(ev, seq)
            elif ev["op"] == "UPDATE":
                seq += 1
                facts[ev["key"]] = _payload(ev, seq, facts.get(ev["key"]))
            elif ev["op"] == "DELETE":
                keys = _keys_of(ev)
                topics = {facts[k]["topic"] for k in keys if k in facts and facts[k]["topic"]}
                for key in list(facts):
                    if key in keys or facts[key]["topic"] in topics:
                        facts.pop(key, None)
        facts = _evict_fifo(facts, BUDGET)
        texts = {k: v["text"] for k, v in facts.items()}
        language = _pick_text(probe, texts)
        action = _pick_action(probe, {k: v["action"] for k, v in facts.items()})
        plan = "summary_hit" if language not in (ABSTAIN, ACK, CONFIRM) else "summary_miss"
        summary = "\n".join(v["text"] for v in facts.values())
        return _pack(probe, language, action, plan, {"summary": summary}, portrait="")


class GraphEdge(_Policy):
    """节点可以删、可以换版本；关联边和 superseded 边保留旧文。"""

    name = "A5_graph_edge"
    label = "图记忆（边残留）"

    def answer(self, events: list[dict], probe: dict) -> Trace:
        nodes: dict[str, dict] = {}
        edges: list[dict] = []
        seq = 0
        for ev in _events_until(events, probe["sim_day"]):
            if ev["op"] == "ADD":
                seq += 1
                fact = _payload(ev, seq)
                for other in nodes.values():
                    same = fact["topic"] and other["topic"] == fact["topic"]
                    linked = other["key"] in fact["links"] or fact["key"] in other.get("links", [])
                    if same or linked:
                        edges.append({
                            "keys": [fact["key"], other["key"]],
                            "text": f"{fact['text']} 与 {other['text']} 相关",
                        })
                nodes[fact["key"]] = fact
            elif ev["op"] == "UPDATE":
                seq += 1
                old = nodes.get(ev["key"])
                if old:
                    edges.append({"keys": [ev["key"]], "text": f"superseded {old['text']}"})
                nodes[ev["key"]] = _payload(ev, seq, old)
            elif ev["op"] == "DELETE":
                for key in _keys_of(ev):
                    nodes.pop(key, None)
        texts = {k: v["text"] for k, v in nodes.items()}
        language = _pick_text(probe, texts)
        if probe["family"] == "multihop":
            extra = [e["text"] for e in edges if set(e["keys"]) & set(probe["target_keys"])]
            if extra:
                language = f"{language}\n" + "\n".join(extra)
        action = _pick_action(probe, {k: v["action"] for k, v in nodes.items()})
        plan = "graph_node_hit" if any(k in nodes for k in probe["target_keys"]) else "graph_node_miss"
        if not probe["target_keys"]:
            plan = "control"
        substrates = {
            "nodes": "\n".join(v["text"] for v in nodes.values()),
            "edges": "\n".join(e["text"] for e in edges),
        }
        return _pack(probe, language, action, plan, substrates)


class KvRawLog(_Policy):
    """键值层精确删除，原始日志只追加。容量压力只淘汰键值层。"""

    name = "A6_kv_rawlog"
    label = "键值+原始日志"

    def answer(self, events: list[dict], probe: dict) -> Trace:
        kv: dict[str, dict] = {}
        raw: list[str] = []
        seq = 0
        for ev in _events_until(events, probe["sim_day"]):
            if ev["op"] == "ADD":
                seq += 1
                kv[ev["key"]] = _payload(ev, seq)
                raw.append(ev["text"])
            elif ev["op"] == "UPDATE":
                seq += 1
                kv[ev["key"]] = _payload(ev, seq, kv.get(ev["key"]))
                raw.append(ev["text"])
            elif ev["op"] == "DELETE":
                for key in _keys_of(ev):
                    kv.pop(key, None)
                if ev.get("utterance"):
                    raw.append(ev["utterance"])
        kv = _evict_priority(kv, BUDGET)
        texts = {k: v["text"] for k, v in kv.items()}
        language = _pick_text(probe, texts)
        action = _pick_action(probe, {k: v["action"] for k, v in kv.items()})
        hit = any(k in kv for k in probe["target_keys"])
        plan = "kv_hit" if hit else ("control" if not probe["target_keys"] else "kv_miss")
        substrates = {
            "kv": "\n".join(v["text"] for v in kv.values()),
            "raw_log": "\n".join(raw),
        }
        return _pack(probe, language, action, plan, substrates)


class OracleDelete(_Policy):
    """上界：同一条策略清掉所有存放处，衰减是真删除，容量淘汰也清日志。"""

    name = "Astar_oracle"
    label = "Oracle 全通道删除"

    def answer(self, events: list[dict], probe: dict) -> Trace:
        kv: dict[str, dict] = {}
        last_day: dict[str, int] = {}
        seq = 0
        for ev in _events_until(events, probe["sim_day"]):
            if ev["op"] == "ADD":
                seq += 1
                kv[ev["key"]] = _payload(ev, seq)
                last_day[ev["key"]] = int(ev["day"])
            elif ev["op"] == "UPDATE":
                seq += 1
                kv[ev["key"]] = _payload(ev, seq, kv.get(ev["key"]))
                last_day[ev["key"]] = int(ev["day"])
            elif ev["op"] == "DELETE":
                for key in _keys_of(ev):
                    kv.pop(key, None)
                    last_day.pop(key, None)
        sim = int(probe["sim_day"])
        for key in list(kv):
            if sim - last_day.get(key, sim) >= DECAY_HIDE_DAYS:
                kv.pop(key, None)
        kv = _evict_priority(kv, BUDGET)
        texts = {k: v["text"] for k, v in kv.items()}
        language = _pick_text(probe, texts)
        action = _pick_action(probe, {k: v["action"] for k, v in kv.items()})
        hit = any(k in kv for k in probe["target_keys"])
        plan = "oracle_hit" if hit else ("control" if not probe["target_keys"] else "oracle_miss")
        blob = "\n".join(v["text"] for v in kv.values())
        return _pack(probe, language, action, plan, {"kv": blob})


ARCHITECTURES: list[_Policy] = [
    NoMemory(),
    FullContext(),
    EpisodicStale(),
    TopicSummary(),
    GraphEdge(),
    KvRawLog(),
    OracleDelete(),
]

BY_NAME = {a.name: a for a in ARCHITECTURES}
