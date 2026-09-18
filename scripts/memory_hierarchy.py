"""分层记忆（Memory Hierarchy）——仅依赖标准库的可运行实现。

借鉴 MemoryBank（arXiv:2305.10250）的「存储 / 检索 / 画像」思路，适配本项目的
多模态情感识别管线：上游 MLLM 已经把人脸表情、肢体动作、语音转写整理成结构化
文本（user_action / emotion / robot_reaction），本模块在其之上做长期记忆分层。

三层：
  - 语义层（portrait）：用户长期情绪画像，直接按 user_id 读取，毫秒级。
  - 情节层（episodic）：每轮感知记忆，向量检索（可选）或关键词回退（默认）。
  - 事件层（event）：按需把某条记忆精确回链到原始离线 JSON（source_ref）。

设计原则：
  - 纯标准库即可跑通「关键词回退 + 全部断言」，无需联网、无需 API。
  - 句向量 / FAISS 为可选增强；缺失时自动回退，保证冒烟在任何环境可复现。
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

# 复用既有标签映射，保证情绪类别口径与离线评测一致。
try:  # 作为脚本 / 包导入两种方式都能工作
    from emotion_label_map import CANON_LABELS, map_text_to_label
except ImportError:  # pragma: no cover - 仅在打包路径不同的情况下触发
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from emotion_label_map import CANON_LABELS, map_text_to_label


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class MemoryAtom:
    """情节层的最小存储单元，对应一轮多模态感知结果。"""

    mem_id: str
    user_id: str
    ts: str
    cycle: int
    emotion_text: str
    user_action: str = ""
    robot_reaction: str = ""
    transcript: str = ""
    confidence: float = 1.0
    modality_present: dict = field(default_factory=lambda: {"face": True, "audio": False})
    emotion_class: Optional[str] = None
    source_ref: str = ""
    # 给后续「遗忘曲线」冒烟预留，本模块只占位、不计算。
    strength: int = 1
    last_recalled_ts: Optional[str] = None

    def text_blob(self) -> str:
        """用于检索 / 映射的文本拼接。"""
        return " ".join(p for p in (self.emotion_text, self.user_action, self.transcript) if p)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_record(cls, rec: dict, idx: int) -> "MemoryAtom":
        """把一条原始会话记录规范化为记忆原子，自动补全 emotion_class。"""
        user_id = str(rec.get("user_id", "anon"))
        cycle = int(rec.get("cycle", idx))
        mem_id = rec.get("mem_id") or f"{user_id}-{cycle:04d}"
        emotion_text = str(rec.get("emotion_text") or rec.get("emotion") or "")
        atom = cls(
            mem_id=mem_id,
            user_id=user_id,
            ts=str(rec.get("ts") or ""),
            cycle=cycle,
            emotion_text=emotion_text,
            user_action=str(rec.get("user_action") or ""),
            robot_reaction=str(rec.get("robot_reaction") or ""),
            transcript=str(rec.get("transcript") or ""),
            confidence=float(rec.get("confidence", 1.0)),
            modality_present=dict(rec.get("modality_present") or {"face": True, "audio": False}),
            emotion_class=rec.get("emotion_class"),
            source_ref=str(rec.get("source_ref") or ""),
            strength=int(rec.get("strength", 1)),
            last_recalled_ts=rec.get("last_recalled_ts"),
        )
        if not atom.emotion_class:
            atom.emotion_class = map_text_to_label(atom.text_blob())
        return atom


# --------------------------------------------------------------------------- #
# 关键词检索（标准库回退路径）
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[a-zA-Z]+|[\u4e00-\u9fff]")


def _tokenize(text: str) -> list[str]:
    """英文按词、中文按字切分，足够支撑关键词重叠打分。"""
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _keyword_score(query_tokens: list[str], atom: MemoryAtom, query_class: Optional[str]) -> float:
    """情绪类别命中给强权重，再叠加 token 重叠（Jaccard 近似）。"""
    atom_tokens = set(_tokenize(atom.text_blob()))
    if not atom_tokens:
        return 0.0
    q = set(query_tokens)
    overlap = len(q & atom_tokens)
    union = len(q | atom_tokens) or 1
    score = overlap / union
    if query_class and atom.emotion_class == query_class:
        score += 1.0
    return score


# --------------------------------------------------------------------------- #
# 可选向量后端（句向量 + FAISS / numpy）
# --------------------------------------------------------------------------- #
class _VectorBackend:
    """句向量检索后端；任一依赖缺失即判定不可用，由上层回退到关键词。"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import numpy as np  # type: ignore

        self._np = np
        self._model = SentenceTransformer(model_name)
        self._matrix = None  # (N, D) 归一化向量
        self._mem_ids: list[str] = []

    def _encode(self, texts: list[str]):
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return self._np.asarray(vecs, dtype="float32")

    def build(self, atoms: list[MemoryAtom]) -> None:
        self._mem_ids = [a.mem_id for a in atoms]
        self._matrix = self._encode([a.text_blob() for a in atoms]) if atoms else None

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._matrix is None or len(self._mem_ids) == 0:
            return []
        q = self._encode([query])[0]
        sims = self._matrix @ q  # 余弦相似度（向量已归一化）
        order = self._np.argsort(-sims)[:top_k]
        return [(self._mem_ids[i], float(sims[i])) for i in order]


def _try_vector_backend() -> Optional[_VectorBackend]:
    try:
        return _VectorBackend()
    except Exception:  # 依赖缺失 / 加载失败 -> 回退关键词
        return None


# --------------------------------------------------------------------------- #
# 记忆库
# --------------------------------------------------------------------------- #
class MemoryStore:
    """单用户的三层记忆库，落盘到 root/<user_id>/ 下。"""

    def __init__(
        self,
        root: str,
        user_id: str,
        min_confidence: float = 0.3,
        use_vector: Optional[bool] = None,
    ) -> None:
        self.user_id = user_id
        self.min_confidence = float(min_confidence)
        self.user_dir = os.path.join(root, user_id)
        os.makedirs(self.user_dir, exist_ok=True)
        self.episodic_path = os.path.join(self.user_dir, "episodic.jsonl")
        self.portrait_path = os.path.join(self.user_dir, "portrait.json")
        self._atoms: list[MemoryAtom] = []
        self._by_id: dict[str, MemoryAtom] = {}
        # use_vector=None 表示自动探测；True 强制要求；False 强制关键词回退。
        if use_vector is False:
            self._vector = None
        else:
            self._vector = _try_vector_backend()
            if use_vector is True and self._vector is None:
                raise RuntimeError("要求向量后端，但 sentence-transformers / numpy 不可用")
        self.backend = "vector" if self._vector is not None else "keyword"

    # -- 情节层入库 --------------------------------------------------------- #
    def ingest(self, records: Iterable[dict]) -> dict:
        """规范化 -> 低置信过滤 -> 去重 -> 落盘 + 建索引。返回统计。"""
        seen = set()
        kept: list[MemoryAtom] = []
        n_in = n_lowconf = n_dup = 0
        for idx, rec in enumerate(records):
            n_in += 1
            atom = MemoryAtom.from_record(rec, idx)
            if atom.confidence < self.min_confidence:
                n_lowconf += 1
                continue
            if atom.mem_id in seen:
                n_dup += 1
                continue
            seen.add(atom.mem_id)
            kept.append(atom)
        self._atoms = kept
        self._by_id = {a.mem_id: a for a in kept}
        with open(self.episodic_path, "w", encoding="utf-8") as f:
            for a in kept:
                f.write(json.dumps(a.to_json(), ensure_ascii=False) + "\n")
        if self._vector is not None:
            self._vector.build(kept)
        return {
            "n_in": n_in,
            "n_kept": len(kept),
            "n_lowconf_dropped": n_lowconf,
            "n_dup_dropped": n_dup,
            "backend": self.backend,
        }

    def load(self) -> "MemoryStore":
        """从已有 episodic.jsonl 重建内存索引（供检索复用）。"""
        atoms: list[MemoryAtom] = []
        if os.path.isfile(self.episodic_path):
            with open(self.episodic_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        atoms.append(MemoryAtom(**json.loads(line)))
        self._atoms = atoms
        self._by_id = {a.mem_id: a for a in atoms}
        if self._vector is not None:
            self._vector.build(atoms)
        return self

    # -- 语义层画像 --------------------------------------------------------- #
    def build_portrait(self) -> dict:
        """聚合该用户全部已入库记忆，得到主导情绪与简单画像标签。"""
        # 置信度加权计票
        weighted: dict[str, float] = {}
        for a in self._atoms:
            if a.emotion_class:
                weighted[a.emotion_class] = weighted.get(a.emotion_class, 0.0) + max(a.confidence, 0.0)
        ranked = sorted(weighted.items(), key=lambda kv: (-kv[1], kv[0]))
        total = sum(weighted.values()) or 1.0
        dominant = [[lbl, round(w / total, 3)] for lbl, w in ranked]
        portrait = {
            "user_id": self.user_id,
            "dominant_emotions": dominant,
            "traits": _traits_from_dominant(ranked),
            "n_mem": len(self._atoms),
            "updated_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(self.portrait_path, "w", encoding="utf-8") as f:
            json.dump(portrait, f, ensure_ascii=False, indent=2)
        return portrait

    def load_portrait(self) -> Optional[dict]:
        if os.path.isfile(self.portrait_path):
            with open(self.portrait_path, encoding="utf-8") as f:
                return json.load(f)
        return None

    # -- 情节层检索 --------------------------------------------------------- #
    def search_episodes(self, query_text: str, top_k: int = 3) -> list[tuple[MemoryAtom, float]]:
        if not self._atoms:
            return []
        if self._vector is not None:
            hits = self._vector.search(query_text, top_k)
            return [(self._by_id[mid], score) for mid, score in hits if mid in self._by_id]
        # 关键词回退
        q_tokens = _tokenize(query_text)
        q_class = map_text_to_label(query_text)
        scored = [(a, _keyword_score(q_tokens, a, q_class)) for a in self._atoms]
        scored = [(a, s) for a, s in scored if s > 0.0]
        scored.sort(key=lambda kv: (-kv[1], kv[0].cycle))
        return scored[:top_k]

    # -- 事件层（按需精确回链） -------------------------------------------- #
    def resolve_event(self, mem_id: str, base_dir: str = "") -> Optional[dict]:
        """把某条记忆回链到原始离线 JSON；找不到文件则返回记忆自身的 source_ref 信息。"""
        atom = self._by_id.get(mem_id)
        if atom is None:
            return None
        ref = atom.source_ref
        path = ref if (not base_dir or os.path.isabs(ref)) else os.path.join(base_dir, ref)
        exists = bool(ref) and os.path.isfile(path)
        out = {"mem_id": mem_id, "source_ref": ref, "resolved_path": path, "exists": exists}
        if exists:
            try:
                with open(path, encoding="utf-8") as f:
                    out["payload"] = json.load(f)
            except Exception:
                out["payload"] = None
        return out

    # -- 组装注入 prompt 的上下文 ------------------------------------------ #
    def retrieve(self, query_text: str, top_k: int = 3, base_dir: str = "") -> dict:
        """三层联合检索：画像 + 情节 Top-k + 命中项的事件回链。"""
        portrait = self.load_portrait() or self.build_portrait()
        episodes = self.search_episodes(query_text, top_k=top_k)
        events = []
        for atom, _score in episodes:
            ev = self.resolve_event(atom.mem_id, base_dir=base_dir)
            if ev is not None:
                events.append(ev)
        return {
            "portrait": portrait,
            "episodes": [
                {"mem_id": a.mem_id, "cycle": a.cycle, "emotion_class": a.emotion_class,
                 "emotion_text": a.emotion_text, "score": round(s, 4)}
                for a, s in episodes
            ],
            "events": events,
            "backend": self.backend,
        }

    def build_prompt_context(self, query_text: str, top_k: int = 3) -> str:
        """把检索结果拼成注入 MLLM 的记忆上下文文本（用于 prompt 预算断言）。"""
        r = self.retrieve(query_text, top_k=top_k)
        lines = ["[长期画像]"]
        dom = ", ".join(f"{lbl}:{w}" for lbl, w in (r["portrait"].get("dominant_emotions") or [])[:3])
        lines.append(f"主导情绪: {dom or '未知'}")
        traits = "、".join(r["portrait"].get("traits") or [])
        if traits:
            lines.append(f"画像标签: {traits}")
        lines.append("[相关情节]")
        for ep in r["episodes"]:
            lines.append(f"- (c{ep['cycle']}/{ep['emotion_class']}) {ep['emotion_text']}")
        return "\n".join(lines)


def _traits_from_dominant(ranked: list[tuple[str, float]]) -> list[str]:
    """从主导情绪派生少量可读画像标签（规则映射，便于冒烟断言）。"""
    rule = {
        "fearful": "易紧张",
        "sad": "情绪偏低落",
        "angry": "易激动",
        "happy": "情绪积极",
        "neutral": "情绪平稳",
        "disgust": "对刺激敏感",
        "surprised": "反应外显",
    }
    traits: list[str] = []
    for lbl, _w in ranked[:2]:
        t = rule.get(lbl)
        if t and t not in traits:
            traits.append(t)
    return traits


def estimate_tokens(text: str) -> int:
    """粗略 token 估计：英文按词、中文按字，足够做预算上限断言。"""
    return len(_tokenize(text))
