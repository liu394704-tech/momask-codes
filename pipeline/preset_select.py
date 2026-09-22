#!/usr/bin/env python3
"""Multimodal phrase scoring plus anti-repeat selection.

Hard rules (stop / short locomotion) run first. Remaining social phrases are
scored from vision + audio, then filtered by recent history so consecutive
motions do not look like a canned loop.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence, Set, Tuple

from .actions import (
    ACTION_ALLOWLIST,
    KEYWORD_TO_ACTION,
    LOCOMOTION_ACTIONS,
    LOCOMOTION_HINTS,
    is_stop_signal,
    locomotion_hint_from_group,
    merge_keyword_transcript,
    normalize_emotion,
    sanitize_action_group,
)
from .preset_catalog import CATALOG
from .preset_phrases import LOCO_PHRASE_IDS, PHRASE_INDEX, PHRASES, Phrase
from .schemas import Decision, Perception


PHRASE_HIST = 8
CLIP_PHRASE_GAP = 3
TAG_HIST = 6
TOP_K = 8
JACCARD_BAN = 0.5

_SMILE = frozenset(("smile", "big_smile", "laugh_combo"))
_FROWN = frozenset((
    "frown", "brow_furrow", "unhappy_furrow", "unhappy_brow_up",
    "brow_inner_up", "brow_furrow",
))
_OPEN = frozenset(("mouth_open", "surprise_combo", "brow_raise"))

_TRANSCRIPT_INTENT: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("停", "不要", "停止", "stop", "enough", "别动"), "stop"),
    (("前进", "往前", "forward"), "loco:go_forward"),
    (("后退", "往后", "back"), "loco:back_fast"),
    (("左转", "turn left", "turn_left"), "loco:turn_left"),
    (("右转", "turn right", "turn_right"), "loco:turn_right"),
    (("你好", "hello", "hi", "hey", "早上好", "晚上好"), "greeting"),
    (("累", "陪", "安慰", "难过", "伤心", "tired", "sad", "comfort", "lonely"), "comfort_request"),
    (("玩", "开心", "高兴", "play", "fun", "happy", "dance"), "play"),
    (("帮", "帮助", "help", "救"), "help"),
)


@dataclass
class PhraseChoice:
    phrase_id: str
    clips: List[str]
    recovery: Optional[str]
    source: str
    score: float = 0.0
    bans: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    laterality: str = "none"
    energy: str = "mid"

    @property
    def action(self) -> str:
        return "+".join(self.clips) if self.clips else ""


def clip_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    a = set(left or ())
    b = set(right or ())
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    return float(len(a & b)) / float(len(a | b))


def _face_actions(perception: Optional[Perception]) -> List[str]:
    if perception is None:
        return []
    actions = list(getattr(perception, "face_actions", None) or [])
    extras = getattr(perception, "extras", None) or {}
    if not actions:
        actions = list(extras.get("face_actions") or [])
    return [str(x) for x in actions]


def _transcript_intent(text: str) -> Optional[str]:
    blob = (text or "").strip().lower()
    if not blob:
        return None
    for keys, intent in _TRANSCRIPT_INTENT:
        for key in keys:
            if key.lower() in blob:
                return intent
    return None


def _locomotion_name(
    keyword: Optional[str],
    transcript: str,
    action_group: Optional[Sequence[str]],
    intent: str,
) -> Optional[str]:
    mapped = KEYWORD_TO_ACTION.get((keyword or "").strip().lower())
    if mapped in LOCOMOTION_ACTIONS or mapped in LOCOMOTION_HINTS:
        return mapped
    parsed = _transcript_intent(transcript)
    if parsed and parsed.startswith("loco:"):
        return parsed.split(":", 1)[1]
    hint = locomotion_hint_from_group(action_group)
    if hint:
        return hint
    if intent in LOCOMOTION_ACTIONS:
        return intent
    return None


def _desired_energy(intensity: str, confidence: float) -> str:
    if intensity == "strong" or confidence >= 0.55:
        return "high"
    if confidence < 0.40:
        return "low"
    return "mid"


def score_phrase(
    phrase: Phrase,
    *,
    emotion: str,
    intent: str,
    energy: str,
    face_actions: Sequence[str],
    hint_clips: Sequence[str],
    phrase_hint: str = "",
) -> float:
    score = 0.0
    if phrase.intent == intent and intent not in ("", "unknown"):
        score += 2.0
    if phrase_hint and phrase.intent == phrase_hint:
        score += 0.8
    if emotion in phrase.emotion_fit:
        score += 1.5
    elif emotion == "neutral" and "idle" in phrase.tags:
        score += 1.0

    hint_set = set(hint_clips or ())
    if hint_set:
        score += 0.4 * len(hint_set & set(phrase.clips))

    if phrase.energy == energy:
        score += 0.6
    elif phrase.energy == "mid" or energy == "mid":
        score += 0.2

    au = set(face_actions or ())
    tags = set(phrase.tags)
    if au & _SMILE and (tags & {"greet", "celebrate"}):
        score += 0.5
    if au & _FROWN and ("comfort" in tags):
        score += 0.5
    if au & _OPEN and ("startle" in tags):
        score += 0.5
    return score


def _tag_penalty(phrase: Phrase, tag_hist: Sequence[Set[str]]) -> float:
    if not tag_hist:
        return 0.0
    tags = set(phrase.tags)
    if not tags:
        return 0.0
    overlap = 0.0
    for prev in tag_hist:
        if not prev:
            continue
        overlap += float(len(tags & prev)) / float(len(tags | prev))
    return 0.35 * (overlap / float(len(tag_hist)))


def _laterality_penalty(phrase: Phrase, last_laterality: Optional[str]) -> float:
    if not last_laterality or last_laterality == "none":
        return 0.0
    if phrase.laterality in ("none",):
        return 0.0
    if phrase.laterality == last_laterality:
        return 0.15
    return 0.0


def _choose_recovery(clips: Sequence[str], rng: random.Random) -> Optional[str]:
    if not clips:
        return "stand_slow"
    if clips[-1] in ("stand", "stand_slow"):
        return None
    return "stand" if rng.random() < 0.5 else "stand_slow"


def _from_phrase(
    phrase: Phrase,
    *,
    source: str,
    score: float,
    bans: List[str],
    rng: random.Random,
) -> PhraseChoice:
    clips = list(phrase.clips)
    return PhraseChoice(
        phrase_id=phrase.id,
        clips=clips,
        recovery=_choose_recovery(clips, rng),
        source=source,
        score=score,
        bans=bans,
        tags=list(phrase.tags),
        laterality=phrase.laterality,
        energy=phrase.energy,
    )


class PhraseSelector:
    """Session-scoped history + multimodal pick."""

    def __init__(
        self,
        phrase_hist: int = PHRASE_HIST,
        clip_gap: int = CLIP_PHRASE_GAP,
        tag_hist: int = TAG_HIST,
        top_k: int = TOP_K,
    ):
        self.phrase_hist_n = int(phrase_hist)
        self.clip_gap = int(clip_gap)
        self.tag_hist_n = int(tag_hist)
        self.top_k = int(top_k)
        self.phrase_hist: Deque[str] = deque(maxlen=self.phrase_hist_n)
        self.clip_rounds: Deque[Tuple[str, ...]] = deque(maxlen=max(self.clip_gap, 4))
        self.tag_hist: Deque[Set[str]] = deque(maxlen=self.tag_hist_n)
        self.last_laterality: Optional[str] = None
        self.last_clips: Tuple[str, ...] = ()

    def reset(self) -> None:
        self.phrase_hist.clear()
        self.clip_rounds.clear()
        self.tag_hist.clear()
        self.last_laterality = None
        self.last_clips = ()

    def commit(self, choice: PhraseChoice) -> None:
        if not choice or not choice.clips:
            return
        self.phrase_hist.append(choice.phrase_id)
        clips = tuple(choice.clips)
        self.clip_rounds.append(clips)
        self.last_clips = clips
        self.tag_hist.append(set(choice.tags))
        self.last_laterality = choice.laterality

    def _recent_clips(self) -> Set[str]:
        banned: Set[str] = set()
        for clips in list(self.clip_rounds)[-self.clip_gap:]:
            banned.update(clips)
        return banned

    def _hard_bans(self, phrase: Phrase) -> List[str]:
        reasons: List[str] = []
        if phrase.id in self.phrase_hist:
            reasons.append("phrase_repeat")
        if self.last_clips and phrase.clips and phrase.clips[0] == self.last_clips[-1]:
            reasons.append("seam")
        if self.last_clips and clip_jaccard(phrase.clips, self.last_clips) >= JACCARD_BAN:
            reasons.append("jaccard")
        recent = self._recent_clips()
        if recent and set(phrase.clips) & recent:
            reasons.append("clip_gap")
        return reasons

    def _pick_loco(self, loco_name: str, rng: random.Random) -> PhraseChoice:
        ids = LOCO_PHRASE_IDS.get(loco_name) or ()
        candidates = [PHRASE_INDEX[i] for i in ids if i in PHRASE_INDEX]
        open_ones = [p for p in candidates if not self._hard_bans(p)]
        pool = open_ones or candidates
        if not pool:
            clips = list(LOCOMOTION_HINTS.get(loco_name) or (loco_name,))
            clips = sanitize_action_group(clips) or ["stand"]
            return PhraseChoice(
                phrase_id="loco_fallback_%s" % loco_name,
                clips=clips[:1],
                recovery=_choose_recovery(clips[:1], rng),
                source="locomotion_phrase",
                tags=["locomote"],
            )
        phrase = rng.choice(pool)
        return _from_phrase(phrase, source="locomotion_phrase", score=3.0, bans=[], rng=rng)

    def _atomic_fallback(self, emotion: str, intent: str, rng: random.Random) -> PhraseChoice:
        recent = self._recent_clips()
        last = self.last_clips[-1] if self.last_clips else None
        prefer_tags = {
            "greeting": {"greet", "celebrate"},
            "play": {"celebrate", "greet"},
            "comfort_request": {"comfort"},
            "help": {"greet"},
            "unknown": {"idle", "greet", "torso"},
        }.get(intent, {"greet", "idle"})
        if emotion == "surprised":
            prefer_tags = prefer_tags | {"startle", "torso"}
        if emotion == "unhappy":
            prefer_tags = prefer_tags | {"comfort"}
        ranked: List[Tuple[int, str]] = []
        for name, clip in CATALOG.items():
            if clip.recover or name not in ACTION_ALLOWLIST:
                continue
            if name in recent or name == last:
                continue
            hit = 1 if set(clip.tags) & prefer_tags else 0
            ranked.append((hit, name))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        if not ranked:
            return PhraseChoice(
                phrase_id="fallback_stand_slow",
                clips=["stand_slow"],
                recovery=None,
                source="phrase_stand",
                bans=["exhausted"],
                tags=["recover", "idle"],
                energy="low",
            )
        name = ranked[0][1]
        clips = [name]
        return PhraseChoice(
            phrase_id="fallback_%s" % name,
            clips=clips,
            recovery=_choose_recovery(clips, rng),
            source="phrase_fallback",
            bans=["top_k_banned"],
            tags=list(CATALOG[name].tags),
            laterality=CATALOG[name].laterality,
            energy=CATALOG[name].energy,
        )

    def select(
        self,
        perception: Optional[Perception],
        decision: Optional[Decision],
        keyword: Optional[str] = None,
        intensity: str = "mild",
        rng: Optional[random.Random] = None,
    ) -> PhraseChoice:
        rng = rng or random.Random()
        extras = {}
        if decision is not None:
            extras.update(getattr(decision, "extras", None) or {})
        if perception is not None:
            extras.update(getattr(perception, "extras", None) or {})
        keyword = keyword or extras.get("keyword")
        transcript = ""
        if perception is not None:
            transcript = merge_keyword_transcript(keyword, perception.transcript or "")
        elif decision is not None:
            transcript = merge_keyword_transcript(keyword, "")

        intent = (decision.intent if decision else "unknown") or "unknown"
        emotion = normalize_emotion(
            (decision.emotion if decision else None)
            or (perception.vision_emotion if perception else None)
        )
        confidence = float(decision.confidence if decision else 0.0)
        if perception is not None and perception.vision_conf:
            confidence = max(confidence, float(perception.vision_conf))
        face_actions = _face_actions(perception)
        hint_clips = sanitize_action_group(decision.action_group if decision else None)
        phrase_hint = str(extras.get("phrase_hint") or "")
        parsed = _transcript_intent(transcript)
        if parsed == "stop" or is_stop_signal(keyword=keyword, transcript=transcript, intent=intent):
            self.reset()
            return PhraseChoice(
                phrase_id="stop_stand",
                clips=["stand"],
                recovery=None,
                source="stop_signal",
                score=10.0,
                tags=["recover"],
                energy="low",
            )
        if parsed and parsed.startswith("loco:"):
            return self._pick_loco(parsed.split(":", 1)[1], rng)
        loco = _locomotion_name(keyword, transcript, hint_clips, intent)
        if loco:
            return self._pick_loco(loco, rng)
        if parsed in ("greeting", "comfort_request", "play", "help"):
            intent = parsed
        if not phrase_hint:
            phrase_hint = intent

        energy = _desired_energy(intensity, confidence)
        loco_ids = {"stop_stand"}
        for ids in LOCO_PHRASE_IDS.values():
            loco_ids.update(ids)
        social = [p for p in PHRASES if p.id not in loco_ids]
        scored: List[Tuple[float, Phrase, List[str]]] = []
        for phrase in social:
            bans = self._hard_bans(phrase)
            raw = score_phrase(
                phrase,
                emotion=emotion,
                intent=intent,
                energy=energy,
                face_actions=face_actions,
                hint_clips=hint_clips,
                phrase_hint=phrase_hint,
            )
            scored.append((raw, phrase, bans))

        open_scored = [(raw, phrase, bans) for raw, phrase, bans in scored if not bans]
        open_scored.sort(key=lambda item: (-item[0], item[1].id))
        top = open_scored[: self.top_k]
        if top:
            ranked = []
            for raw, phrase, bans in top:
                penalty = _tag_penalty(phrase, list(self.tag_hist))
                penalty += _laterality_penalty(phrase, self.last_laterality)
                ranked.append((raw - penalty, raw, phrase, bans))
            ranked.sort(key=lambda item: (-item[0], item[2].id))
            # Do not always take raw-max: penalties reorder, then jitter among close scores.
            best = ranked[0][0]
            close = [item for item in ranked if best - item[0] <= 0.35]
            _, raw, phrase, bans = rng.choice(close)
            return _from_phrase(
                phrase, source="phrase_select", score=raw, bans=bans, rng=rng
            )

        return self._atomic_fallback(emotion, intent, rng)


_DEFAULT_SELECTOR: Optional[PhraseSelector] = None


def get_default_selector() -> PhraseSelector:
    global _DEFAULT_SELECTOR
    if _DEFAULT_SELECTOR is None:
        _DEFAULT_SELECTOR = PhraseSelector()
    return _DEFAULT_SELECTOR
