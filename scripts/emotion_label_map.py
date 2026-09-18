"""标签解析与中→英情绪类别映射，仅用标准库，便于离线评测脚本与单测复用。

支持的数据集（按文件名约定解析 ground truth）:
  - ravdess: 文件名形如 03-01-06-01-02-01-12.mp4，第 3 段（1-indexed）为 emotion 编码。
      01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, 08 surprised
  - cremad: 文件名形如 1001_DFA_ANG_XX.mp4 / .wav，第 3 段为 emotion 缩写。
      ANG/DIS/FEA/HAP/NEU/SAD

统一标签空间（CANON_LABELS）按 7 类常见情绪 + neutral（calm 归并到 neutral）。
"""
from __future__ import annotations

import os
import re
from typing import Optional

CANON_LABELS: tuple[str, ...] = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
)

_RAVDESS_EMOTION_BY_CODE: dict[str, str] = {
    "01": "neutral",
    "02": "neutral",  # calm -> neutral（如需保留 calm，可自行扩 CANON_LABELS）
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

_CREMAD_EMOTION_BY_CODE: dict[str, str] = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def parse_label_from_filename(path: str, dataset: str) -> Optional[str]:
    """从文件名解析 ground-truth 情绪，未识别返回 None。"""
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    ds = dataset.strip().lower()
    if ds == "ravdess":
        parts = stem.split("-")
        if len(parts) >= 7:
            return _RAVDESS_EMOTION_BY_CODE.get(parts[2])
        return None
    if ds in ("cremad", "crema-d", "crema_d"):
        parts = stem.split("_")
        if len(parts) >= 4:
            return _CREMAD_EMOTION_BY_CODE.get(parts[2].upper())
        return None
    return None


def resolve_default_audio_for_batch(video_path: str) -> str:
    """批推理可选音频路径：优先同目录同名 wav；CREMA-D 官方布局为 VideoFlash/*.flv 与 AudioWAV|AudioMP3 同主文件名。"""
    d, base = os.path.split(video_path)
    stem, _ = os.path.splitext(base)
    for cand in ("%s.wav" % stem, "%s.WAV" % stem):
        p = os.path.join(d, cand)
        if os.path.isfile(p):
            return p
    dnorm = d.replace("\\", "/").lower()
    if "videoflash" in dnorm:
        root = os.path.dirname(d)
        for sub, ext in (("AudioWAV", ".wav"), ("AudioMP3", ".mp3")):
            p = os.path.join(root, sub, stem + ext)
            if os.path.isfile(p):
                return p
    return ""


_CN_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "happy",
        (
            "高兴", "开心", "愉悦", "愉快", "喜悦", "欢乐", "兴奋", "得意", "雀跃",
            "笑容", "微笑", "大笑", "欢笑", "欢欣",
        ),
    ),
    (
        "sad",
        (
            "悲伤", "悲", "伤心", "难过", "沮丧", "失落", "哀伤", "忧伤", "落寞",
            "委屈", "压抑", "消沉", "低落", "哭", "泪",
        ),
    ),
    (
        "angry",
        (
            "愤怒", "生气", "恼怒", "暴怒", "愤慨", "怒气", "气愤", "动怒",
        ),
    ),
    (
        "fearful",
        (
            "害怕", "恐惧", "畏惧", "惊恐", "胆怯", "紧张", "担忧", "焦虑", "不安",
            "惶恐",
        ),
    ),
    (
        "disgust",
        ("厌恶", "反感", "讨厌", "嫌弃", "恶心", "鄙夷"),
    ),
    (
        "surprised",
        ("惊讶", "吃惊", "惊奇", "震惊", "诧异", "意外"),
    ),
    (
        "neutral",
        (
            "平静", "冷静", "中性", "平和", "镇定", "镇静", "淡然", "从容", "放松",
            "无明显情绪", "没有明显情绪", "无明显的情绪", "无情绪",
        ),
    ),
)

_EN_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("happy", ("happy", "happiness", "joy", "joyful", "cheerful", "pleased", "smiling", "amused")),
    ("sad", ("sad", "sadness", "sorrow", "downcast", "depressed", "melancholy", "unhappy")),
    ("angry", ("angry", "anger", "furious", "irritated", "annoyed", "mad")),
    ("fearful", ("fear", "fearful", "afraid", "scared", "anxious", "nervous", "tense", "worried")),
    ("disgust", ("disgust", "disgusted", "revolted", "repulsed")),
    ("surprised", ("surprise", "surprised", "shocked", "astonished", "amazed", "startled")),
    ("neutral", ("neutral", "calm", "composed", "expressionless", "no obvious", "no clear")),
)


def _score_text(text: str) -> dict[str, int]:
    if not text:
        return {}
    lowered = text.lower()
    scores: dict[str, int] = {}
    for label, kws in _CN_PATTERNS:
        c = sum(text.count(k) for k in kws)
        if c:
            scores[label] = scores.get(label, 0) + c
    for label, kws in _EN_PATTERNS:
        c = sum(len(re.findall(r"\b%s\b" % re.escape(k), lowered)) for k in kws)
        if c:
            scores[label] = scores.get(label, 0) + c
    return scores


def map_text_to_label(text: str) -> Optional[str]:
    """把模型输出（中/英自由文本）映射到 CANON_LABELS 之一；无法判定返回 None。"""
    scores = _score_text(text or "")
    if not scores:
        return None
    best_label = None
    best_count = -1
    for label in CANON_LABELS:
        c = scores.get(label, 0)
        if c > best_count:
            best_label = label
            best_count = c
    return best_label if best_count > 0 else None


def detect_dataset_from_path(path: str) -> Optional[str]:
    """根据路径片段粗略推断数据集名，便于 --dataset auto。"""
    p = path.replace("\\", "/").lower()
    if "ravdess" in p:
        return "ravdess"
    if "crema" in p:
        return "cremad"
    return None
