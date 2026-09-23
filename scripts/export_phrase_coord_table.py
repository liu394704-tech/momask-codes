#!/usr/bin/env python3
"""Write the 778-phrase emotion table and the 33-clip joint keyframes.

Coordinates are the simulated TonyPi layer (stand pulse 500, millimetres).
Factory .d6a files still drive the servos.
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.preset_phrases import PHRASES
from pipeline.tonypi_body import PULSE_CENTER
from pipeline.tonypi_coords import CLIP_COORDS, simulate_clip

INTENT_ZH = {
    "greeting": "打招呼",
    "comfort_request": "安慰",
    "play": "玩耍",
    "unknown": "其他",
    "help": "求助",
    "stop": "停止",
}
EMOTION_ZH = {
    "happy": "高兴",
    "neutral": "中性",
    "surprised": "惊讶",
    "unhappy": "不高兴",
}
SERVO_ZH = {
    1: "右脚横滚",
    2: "右踝俯仰",
    3: "右膝",
    4: "右髋俯仰",
    5: "右髋横滚",
    6: "右肩",
    7: "右臂旋转",
    8: "右肘",
    9: "左脚横滚",
    10: "左踝俯仰",
    11: "左膝",
    12: "左髋俯仰",
    13: "左髋横滚",
    14: "左肩",
    15: "左臂旋转",
    16: "左肘",
}


def _labels(keys, table):
    return "/".join("%s/%s" % (key, table.get(key, key)) for key in keys)


def _xyz(pose):
    values = []
    for key in ("head", "left_hand", "right_hand", "com"):
        values.extend(round(v, 1) for v in pose.points[key])
    return values


def _pose_cells(pose):
    return list(pose.pulses) + _xyz(pose) + [
        round(pose.root[0], 1),
        round(pose.root[1], 1),
        round(math.degrees(pose.yaw_rad), 1),
    ]


def _peak(name, cache):
    if name not in cache:
        poses = simulate_clip(name)
        cache[name] = max(
            poses,
            key=lambda pose: (sum(abs(v - PULSE_CENTER) for v in pose.pulses), pose.time_ms),
        )
    return cache[name]


def export(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    servo_cols = ["舵机%d_%s" % (i, SERVO_ZH[i]) for i in range(1, 17)]
    xyz_cols = [
        "%s_%s_mm" % (label, axis)
        for label in ("头", "左手", "右手", "质心")
        for axis in ("x", "y", "z")
    ]
    phrase_header = [
        "短语编号", "适用情绪", "意图", "力度", "片段数", "动作序列", "峰值所在片段",
        *servo_cols, *xyz_cols, "根_x_mm", "根_y_mm", "偏航_度",
    ]
    for index in (1, 2, 3):
        phrase_header += ["片段%d" % index]
        phrase_header += ["片段%d_%s" % (index, col) for col in servo_cols]
        phrase_header += ["片段%d_%s" % (index, col) for col in xyz_cols]

    cache = {}
    phrase_path = out_dir / "phrase_emotion_coords.csv"
    with phrase_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(phrase_header)
        for phrase in PHRASES:
            poses = [(name, _peak(name, cache)) for name in phrase.clips]
            name, pose = max(
                poses,
                key=lambda item: sum(abs(v - PULSE_CENTER) for v in item[1].pulses),
            )
            row = [
                phrase.id,
                _labels(phrase.emotion_fit, EMOTION_ZH),
                "%s/%s" % (phrase.intent, INTENT_ZH.get(phrase.intent, phrase.intent)),
                phrase.energy,
                len(phrase.clips),
                " > ".join(phrase.clips),
                name,
                *_pose_cells(pose),
            ]
            for index in range(3):
                if index < len(poses):
                    clip_name, clip_pose = poses[index]
                    row += [clip_name, *list(clip_pose.pulses), *_xyz(clip_pose)]
                else:
                    row += [""] * (1 + 16 + 12)
            writer.writerow(row)

    clip_emotions = defaultdict(set)
    for phrase in PHRASES:
        for name in phrase.clips:
            clip_emotions[name].update(phrase.emotion_fit)
    frame_path = out_dir / "clip_keyframes.csv"
    with frame_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "片段", "该片段出现在哪些情绪", "帧序号", "时长_ms",
            *servo_cols, *xyz_cols, "根_x_mm", "根_y_mm", "偏航_度",
        ])
        for name in CLIP_COORDS:
            emotions = _labels(sorted(clip_emotions.get(name, ())), EMOTION_ZH)
            for index, pose in enumerate(simulate_clip(name), start=1):
                writer.writerow([name, emotions, index, pose.time_ms, *_pose_cells(pose)])


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    export(root / "tables")
    print("wrote", root / "tables")
