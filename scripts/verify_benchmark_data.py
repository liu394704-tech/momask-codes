#!/usr/bin/env python3
"""检查 test_data/external 下 RAVDESS / CREMA-D 是否就绪（数量、扩展名、CREMA 音视频配对）。

  python scripts/verify_benchmark_data.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_SCRIPTS, _ROOT]

from emotion_label_map import (  # noqa: E402
    parse_label_from_filename,
    resolve_default_audio_for_batch,
)


def _count_ext(root: str, exts: set[str]) -> dict[str, int]:
    c = {e: 0 for e in exts}
    if not os.path.isdir(root):
        return c
    for _dp, _dn, names in os.walk(root):
        for name in names:
            e = os.path.splitext(name)[1].lower().lstrip(".")
            if e in c:
                c[e] += 1
    return c


def main() -> None:
    ext_set = {"mp4", "flv", "wav", "mp3"}
    rav = os.path.join(_ROOT, "test_data", "external", "RAVDESS")
    cre = os.path.join(_ROOT, "test_data", "external", "CREMA-D", "CREMA-D-master")

    print("==> RAVDESS:", rav)
    if not os.path.isdir(rav):
        print("    （目录不存在）")
    else:
        rc = _count_ext(rav, ext_set)
        print("    mp4=%d  wav=%d  …" % (rc["mp4"], rc["wav"]))
        n = 0
        for dp, _dn, names in os.walk(rav):
            for name in sorted(names):
                if not name.lower().endswith(".mp4"):
                    continue
                full = os.path.join(dp, name)
                rel = os.path.relpath(full, _ROOT)
                lab = parse_label_from_filename(name, "ravdess")
                au = resolve_default_audio_for_batch(full)
                print("    例:", rel, "| label=%s | audio=%s" % (lab, "有" if au else "无"))
                n += 1
                if n >= 3:
                    break
            if n >= 3:
                break

    cre_root = os.path.join(_ROOT, "test_data", "external", "CREMA-D")
    print("\n==> CREMA-D:", cre_root)
    if not os.path.isdir(cre_root):
        print("    （目录不存在）")
        return
    cc = _count_ext(cre_root, ext_set)
    print("    flv=%d  wav=%d  mp3=%d  mp4=%d" % (cc["flv"], cc["wav"], cc["mp3"], cc["mp4"]))

    candidates = [
        os.path.join(cre_root, "CREMA-D-master", "VideoFlash"),
        os.path.join(cre_root, "VideoFlash"),
    ]
    vf = next((p for p in candidates if os.path.isdir(p)), "")
    if vf:
        paired = 0
        checked = 0
        for name in sorted(os.listdir(vf)):
            if not name.lower().endswith(".flv"):
                continue
            if resolve_default_audio_for_batch(os.path.join(vf, name)):
                paired += 1
            checked += 1
            if checked >= 500:
                break
        print("    VideoFlash 前 500 条中已配对到 AudioWAV/MP3: %d/%d" % (paired, checked))

    aw_candidates = [
        os.path.join(cre_root, "AudioWAV"),
        os.path.join(cre_root, "CREMA-D-master", "AudioWAV"),
    ]
    aw = next((p for p in aw_candidates if os.path.isdir(p)), "")
    if aw and not vf:
        print("    模式: 仅音频（Kaggle 版常见布局）→ 用 --mode audio --root", os.path.relpath(aw, _ROOT))
        wavs = [n for n in os.listdir(aw) if n.lower().endswith(".wav")]
        for name in sorted(wavs)[:3]:
            lab = parse_label_from_filename(name, "cremad")
            sz = os.path.getsize(os.path.join(aw, name))
            print("    例:", name, "| label=%s | size=%d B" % (lab, sz))


if __name__ == "__main__":
    main()
