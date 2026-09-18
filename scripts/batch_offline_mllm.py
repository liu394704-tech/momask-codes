#!/usr/bin/env python3
"""
递归扫描目录下的视频，逐条调用与 offline_mllm_clip 相同的 MLLM 流程，汇总 CSV + 每条 JSON。

用于大规模标准集离线评测（RAVDESS / 自建库等）。需 OPENAI_API_KEY、OPENAI_BASE_URL。

默认根目录: test_data/external/
默认若同目录存在「与视频同主文件名」的 .wav / .WAV，则自动作为 --audio（可用 --no-auto-audio 关闭）。
CREMA-D 官方目录：VideoFlash/*.flv 与 AudioWAV/*.wav（或 AudioMP3/*.mp3）同主文件名，脚本会自动配对。

用法：
  # 视频 + (可选)音频：RAVDESS 仅 Speech 子集
  python scripts/batch_offline_mllm.py --mode video \\
      --root test_data/external/RAVDESS \\
      --include-regex "Video_Speech_Actor_" \\
      --out-csv experiment/offline_runs/ravdess_speech.csv \\
      --out-json-dir experiment/offline_runs/ravdess_speech_json \\
      --sleep 0.3 --resume

  # 纯音频：Kaggle CREMA-D AudioWAV 子集
  python scripts/batch_offline_mllm.py --mode audio \\
      --root test_data/external/CREMA-D/AudioWAV \\
      --out-csv experiment/offline_runs/cremad_audio.csv \\
      --out-json-dir experiment/offline_runs/cremad_audio_json \\
      --sleep 0.3 --resume
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for _p in (_SCRIPTS, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from emotion_label_map import resolve_default_audio_for_batch  # noqa: E402


def _load_offline_module():
    path = os.path.join(os.path.dirname(__file__), "offline_mllm_clip.py")
    spec = importlib.util.spec_from_file_location("offline_mllm_clip", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 offline_mllm_clip.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _collect_files(
    root: str,
    extensions: frozenset[str],
    include_re: "re.Pattern[str] | None",
    exclude_re: "re.Pattern[str] | None",
) -> list[str]:
    out: list[str] = []
    root = os.path.abspath(root)
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower().lstrip(".")
            if ext not in extensions:
                continue
            full = os.path.join(dirpath, name)
            rel_posix = os.path.relpath(full, root).replace(os.sep, "/")
            if include_re and not include_re.search(rel_posix):
                continue
            if exclude_re and exclude_re.search(rel_posix):
                continue
            out.append(full)
    out.sort()
    return out


def _json_rel_name(video_path: str, root: str) -> str:
    rel = os.path.relpath(video_path, root)
    return rel.replace(os.sep, "__") + ".json"


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch offline MLLM over videos under a directory tree")
    ap.add_argument("--root", default=os.path.join(_ROOT, "test_data", "external"), help="扫描根目录")
    ap.add_argument(
        "--mode",
        default="video",
        choices=("video", "audio"),
        help="video=抽帧+(可选)Whisper+视觉JSON；audio=Whisper+文本JSON（无图像）",
    )
    ap.add_argument(
        "--extensions",
        default="",
        help=(
            "逗号分隔扩展名（不含点）；默认随 --mode 决定: "
            "video=mp4,avi,mov,mkv,flv；audio=wav,mp3,m4a,flac"
        ),
    )
    ap.add_argument(
        "--include-regex",
        default="",
        help="保留：仅当 (相对 --root 的 POSIX 路径) 匹配此正则才处理（不区分大小写）。",
    )
    ap.add_argument(
        "--exclude-regex",
        default="",
        help="排除：匹配此正则的路径被剔除（不区分大小写）。",
    )
    ap.add_argument("--out-csv", required=True, help="汇总 CSV 路径（UTF-8 BOM，便于 Excel）")
    ap.add_argument("--out-json-dir", default="", help="每条样本的 JSON 输出目录；留空则不写单条 JSON")
    ap.add_argument("--num-frames", type=int, default=4)
    ap.add_argument("--max-width", type=int, default=640)
    ap.add_argument("--max-tokens", type=int, default=448)
    ap.add_argument("--limit", type=int, default=0, help="最多处理几条，0 表示不限制")
    ap.add_argument("--sleep", type=float, default=0.0, help="每条成功后额外休眠秒数（限流）")
    ap.add_argument("--resume", action="store_true", help="若 out-json-dir 中已有对应 json 则跳过")
    ap.add_argument("--no-auto-audio", action="store_true", help="不自动配对同目录同名 wav")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit("根目录不存在: %s" % root)

    ext_arg = args.extensions.strip()
    if not ext_arg:
        ext_arg = "wav,mp3,m4a,flac" if args.mode == "audio" else "mp4,avi,mov,mkv,flv"
    exts = frozenset(x.strip().lower().lstrip(".") for x in ext_arg.split(",") if x.strip())

    include_re = re.compile(args.include_regex, re.IGNORECASE) if args.include_regex.strip() else None
    exclude_re = re.compile(args.exclude_regex, re.IGNORECASE) if args.exclude_regex.strip() else None

    files = _collect_files(root, exts, include_re, exclude_re)
    if args.limit > 0:
        files = files[: args.limit]

    if not files:
        raise SystemExit("未找到匹配文件，请检查 --root / --extensions / --include-regex / --exclude-regex")

    offline_mod = _load_offline_module()
    run_one_clip = offline_mod.run_one_clip
    run_one_clip_audio_only = getattr(offline_mod, "run_one_clip_audio_only", None)
    if args.mode == "audio" and run_one_clip_audio_only is None:
        raise SystemExit("offline_mllm_clip.py 缺少 run_one_clip_audio_only，请更新代码")

    json_dir = ""
    if args.out_json_dir.strip():
        json_dir = os.path.abspath(args.out_json_dir)
        os.makedirs(json_dir, exist_ok=True)

    csv_path = args.out_csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(_ROOT, csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    fieldnames = [
        "ts_iso",
        "video_relpath",
        "audio_path",
        "vision_model",
        "whisper_model",
        "num_frames",
        "whisper_s",
        "vision_s",
        "wall_s",
        "user_action",
        "emotion",
        "robot_reaction",
        "error",
    ]

    new_csv = not os.path.isfile(csv_path)
    print(
        "[batch] mode=%s root=%s items=%d csv=%s" % (args.mode, root, len(files), csv_path),
        file=sys.stderr,
    )

    processed = 0
    skipped = 0
    failed = 0
    n = len(files)

    for k, vp in enumerate(files, start=1):
        rel_v = os.path.relpath(vp, _ROOT)
        if args.mode == "audio":
            audio = vp
        else:
            audio = "" if args.no_auto_audio else resolve_default_audio_for_batch(vp)
        jname = _json_rel_name(vp, root)
        jpath = os.path.join(json_dir, jname) if json_dir else ""

        if args.resume and jpath and os.path.isfile(jpath):
            skipped += 1
            continue

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        row_base = {
            "ts_iso": ts,
            "video_relpath": rel_v,
            "audio_path": audio if audio else "",
            "num_frames": 0 if args.mode == "audio" else args.num_frames,
        }
        err = ""
        ua = em = rr = ""
        vm = wm = ""
        whisper_s = vision_s = wall_s = ""

        try:
            if args.mode == "audio":
                data, timings, vm, wm = run_one_clip_audio_only(
                    vp,
                    max_tokens=args.max_tokens,
                )
            else:
                data, timings, vm, wm = run_one_clip(
                    vp,
                    audio,
                    num_frames=args.num_frames,
                    max_width=args.max_width,
                    max_tokens=args.max_tokens,
                )
            whisper_s = "%.6f" % timings["whisper_s"]
            vision_s = "%.6f" % timings["vision_s"]
            wall_s = "%.6f" % timings["wall_s"]
            if isinstance(data, dict):
                ua = str(data.get("user_action", "") or "")
                em = str(data.get("emotion", "") or "")
                rr = str(data.get("robot_reaction", "") or "")
            if jpath:
                with open(jpath, "w", encoding="utf-8") as jf:
                    json.dump(data, jf, ensure_ascii=False, indent=2)
        except SystemExit as e:
            err = str(e) or "SystemExit"
            failed += 1
        except Exception as e:
            err = "%s: %s" % (type(e).__name__, e)
            failed += 1

        row = {
            **row_base,
            "vision_model": vm,
            "whisper_model": wm,
            "whisper_s": whisper_s,
            "vision_s": vision_s,
            "wall_s": wall_s,
            "user_action": ua,
            "emotion": em,
            "robot_reaction": rr,
            "error": err,
        }
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as cf:
            w = csv.DictWriter(cf, fieldnames=fieldnames, extrasaction="ignore")
            if new_csv:
                w.writeheader()
                new_csv = False
            w.writerow(row)

        processed += 1
        print("[%d/%d] %s%s" % (k, n, rel_v, (" ERR:" + err) if err else ""), flush=True)

        if args.sleep > 0 and not err:
            time.sleep(args.sleep)

    print(
        "[batch] done processed=%d skipped_resume=%d failed=%d" % (processed, skipped, failed),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
