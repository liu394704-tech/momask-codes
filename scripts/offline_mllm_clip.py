#!/usr/bin/env python3
"""
从本地视频（+ 可选 wav）抽帧，调用与 live_emotion_monitor 相同的多模态 MLLM，输出 JSON。

用于标准测试集离线回放（无需摄像头）。需已设置 OPENAI_API_KEY、OPENAI_BASE_URL。

示例:
  export OPENAI_API_KEY=...
  export OPENAI_BASE_URL=https://.../v1
  python scripts/offline_mllm_clip.py --video test_data/synthetic/synthetic_rgb.mp4 \\
      --num-frames 4 --max-width 640 --out-json /tmp/out.json

可选语音（经 Whisper 或本机规则与 live_emotion_monitor 一致的环境变量）:
  python scripts/offline_mllm_clip.py --video ... --audio test_data/synthetic/synthetic.wav
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from typing import Any

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import cv2  # noqa: E402

import live_emotion_monitor as lem  # noqa: E402


def _sample_frame_paths(video_path: str, num_frames: int, max_width: int, temp_dir: str) -> list[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit("无法打开视频: %s" % video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    paths: list[str] = []
    indices: list[int] = []
    if total <= 0 or num_frames <= 0:
        raise SystemExit("无效帧数或无法读取总帧数")
    for k in range(num_frames):
        idx = int((k + 0.5) * total / num_frames) if total >= num_frames else min(k, total - 1)
        indices.append(max(0, min(idx, total - 1)))
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        small = lem.resize_frame_for_vision(frame, max_width)
        out = os.path.join(temp_dir, "offline_frame_%d.jpg" % i)
        cv2.imwrite(out, small, [cv2.IMWRITE_JPEG_QUALITY, 78])
        paths.append(os.path.abspath(out))
    cap.release()
    if len(paths) < num_frames:
        raise SystemExit("抽帧不足 %d 张，请检查视频是否过短。" % num_frames)
    return paths[:num_frames]


def run_one_clip(
    video_path: str,
    audio_path: str = "",
    *,
    num_frames: int = 4,
    max_width: int = 640,
    max_tokens: int = 448,
    temperature: float = 0.35,
) -> tuple[dict[str, Any], dict[str, float], str, str]:
    """Whisper（可选）+ 抽帧 + MLLM。返回 (json 字典, 耗时秒, vision_model, whisper_model)。"""
    t_all0 = time.perf_counter()
    model = lem._env("OPENAI_MODEL", lem.DEFAULT_MLLM_MODEL)
    whisper_model = lem._env("WHISPER_MODEL", "whisper-1")
    client = lem.make_openai_client()

    user_speech = "（本轮无语音输入）"
    t_wh0 = time.perf_counter()
    if audio_path.strip() and os.path.isfile(audio_path):
        try:
            user_speech = lem.transcribe_audio(client, audio_path, whisper_model)
        except Exception as e:
            print("[offline] Whisper 失败，按无语音继续:", e, file=sys.stderr)
            user_speech = "（语音识别失败，仅根据画面分析）"
    whisper_s = time.perf_counter() - t_wh0

    with tempfile.TemporaryDirectory(prefix="offline_mllm_") as td:
        frame_paths = _sample_frame_paths(
            video_path,
            num_frames=max(1, num_frames),
            max_width=max_width,
            temp_dir=td,
        )
        vd = lem._env("LIVE_MONITOR_VISION_DETAIL", "").strip().lower()
        vd_arg = vd if vd in ("low", "high", "auto") else None
        t_vis0 = time.perf_counter()
        data = lem.run_mllm_vision_json(
            client,
            model=model,
            user_speech=user_speech,
            frame_paths=frame_paths,
            vision_detail=vd_arg,
            max_tokens=max(0, max_tokens),
            temperature=temperature,
        )
        vision_s = time.perf_counter() - t_vis0

    wall_s = time.perf_counter() - t_all0
    timings = {"whisper_s": whisper_s, "vision_s": vision_s, "wall_s": wall_s}
    return data, timings, model, whisper_model


_AUDIO_ONLY_PROMPT = (
    "You are the affect module of a humanoid robot that will socially interact with the speaker."
    " You only have access to the AUDIO of the user; no camera frames are provided.\n"
    'Transcribed speech from the user (may be empty or a placeholder if no audio content): "%s"\n\n'
    "Output ONLY valid JSON with exactly these keys (no markdown):\n"
    '"user_action": string, in Chinese — what the speaker seems to be doing or expressing based ONLY on the speech content, prosody cues and any non-verbal sounds you can infer; if there is no clear evidence, write "未知".\n'
    '"emotion": string, in Chinese — infer affect with HIGH sensitivity to vocal prosody (tone, energy, tempo, tremor, sighs, breath) and to the lexical content. Prefer a specific label (e.g. 略带紧张、隐忍的不耐烦、克制的期待、微妙的尴尬) over generic fillers like 平静、放松、中性 unless the voice truly shows calm neutrality. Keep it as one short phrase under 28 Chinese characters.\n'
    '"robot_reaction": string, in English — ONE short HumanML3D-style motion description of how the ROBOT should move as social feedback to this speaker (e.g. open posture with slow nod to acknowledge tension; gentle side-step to give space). Describe ONLY the robot\'s own body from the robot\'s viewpoint; do not imitate the user.\n'
)


def _run_text_only_mllm_json(
    client,
    *,
    model: str,
    user_speech: str,
    max_tokens: int,
    temperature: float,
) -> dict:
    prompt = _AUDIO_ONLY_PROMPT % user_speech.replace('"', "'")[:800]
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=temperature,
        max_tokens=max_tokens,
    )
    result_str = response.choices[0].message.content or "{}"
    try:
        return json.loads(result_str)
    except json.JSONDecodeError:
        s = result_str.strip()
        if "```" in s:
            s = s.split("```", 1)[-1].split("```", 1)[0]
            if s.startswith("json"):
                s = s[4:].strip()
        return json.loads(s)


def run_one_clip_audio_only(
    audio_path: str,
    *,
    max_tokens: int = 384,
    temperature: float = 0.35,
) -> tuple[dict[str, Any], dict[str, float], str, str]:
    """纯音频路径：Whisper 转写 + 文本-only LLM 出同样 JSON。"""
    t_all0 = time.perf_counter()
    if not audio_path or not os.path.isfile(audio_path):
        raise SystemExit("audio 不存在: %s" % audio_path)
    model = lem._env("OPENAI_MODEL", lem.DEFAULT_MLLM_MODEL)
    whisper_model = lem._env("WHISPER_MODEL", "whisper-1")
    client = lem.make_openai_client()

    t_wh0 = time.perf_counter()
    try:
        user_speech = lem.transcribe_audio(client, audio_path, whisper_model)
    except Exception as e:
        print("[offline-audio] Whisper 失败:", e, file=sys.stderr)
        user_speech = "（语音识别失败）"
    whisper_s = time.perf_counter() - t_wh0

    t_vis0 = time.perf_counter()
    data = _run_text_only_mllm_json(
        client,
        model=model,
        user_speech=user_speech,
        max_tokens=max(0, max_tokens),
        temperature=temperature,
    )
    vision_s = time.perf_counter() - t_vis0
    wall_s = time.perf_counter() - t_all0
    timings = {"whisper_s": whisper_s, "vision_s": vision_s, "wall_s": wall_s}
    return data, timings, model, whisper_model


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline multimodal MLLM on a local video clip")
    ap.add_argument("--video", required=True, help="本地 mp4/avi 等")
    ap.add_argument("--audio", default="", help="可选 wav，用于 Whisper 后并入 prompt")
    ap.add_argument("--num-frames", type=int, default=4)
    ap.add_argument("--max-width", type=int, default=640)
    ap.add_argument("--max-tokens", type=int, default=448)
    ap.add_argument("--out-json", default="", help="若设则写入该路径")
    args = ap.parse_args()

    data, _timings, _model, _wm = run_one_clip(
        args.video,
        args.audio,
        num_frames=args.num_frames,
        max_width=args.max_width,
        max_tokens=args.max_tokens,
    )

    text = json.dumps(data, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(text)
        print("\n[offline] 已写入", args.out_json, file=sys.stderr)


if __name__ == "__main__":
    main()
