#!/usr/bin/env python3
"""
摄像头（仅视觉）→ 智谱多模态 JSON 规划 → 调用 MoMask gen_t2m 生成动作。

长时连贯：维护最近若干轮「情绪 + 英文动作句」摘要，随每帧请求发给 MLLM，便于叙事延续。

环境变量:
  ZHIPU_API_KEY 或 ZHIPUAI_API_KEY  （必填）
  ZHIPU_VISION_MODEL=glm-4v-flash   （推荐；纯文本模型无法理解图片）

示例:
  export ZHIPU_API_KEY=...
  cd /path/to/momask-codes
  python scripts/live_vision_momask.py --gpu_id 0 --ext live_vis --interval 10 --dry-run

首次建议 --dry-run 只看 JSON 不跑 gen_t2m；确认无误后去掉 --dry-run。

macOS 注意: OpenCV 的 imshow/waitKey 必须在主线程调用；本脚本主线程跑摄像头预览，后台线程跑 VLM + gen_t2m。

无摄像头（SSH 云端）: 使用 --sample-image /path/to.jpg 从磁盘读一张图代替摄像头（可多帧复制为同一张）。
"""
from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
import threading
import time
from typing import List

import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from llm_bridge.json_utils import extract_json_object, validate_vision_momask_plan
from llm_bridge.llm_client import call_zhipu
from llm_bridge.schema import vision_momask_system_prompt


def _jpeg_b64(frame) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def run_camera_on_main_thread(
    buffer: List,
    lock: threading.Lock,
    frame_interval: float,
    running: threading.Event,
    camera_index: int,
    no_preview: bool,
) -> None:
    """Must run on the main thread (required on macOS for cv2.imshow)."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[camera] cannot open device %d" % camera_index, flush=True)
        running.clear()
        return
    last_t = time.time()
    win = "Live Vision → MoMask (q=quit)"
    if not no_preview:
        print("[camera] window open (main thread). Press 'q' to quit.", flush=True)
    else:
        print("[camera] no-preview mode: grab frames only; Ctrl+C to stop.", flush=True)
    while running.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        frame_flipped = cv2.flip(frame, 1)
        if not no_preview:
            cv2.imshow(win, frame_flipped)
        now = time.time()
        if now - last_t >= frame_interval:
            with lock:
                buffer.append(frame_flipped.copy())
                if len(buffer) > 16:
                    buffer[:] = buffer[-16:]
            last_t = now
        if no_preview:
            time.sleep(0.001)
        else:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if not no_preview:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    running.clear()


def build_user_message(lang: str, session_lines: List[str], num_frames: int) -> str:
    mem = "\n".join(session_lines) if session_lines else "(none yet)"
    if lang.startswith("zh"):
        return (
            "以下为会话记忆（按时间从旧到新）：\n%s\n\n"
            "你将看到 %d 张按时间顺序排列的截图（最早→最晚）。"
            "请根据**最新画面**与记忆，输出下一 JSON 规划。"
        ) % (mem, num_frames)
    return (
        "Session memory (oldest to newest):\n%s\n\n"
        "You will see %d frames in chronological order. "
        "Use the **latest** frame as primary evidence; keep continuity with memory when appropriate."
    ) % (mem, num_frames)


def worker_loop(
    *,
    gpu_id: int,
    ext: str,
    interval: float,
    warmup: float,
    num_frames: int,
    context_turns: int,
    lang: str,
    dry_run: bool,
    overwrite: bool,
    once: bool,
    vision_model: str,
    running: threading.Event,
    buffer: List,
    lock: threading.Lock,
) -> None:
    session_lines: List[str] = []
    step = 0
    if warmup > 0:
        print("[worker] warmup %.1fs ..." % warmup, flush=True)
        time.sleep(warmup)

    while running.is_set():
        t0 = time.time()
        with lock:
            frames = list(buffer)
        if len(frames) < num_frames:
            print(
                "[worker] need %d frame(s), have %d — wait ..." % (num_frames, len(frames)),
                flush=True,
            )
            time.sleep(0.5)
            continue

        picked = frames[-num_frames:]
        images_b64 = [_jpeg_b64(f) for f in picked]

        system = vision_momask_system_prompt(lang)
        user = build_user_message(lang, session_lines, num_frames)

        print("[worker] calling VLM (%s) step %d ..." % (vision_model, step), flush=True)
        try:
            raw = call_zhipu(
                system,
                user,
                model=vision_model,
                images_base64=images_b64,
            )
        except Exception as e:
            print("[worker] LLM error:", e, flush=True)
            time.sleep(2.0)
            continue

        print("--- LLM raw ---\n", raw[:3000], "\n--- end ---\n", flush=True)

        try:
            data = extract_json_object(raw)
            text_prompt, motion_length, emotion = validate_vision_momask_plan(data)
        except Exception as e:
            print("[worker] JSON parse failed:", e, flush=True)
            time.sleep(2.0)
            continue

        print(
            "[plan] emotion=%s | motion_length=%s\n       text_prompt=%s"
            % (emotion, motion_length, text_prompt),
            flush=True,
        )

        one_line = "[%s] %s" % (emotion, text_prompt[:220])
        session_lines.append(one_line)
        if len(session_lines) > context_turns:
            session_lines[:] = session_lines[-context_turns:]

        if dry_run:
            print("[worker] dry-run: skip gen_t2m", flush=True)
        else:
            gen_py = os.path.join(ROOT, "gen_t2m.py")
            cmd = [
                sys.executable,
                gen_py,
                "--gpu_id",
                str(gpu_id),
                "--ext",
                ext,
                "--text_prompt",
                text_prompt,
            ]
            if overwrite:
                cmd.append("--overwrite")
            if motion_length > 0:
                cmd.extend(["--motion_length", str(motion_length)])
            print("[worker] running gen_t2m ...", flush=True)
            r = subprocess.run(cmd, cwd=ROOT)
            if r.returncode != 0:
                print("[worker] gen_t2m exited with", r.returncode, flush=True)

        step += 1
        if once:
            running.clear()
            break

        elapsed = time.time() - t0
        sleep_s = max(0.5, interval - elapsed)
        time.sleep(sleep_s)


def main() -> None:
    ap = argparse.ArgumentParser(description="Webcam vision → Zhipu → MoMask gen_t2m")
    ap.add_argument("--gpu_id", type=int, default=-1)
    ap.add_argument("--ext", default="live_vision_momask")
    ap.add_argument("--interval", type=float, default=12.0, help="seconds between VLM+gen cycles")
    ap.add_argument("--warmup", type=float, default=3.0, help="seconds before first cycle")
    ap.add_argument("--frame-interval", type=float, default=0.6, dest="frame_interval", help="camera buffer cadence")
    ap.add_argument("--num-frames", type=int, default=2, help="send last N frames to VLM (>=1)")
    ap.add_argument("--context-turns", type=int, default=6, help="how many past plans to keep in prompt")
    ap.add_argument("--lang", default="zh", choices=["zh", "en"])
    ap.add_argument("--dry-run", action="store_true", help="only VLM + JSON, no gen_t2m")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--once", action="store_true", help="single cycle then exit")
    ap.add_argument(
        "--vision-model",
        default=os.environ.get("ZHIPU_VISION_MODEL", "glm-4v-flash").strip(),
        help="Vision MLLM (must support images)",
    )
    ap.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera device index",
    )
    ap.add_argument(
        "--no-preview",
        action="store_true",
        help="do not open cv2 window (no imshow); macOS headless or if GUI still crashes",
    )
    ap.add_argument(
        "--sample-image",
        default="",
        help="path to a JPEG/PNG on disk: use as frame source instead of /dev/video (for headless servers)",
    )
    args = ap.parse_args()

    if not os.environ.get("ZHIPU_API_KEY", "").strip() and not os.environ.get(
        "ZHIPUAI_API_KEY", ""
    ).strip():
        print("Set ZHIPU_API_KEY or ZHIPUAI_API_KEY", file=sys.stderr)
        sys.exit(2)

    nf = max(1, args.num_frames)
    buf: List = []
    lk = threading.Lock()
    running = threading.Event()
    running.set()

    sample_path = (args.sample_image or "").strip()
    if sample_path:
        path = os.path.abspath(os.path.expanduser(sample_path))
        if not os.path.isfile(path):
            print("[sample-image] file not found: %s" % path, file=sys.stderr)
            sys.exit(2)
        img = cv2.imread(path)
        if img is None:
            print("[sample-image] cv2.imread failed (not an image?): %s" % path, file=sys.stderr)
            sys.exit(2)
        # 与摄像头预览一致：镜像（若不想镜像可改）
        img = cv2.flip(img, 1)
        with lk:
            buf.clear()
            for _ in range(nf):
                buf.append(img.copy())
        print(
            "[sample-image] loaded %s (%dx%d), filled %d frame slot(s); no webcam."
            % (path, img.shape[1], img.shape[0], nf),
            flush=True,
        )

    wk_t = threading.Thread(
        target=worker_loop,
        kwargs=dict(
            gpu_id=args.gpu_id,
            ext=args.ext,
            interval=args.interval,
            warmup=args.warmup,
            num_frames=nf,
            context_turns=max(1, args.context_turns),
            lang=args.lang,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            once=args.once,
            vision_model=args.vision_model,
            running=running,
            buffer=buf,
            lock=lk,
        ),
        daemon=True,
    )
    wk_t.start()
    if sample_path:
        # 云端无 /dev/video：主线程仅等待工作线程（同一张图可反复用于多轮）
        wk_t.join(timeout=86400.0 if not args.once else 120.0)
        running.clear()
    else:
        # Camera + imshow must stay on main thread (macOS / Qt backend requirement).
        run_camera_on_main_thread(
            buf,
            lk,
            args.frame_interval,
            running,
            args.camera_index,
            args.no_preview,
        )
        running.clear()
        wk_t.join(timeout=30.0)


if __name__ == "__main__":
    main()
