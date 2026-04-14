"""
实时摄像头 +（可选：文件或麦克风）→ 中转台 OpenAI 兼容 API（可选 Whisper + MLLM GPT-4o 视觉）→ JSON。

MLLM 固定走 GPT-4o 多模态（可通过 OPENAI_MODEL 覆盖名称以匹配中转台路由）。
请求发往 OPENAI_BASE_URL（须以 /v1 结尾），密钥为中转台下发的 sk-。

无音频时仍会送摄像头帧做视觉分析；有音频则把 Whisper 文本一并交给 GPT-4o。

先测 API（不调摄像头）:
  export OPENAI_API_KEY="sk-..."
  export OPENAI_BASE_URL="https://你的中转台/v1"
  python live_emotion_monitor.py --test-api

环境变量（必填）:
  export OPENAI_API_KEY="sk-..."
  export OPENAI_BASE_URL="https://你的中转台/v1"

可选:
  export OPENAI_MODEL="gpt-4o"   # 默认即为 gpt-4o；若中转台要求别名可改
  export WHISPER_MODEL="whisper-1"
  export LIVE_MONITOR_USE_MIC=1          # 1=用麦克风录一段 wav；0=不用麦克风
  export LIVE_MONITOR_DEMO_AUDIO="demo_audio.m4a"  # 非麦克风时若文件存在则转写，否则仅视觉
  export LIVE_MONITOR_AUDIO_SEC=3
  export LIVE_MONITOR_MOMASK=0           # 实验跑通 MLLM→MoMask 时请设为 1
  export LIVE_MONITOR_MOMASK_RENDER_VIDEO=0   # 1=gen_t2m 同时导出 MP4；0=仅 BVH + joints .npy（默认，更快）
  export LIVE_MONITOR_MAX_VISION_OK=0    # >0：成功完成这么多轮 GPT 视觉后结束工作线程（例如设为 1 做单次实验）
  export LIVE_MONITOR_FRAME_INTERVAL=1.5
  export LIVE_MONITOR_CYCLE_SEC=5
  export LIVE_MONITOR_CAMERA_WARMUP=3
  export LIVE_MONITOR_FRAME_WAIT_SEC=15  # 单轮等待摄像头攒够帧的最长时间

依赖: pip install openai opencv-python
麦克风: pip install pyaudio（可选）或系统 ffmpeg

macOS: 预览窗口在主线程（本脚本结构已满足）。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import threading
import traceback
import time
import wave

import cv2

# 项目根目录（用于 MoMask 子进程 cwd）
_ROOT = os.path.dirname(os.path.abspath(__file__))

# 多模态大模型：本脚本仅通过 OpenAI 兼容接口调用；默认模型名为 GPT-4o（与中转台路由一致时可不改）
DEFAULT_MLLM_MODEL = "gpt-4o"


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_bool(key: str, default: bool = False) -> bool:
    v = _env(key).lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def make_openai_client():
    from openai import OpenAI

    api_key = _env("OPENAI_API_KEY")
    base_url = _env("OPENAI_BASE_URL")
    if not api_key:
        raise SystemExit("请设置环境变量 OPENAI_API_KEY")
    if not base_url:
        raise SystemExit("请设置环境变量 OPENAI_BASE_URL（中转台提供的 OpenAI 兼容根地址，通常以 /v1 结尾）")
    return OpenAI(api_key=api_key, base_url=base_url)


def test_relay_mllm() -> None:
    """仅测中转台文本通道：验证 sk + base_url + 模型名能否完成一次 chat（不调摄像头/Whisper）。"""
    model = _env("OPENAI_MODEL", DEFAULT_MLLM_MODEL)
    base = _env("OPENAI_BASE_URL")
    print(
        "\n[test-api] 目标: 中转台 OpenAI 兼容接口 + MLLM=%s\n"
        "           BASE_URL=%s\n" % (model, base),
        flush=True,
    )
    client = make_openai_client()
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=16,
            temperature=0,
        )
        text = (r.choices[0].message.content or "").strip()
        print("[test-api] 成功。模型返回:\n  %s\n" % text[:500], flush=True)
    except Exception as e:
        print("[test-api] 失败:", e, flush=True)
        raise SystemExit(1) from e
    print(
        "[test-api] 中转台 + GPT-4o（%s）文本调用正常。可去掉 --test-api 运行完整实时流程。\n" % model,
        flush=True,
    )


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def record_wav_simple(out_path: str, duration_sec: float, sample_rate: int = 16000) -> bool:
    """优先 PyAudio 阻塞录音；失败则返回 False。"""
    try:
        import pyaudio  # type: ignore

        chunk = 1024
        fmt = pyaudio.paInt16
        channels = 1
        frames = []
        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(
                format=fmt,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk,
            )
            n_read = int(sample_rate * duration_sec / chunk) + 2
            for _ in range(n_read):
                frames.append(stream.read(chunk, exception_on_overflow=False))
            stream.stop_stream()
            stream.close()
        finally:
            pa.terminate()
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))
        return True
    except Exception:
        return False


def get_audio_path_for_cycle(
    *,
    use_mic: bool,
    demo_path: str,
    audio_sec: float,
    temp_dir: str,
) -> str | None:
    """返回本周期用于 Whisper 的音频文件路径；无音频源时返回 None（不刷屏）。"""
    os.makedirs(temp_dir, exist_ok=True)
    if use_mic:
        wav_path = os.path.join(temp_dir, "temp_live_mic.wav")
        if record_wav_simple(wav_path, audio_sec):
            return wav_path
        return None
    if not demo_path or not os.path.isfile(demo_path):
        return None
    return os.path.abspath(demo_path)


def analyze_audio_whisper(client, audio_path: str, whisper_model: str) -> str:
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=whisper_model,
            file=audio_file,
        )
    return (transcription.text or "").strip() or "(empty)"


def run_gpt4o_vision_json(
    client,
    *,
    model: str,
    user_speech: str,
    frame_paths: list[str],
) -> dict:
    content_list = []
    for i, frame_path in enumerate(frame_paths):
        b64 = encode_image_to_base64(frame_path)
        item = {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,%s" % b64},
        }
        # 部分中转站不支持 detail 字段，省略
        content_list.append(item)

    prompt = """You are the vision brain of a humanoid robot.
Transcribed speech from the user (may be empty or a placeholder if no mic): "%s"
Here are sequential camera frames from the same time window.

Output ONLY valid JSON with exactly these keys (no markdown):
"user_action": string, in Chinese — what the user is doing with their body (pose, gesture, movement) visible in the frames.
"emotion": string, in Chinese — the emotional tone you infer from face/body/context.
"robot_reaction": string, in English — one short HumanML3D-style motion description for the robot (e.g. "A person steps back slowly with arms raised defensively").
""" % (
        user_speech.replace('"', "'")[:800],
    )
    content_list.append({"type": "text", "text": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content_list}],
        response_format={"type": "json_object"},
        temperature=0.4,
    )
    result_str = response.choices[0].message.content or "{}"
    try:
        return json.loads(result_str)
    except json.JSONDecodeError:
        # 极少数情况模型仍带 markdown
        s = result_str.strip()
        if "```" in s:
            s = s.split("```", 1)[-1].split("```", 1)[0]
            if s.startswith("json"):
                s = s[4:].strip()
        return json.loads(s)


def _vision_fields(data: dict) -> tuple[str, str, str]:
    """从 JSON 取出：用户动作、情绪、机器人动作建议（英文）。"""
    ua = (data.get("user_action") or data.get("用户动作") or "").strip()
    em = (data.get("emotion") or "").strip()
    rr = (data.get("robot_reaction") or "").strip()
    return ua, em, rr


def print_cycle_report(
    *,
    cycle: int,
    video_ok: bool,
    video_detail: str,
    audio_ok: bool | None,
    audio_detail: str,
    transcript: str,
    user_action: str,
    emotion: str,
    robot_reaction: str,
    error_line: str | None = None,
) -> None:
    """终端主输出：一眼能看懂当前信号与识别结果。"""
    w = 72
    line = "-" * w
    eq = "=" * w
    print("\n" + eq, flush=True)
    print("  【周期 %d】  实时感知结果" % cycle, flush=True)
    print(line, flush=True)
    vmark = "成功" if video_ok else "失败"
    print("  [视频] %s  |  %s" % (vmark, video_detail), flush=True)
    if audio_ok is None:
        print("  [音频] 未使用  |  %s" % audio_detail, flush=True)
    else:
        amark = "成功" if audio_ok else "失败"
        print("  [音频] %s  |  %s" % (amark, audio_detail), flush=True)
        if transcript and audio_ok is not False:
            ts = transcript.replace("\n", " ")[:240]
            print("        识别文字: %s" % ts, flush=True)
    print(line, flush=True)
    if error_line:
        print("  [错误] %s" % error_line, flush=True)
        print(eq + "\n", flush=True)
        return
    print("  用户动作（画面）: %s" % (user_action or "（无）"), flush=True)
    print("  情绪感受:         %s" % (emotion or "（无）"), flush=True)
    print("  机器人动作建议:   %s" % (robot_reaction or "（无）"), flush=True)
    print(eq + "\n", flush=True)


def trigger_momask_generation(
    action_prompt: str,
    gpu_id: int,
    *,
    render_video: bool,
) -> None:
    gen_py = os.path.join(_ROOT, "gen_t2m.py")
    if not os.path.isfile(gen_py):
        print("[momask] gen_t2m.py not found at", gen_py)
        return
    print("\n[momask] calling gen_t2m … (%s)" % ("含 MP4 渲染" if render_video else "跳过 MP4，仅 joints/BVH"), flush=True)
    cmd = [
        sys.executable,
        gen_py,
        "--gpu_id",
        str(gpu_id),
        "--ext",
        "gpt4o_reaction",
        "--text_prompt",
        action_prompt,
    ]
    if not render_video:
        cmd.append("--no_video_render")
    r = subprocess.run(cmd, cwd=_ROOT)
    if r.returncode == 0:
        print(
            "[momask] 完成 → ./generation/gpt4o_reaction/（或 gpt4o_reaction_runN）\n"
            "        动作数据: joints/*/sample*_*.npy 与 animations/*/*.bvh",
            flush=True,
        )
    else:
        print("[momask] 失败，退出码", r.returncode, flush=True)


def api_worker_thread(
    *,
    client,
    model: str,
    whisper_model: str,
    use_mic: bool,
    demo_audio: str,
    audio_sec: float,
    num_frames: int,
    cycle_sec: float,
    warmup_sec: float,
    frame_wait_sec: float,
    momask: bool,
    momask_gpu: int,
    momask_render_video: bool,
    max_vision_ok: int,
    temp_dir: str,
) -> None:
    cycle = 0
    vision_ok_count = 0
    if warmup_sec > 0:
        print(
            "\n[启动] 摄像头预热 %.1fs（请面向镜头），随后每轮会输出结构化结果。\n" % warmup_sec,
            flush=True,
        )
        time.sleep(warmup_sec)

    while True:
        cycle += 1
        t0 = time.time()

        # ----- 音频（可选）：无文件且未开麦克风则跳过 Whisper，仍做视觉 -----
        user_speech = "（本轮无语音输入）"
        audio_ok: bool | None = None
        audio_detail = "未使用麦克风且未找到演示音频文件，本轮仅根据画面分析"

        if use_mic:
            print("[音频] 正在从麦克风录制 %.1fs …" % audio_sec, flush=True)
        audio_path = get_audio_path_for_cycle(
            use_mic=use_mic,
            demo_path=demo_audio,
            audio_sec=audio_sec,
            temp_dir=temp_dir,
        )

        if audio_path:
            try:
                user_speech = analyze_audio_whisper(client, audio_path, whisper_model)
                audio_ok = True
                audio_detail = "Whisper 转写完成"
            except Exception as e:
                audio_ok = False
                audio_detail = "Whisper 失败: %s" % str(e)[:120]
                user_speech = "（语音识别失败，仅根据画面分析）"
        elif use_mic:
            audio_ok = False
            audio_detail = "麦克风未录到有效音频（可检查权限或安装 pyaudio）"

        # ----- 等待足够视频帧 -----
        skip_cycle = False
        deadline = time.time() + frame_wait_sec
        last_n = -1
        while True:
            with buffer_lock:
                nbuf = len(latest_frames_buffer)
            if nbuf >= num_frames:
                break
            if time.time() > deadline:
                print_cycle_report(
                    cycle=cycle,
                    video_ok=False,
                    video_detail="%.0fs 内未攒够 %d 帧（当前缓冲区 %d 帧），请确认预览窗口已打开"
                    % (frame_wait_sec, num_frames, nbuf),
                    audio_ok=audio_ok,
                    audio_detail=audio_detail,
                    transcript=user_speech if audio_ok else "",
                    user_action="",
                    emotion="",
                    robot_reaction="",
                    error_line="视频帧不足，本轮跳过 GPT 分析。可适当减小 LIVE_MONITOR_FRAME_INTERVAL。",
                )
                time.sleep(max(1.0, cycle_sec))
                skip_cycle = True
                break
            if nbuf != last_n:
                last_n = nbuf
                print(
                    "[视频] 缓冲中… %d / %d 帧" % (nbuf, num_frames),
                    flush=True,
                )
            time.sleep(0.2)

        if skip_cycle:
            continue

        with buffer_lock:
            frames_to_analyze = list(latest_frames_buffer)
        frame_paths = []
        for i, frame in enumerate(frames_to_analyze[-num_frames:]):
            path = os.path.join(temp_dir, "temp_gpt4o_frame_%d.jpg" % i)
            cv2.imwrite(path, frame)
            frame_paths.append(os.path.abspath(path))

        video_detail = "已采集 %d 帧并调用 GPT-4o（%s）" % (num_frames, model)

        vision_ok = False
        reaction = ""
        try:
            print("[大脑] 正在分析画面与语境 …", flush=True)
            data = run_gpt4o_vision_json(
                client,
                model=model,
                user_speech=user_speech,
                frame_paths=frame_paths,
            )
            ua, em, rr = _vision_fields(data)
            vision_ok = True
            reaction = rr.strip()
            print_cycle_report(
                cycle=cycle,
                video_ok=True,
                video_detail=video_detail,
                audio_ok=audio_ok,
                audio_detail=audio_detail,
                transcript=user_speech if audio_ok else "",
                user_action=ua,
                emotion=em,
                robot_reaction=rr,
                error_line=None,
            )
        except Exception as e:
            print_cycle_report(
                cycle=cycle,
                video_ok=True,
                video_detail=video_detail,
                audio_ok=audio_ok,
                audio_detail=audio_detail,
                transcript=user_speech if audio_ok else "",
                user_action="",
                emotion="",
                robot_reaction="",
                error_line=str(e),
            )
            traceback.print_exc()

        if vision_ok:
            vision_ok_count += 1
            if momask and reaction:
                trigger_momask_generation(
                    reaction,
                    momask_gpu,
                    render_video=momask_render_video,
                )

        if max_vision_ok > 0 and vision_ok_count >= max_vision_ok:
            print(
                "\n[实验] 已完成 %d 轮成功 GPT 视觉分析（达到 LIVE_MONITOR_MAX_VISION_OK），工作线程结束。\n"
                "      摄像头预览仍在运行，按 q 退出窗口。\n" % vision_ok_count,
                flush=True,
            )
            break

        elapsed = time.time() - t0
        time.sleep(max(1.0, cycle_sec - elapsed))


latest_frames_buffer: list = []
buffer_lock = threading.Lock()


def camera_interaction_loop(frame_interval: float) -> None:
    global latest_frames_buffer
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[摄像头] 无法打开设备 0，请检查权限与占用。", flush=True)
        return
    last_capture_time = time.time()
    print("[摄像头] 预览已打开，按键盘 q 退出。", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_flipped = cv2.flip(frame, 1)
        cv2.imshow("GPT-4o Omni Monitor (q=quit)", frame_flipped)

        current_time = time.time()
        if current_time - last_capture_time >= frame_interval:
            with buffer_lock:
                latest_frames_buffer.append(frame_flipped.copy())
                if len(latest_frames_buffer) > 12:
                    latest_frames_buffer[:] = latest_frames_buffer[-12:]
            last_capture_time = current_time

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description="Live vision + Whisper + GPT-4o (中转台 OpenAI 兼容)")
    ap.add_argument(
        "--test-api",
        action="store_true",
        help="仅测试中转台 GPT-4o 文本 API，不启动摄像头",
    )
    ap.add_argument(
        "--momask",
        action="store_true",
        help="enable MoMask gen_t2m after each vision result (default: off)",
    )
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=int(_env("LIVE_MONITOR_GPU_ID", "-1")),
        help="GPU for gen_t2m when --momask",
    )
    ap.add_argument(
        "--momask-render-video",
        action="store_true",
        help="gen_t2m 同时渲染 MP4（默认仅导出 BVH + joints .npy，与 LIVE_MONITOR_MOMASK_RENDER_VIDEO=1 等价）",
    )
    ap.add_argument(
        "--max-vision-ok",
        type=int,
        default=int(_env("LIVE_MONITOR_MAX_VISION_OK", "0")),
        help="成功完成这么多轮 GPT 视觉后退出工作线程；0=不限制",
    )
    args = ap.parse_args()

    if args.test_api:
        test_relay_mllm()
        return

    momask = args.momask or _env_bool("LIVE_MONITOR_MOMASK", False)
    momask_render_video = args.momask_render_video or _env_bool(
        "LIVE_MONITOR_MOMASK_RENDER_VIDEO", False
    )
    max_vision_ok = int(args.max_vision_ok)
    if max_vision_ok < 0:
        max_vision_ok = 0
    use_mic = _env_bool("LIVE_MONITOR_USE_MIC", False)
    demo_audio = _env("LIVE_MONITOR_DEMO_AUDIO", "demo_audio.m4a")
    if not os.path.isabs(demo_audio):
        demo_audio = os.path.join(_ROOT, demo_audio)

    audio_sec = float(_env("LIVE_MONITOR_AUDIO_SEC", "3"))
    cycle_sec = float(_env("LIVE_MONITOR_CYCLE_SEC", "5"))
    warmup_sec = float(_env("LIVE_MONITOR_CAMERA_WARMUP", "3"))
    frame_interval = float(_env("LIVE_MONITOR_FRAME_INTERVAL", "1.5"))
    num_frames = int(_env("LIVE_MONITOR_NUM_FRAMES", "3"))
    frame_wait_sec = float(_env("LIVE_MONITOR_FRAME_WAIT_SEC", "15"))

    model = _env("OPENAI_MODEL", DEFAULT_MLLM_MODEL)
    whisper_model = _env("WHISPER_MODEL", "whisper-1")
    temp_dir = os.path.join(_ROOT, ".live_monitor_tmp")
    os.makedirs(temp_dir, exist_ok=True)

    client = make_openai_client()

    print(
        "\n"
        "========================================================================\n"
        "  Live Emotion Monitor — 配置一览\n"
        "  中转台 API: %s\n"
        "  MLLM (GPT-4o): %s  |  Whisper: %s\n"
        "  麦克风: %s  |  MoMask: %s  |  MoMask MP4: %s\n"
        "  成功视觉轮数上限: %s  |  每轮等待画面帧: 最长 %.0fs\n"
        "========================================================================\n"
        % (
            _env("OPENAI_BASE_URL"),
            model,
            whisper_model,
            "开" if use_mic else "关",
            "开" if momask else "关",
            "开" if momask_render_video else "关（仅 .npy/.bvh）",
            str(max_vision_ok) if max_vision_ok else "不限",
            frame_wait_sec,
        ),
        flush=True,
    )

    if not use_mic and not os.path.isfile(demo_audio):
        print(
            "[提示] 未检测到演示音频「%s」，本轮将只做摄像头画面分析（不设文件也能跑）。\n"
            "       若需要语音识别，请设置 LIVE_MONITOR_USE_MIC=1 或放入音频文件。\n" % demo_audio,
            flush=True,
        )

    brain = threading.Thread(
        target=api_worker_thread,
        kwargs=dict(
            client=client,
            model=model,
            whisper_model=whisper_model,
            use_mic=use_mic,
            demo_audio=demo_audio,
            audio_sec=audio_sec,
            num_frames=num_frames,
            cycle_sec=cycle_sec,
            warmup_sec=warmup_sec,
            frame_wait_sec=frame_wait_sec,
            momask=momask,
            momask_gpu=args.gpu_id,
            momask_render_video=momask_render_video,
            max_vision_ok=max_vision_ok,
            temp_dir=temp_dir,
        ),
        daemon=True,
    )
    brain.start()
    camera_interaction_loop(frame_interval)


if __name__ == "__main__":
    main()
