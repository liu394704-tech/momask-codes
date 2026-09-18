"""
实时摄像头 +（可选：文件或麦克风）→ 中转台 OpenAI 兼容 API（可选 Whisper + 多模态 MLLM 视觉）→ JSON。

多模态模型名由 OPENAI_MODEL 决定（如 gpt-5.4-mini、gpt-4o 等，以中转台路由为准）。
请求发往 OPENAI_BASE_URL（须以 /v1 结尾），密钥为中转台下发的 sk-。

无音频时仍会送摄像头帧做视觉分析；有音频则把 Whisper 文本一并交给 MLLM。

先测 API（不调摄像头）:
  export OPENAI_API_KEY="sk-..."
  export OPENAI_BASE_URL="https://你的中转台/v1"
  python live_emotion_monitor.py --test-api

摄像头 + 多模态实时视觉分析（默认不录音，仅画面；按预览窗口里 q 退出）:
  export OPENAI_API_KEY="sk-..."
  export OPENAI_BASE_URL="https://你的中转台/v1"
  python live_emotion_monitor.py
  # 需要麦克风 + Whisper 时: 加参数 --mic 或 export LIVE_MONITOR_USE_MIC=1
  # 串 MoMask（GPU）: python live_emotion_monitor.py --momask --gpu-id 0

环境变量（必填）:
  export OPENAI_API_KEY="sk-..."
  export OPENAI_BASE_URL="https://你的中转台/v1"

可选:
  export OPENAI_MODEL="gpt-4o"   # 例：gpt-5.4-mini、gpt-4o、gpt-4o-mini（以中转台为准）
  export WHISPER_MODEL="whisper-1"   # 中转台无此渠道时会 503，请改为中转台文档中的语音模型名
  export LIVE_MONITOR_WHISPER_LOCAL=0  # 1=用本机 openai-whisper 转写，不经中转台（需 pip install openai-whisper torch）
  export LIVE_MONITOR_WHISPER_LOCAL_SIZE=base  # tiny/base/small… 越大越准越慢
  export LIVE_MONITOR_WHISPER_LANGUAGE=       # 可选：zh 等，帮助本地模型
  export LIVE_MONITOR_USE_MIC=1          # 1=每轮麦克风录音 + Whisper（或命令行加 --mic）
  export LIVE_MONITOR_DEMO_AUDIO="demo_audio.m4a"  # 未开麦克风时：若该文件存在则转写并送入 MLLM
  export LIVE_MONITOR_AUDIO_SEC=4.5      # 开麦时与攒帧并行，总墙钟≈max(录音,攒帧)+Whisper+视觉
  export LIVE_MONITOR_PYAUDIO_DEVICE_INDEX=   # 可选：整数，指定 PyAudio 录音设备编号
  export LIVE_MONITOR_AVFOUNDATION=none:0    # macOS ffmpeg 回退时的设备串，无声音可试 none:1
  export LIVE_MONITOR_ALSA_DEVICE=default    # Linux ffmpeg 回退用 ALSA 设备名
  export LIVE_MONITOR_MOMASK=0           # 实验跑通 MLLM→MoMask 时请设为 1
  export LIVE_MONITOR_MOMASK_JOINTS_ONLY=1 # 1=仅保存关节坐标 .npy（最快，默认）；0 或 --momask-export-bvh 则生成 BVH
  export LIVE_MONITOR_MOMASK_RENDER_VIDEO=0   # 1=在非 joints_only 模式下再导出 MP4（很慢）
  export LIVE_MONITOR_MAX_VISION_OK=0    # >0：成功完成这么多轮 GPT 视觉后结束工作线程（例如设为 1 做单次实验）
  export LIVE_MONITOR_FRAME_INTERVAL=1.0 # 越小同一窗口内帧越密（延迟换时序）；与 NUM_FRAMES 相乘影响覆盖时长
  export LIVE_MONITOR_NUM_FRAMES=4       # 视觉采样帧数；过多会增 API 延迟，可 3～5 折中
  export LIVE_MONITOR_FRAME_BUFFER=24    # 环形缓冲保留最近多少帧，避免长间隔时旧帧被挤掉
  export LIVE_MONITOR_VISION_MAX_WIDTH=640  # 上传前缩放到最大宽度（0=不缩放）；512~640 通常够用
  export LIVE_MONITOR_JPEG_QUALITY=78    # 1~100，越低体积越小、越快
  export LIVE_MONITOR_VISION_DETAIL=     # 留空；若中转台兼容 OpenAI vision，可设 low 以加速/省 token
  export LIVE_MONITOR_MAX_VISION_TOKENS=448  # 回复长度上限
  export LIVE_MONITOR_CYCLE_SEC=2.5      # 目标「每轮」周期间隔；实际 sleep=max(MIN_GAP, CYCLE_SEC-本轮耗时)
  export LIVE_MONITOR_MIN_GAP_SEC=0.35   # 每轮结束后至少间隔（秒），便于连续感知又不过度打 API
  export LIVE_MONITOR_CAMERA_WARMUP=3
  export LIVE_MONITOR_FRAME_WAIT_SEC=15  # 单轮等待摄像头攒够帧的最长时间
  export LIVE_MONITOR_CAMERA_INDEX=0     # 多摄像头时改为 1、2…
  export LIVE_MONITOR_LATENCY_LOG=1      # 1=每轮打印各阶段延时；0=关闭
  export LIVE_MONITOR_LATENCY_CSV=""     # 可选：仅延时列的 CSV（与完整实验表可并存）
  export LIVE_MONITOR_EXPERIMENT_LOG=""  # 可选：完整实验表路径；留空则自动写入 experiment/exp_<运行时间>_<微秒>.csv

依赖: pip install openai opencv-python
麦克风: 优先 pip install pyaudio；失败时可 brew/apt 安装 ffmpeg，脚本会自动用 ffmpeg 录 wav

macOS: 预览窗口在主线程（本脚本结构已满足）。
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import platform
import shutil
from datetime import datetime
import subprocess
import sys
import threading
import traceback
import time
import wave

import cv2

# 项目根目录（用于 MoMask 子进程 cwd）
_ROOT = os.path.dirname(os.path.abspath(__file__))

# 多模态大模型：仅通过 OpenAI 兼容接口调用；默认名可改，须与中转台路由一致
DEFAULT_MLLM_MODEL = "gpt-4o"


def _default_experiment_csv_relative() -> str:
    """每次运行一条新日志：experiment/exp_日期时间_微秒.csv（相对项目根）。"""
    t = datetime.now()
    return os.path.join(
        "experiment",
        "exp_%s_%06d.csv" % (t.strftime("%Y%m%d_%H%M%S"), t.microsecond),
    )


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
        "\n[test-api] 目标: 中转台 OpenAI 兼容接口 + 模型=%s\n"
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
        "[test-api] 中转台 + 模型「%s」文本调用正常。可去掉 --test-api 运行完整实时流程。\n" % model,
        flush=True,
    )


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def resize_frame_for_vision(frame, max_width: int):
    """缩小宽度以降低图像 token 与上传体积；max_width<=0 表示不缩放。"""
    if max_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    new_w = max_width
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def record_wav_simple(out_path: str, duration_sec: float, sample_rate: int = 16000) -> tuple[bool, str]:
    """PyAudio 阻塞录音。返回 (成功, 失败原因)。"""
    dev_raw = _env("LIVE_MONITOR_PYAUDIO_DEVICE_INDEX", "")
    input_device_index = None
    if dev_raw != "":
        try:
            input_device_index = int(dev_raw)
        except ValueError:
            input_device_index = None
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
                input_device_index=input_device_index,
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
        if not os.path.isfile(out_path) or os.path.getsize(out_path) < 256:
            return False, "录音文件过小，可能未采集到有效麦克风数据"
        return True, ""
    except Exception as e:
        return False, str(e)[:220]


def record_wav_via_ffmpeg(out_path: str, duration_sec: float, sample_rate: int = 16000) -> tuple[bool, str]:
    """无 PyAudio 时用 ffmpeg 录麦克风（macOS avfoundation / Linux alsa）。"""
    ff = shutil.which("ffmpeg")
    if not ff:
        return False, "未找到 ffmpeg（可 mac: brew install ffmpeg；Linux: apt install ffmpeg）"
    dur = str(max(1, int(round(duration_sec))))
    sr = str(sample_rate)
    system = platform.system()
    try:
        if system == "Darwin":
            av = _env("LIVE_MONITOR_AVFOUNDATION", "none:0").strip() or "none:0"
            cmd = [
                ff,
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-f",
                "avfoundation",
                "-i",
                av,
                "-t",
                dur,
                "-ac",
                "1",
                "-ar",
                sr,
                "-f",
                "wav",
                out_path,
            ]
        elif system == "Linux":
            alsa = _env("LIVE_MONITOR_ALSA_DEVICE", "default").strip() or "default"
            cmd = [
                ff,
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-f",
                "alsa",
                "-i",
                alsa,
                "-t",
                dur,
                "-ac",
                "1",
                "-ar",
                sr,
                "-f",
                "wav",
                out_path,
            ]
        else:
            return False, "ffmpeg 麦克风回退仅支持 macOS 与 Linux"
        if os.path.isfile(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=float(duration_sec) + 25.0,
        )
        if r.returncode != 0:
            err = (r.stderr or r.stdout or "").strip() or "ffmpeg 退出码 %d" % r.returncode
            return False, err[:220]
        if not os.path.isfile(out_path) or os.path.getsize(out_path) < 256:
            return False, "ffmpeg 输出文件过小或不存在"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "ffmpeg 录音超时"
    except Exception as e:
        return False, str(e)[:220]


def record_mic_wav(wav_path: str, audio_sec: float) -> tuple[bool, str]:
    """录制麦克风到 wav。成功返回 (True, '')，失败 (False, 诊断)。"""
    ok, err = record_wav_simple(wav_path, audio_sec)
    if ok:
        return True, ""
    ok2, err2 = record_wav_via_ffmpeg(wav_path, audio_sec)
    if ok2:
        return True, ""
    return False, "PyAudio: %s | ffmpeg: %s" % (err or "?", err2 or "?")


def get_audio_path_for_cycle(
    *,
    use_mic: bool,
    demo_path: str,
    audio_sec: float,
    temp_dir: str,
) -> tuple[str | None, str]:
    """返回 (Whisper 用 wav 路径, 麦克风失败时的诊断)。无音频源时路径为 None。"""
    os.makedirs(temp_dir, exist_ok=True)
    if use_mic:
        wav_path = os.path.join(temp_dir, "temp_live_mic.wav")
        ok, diag = record_mic_wav(wav_path, audio_sec)
        if ok:
            return wav_path, ""
        return None, diag
    if not demo_path or not os.path.isfile(demo_path):
        return None, ""
    return os.path.abspath(demo_path), ""


_local_whisper_state: tuple[str, object] | None = None


def analyze_audio_whisper_api(client, audio_path: str, whisper_model: str) -> str:
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=whisper_model,
            file=audio_file,
        )
    return (transcription.text or "").strip() or "(empty)"


def analyze_audio_whisper_local(audio_path: str) -> str:
    """本机 openai-whisper，不依赖中转台 /v1/audio/transcriptions。"""
    global _local_whisper_state
    try:
        import whisper  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "未安装 openai-whisper，请执行: pip install openai-whisper  （并需 torch）"
        ) from e
    size = (_env("LIVE_MONITOR_WHISPER_LOCAL_SIZE", "base").strip() or "base")
    if _local_whisper_state is None or _local_whisper_state[0] != size:
        print(
            "[音频] 正在加载本地 Whisper 模型「%s」（首次会下载权重，可能较慢）…" % size,
            flush=True,
        )
        _local_whisper_state = (size, whisper.load_model(size))
    model = _local_whisper_state[1]
    lang = _env("LIVE_MONITOR_WHISPER_LANGUAGE", "").strip() or None
    kwargs: dict = {"fp16": False}
    if lang:
        kwargs["language"] = lang
    result = model.transcribe(audio_path, **kwargs)
    return (result.get("text") or "").strip() or "(empty)"


def transcribe_audio(client, audio_path: str, whisper_model: str) -> str:
    if _env_bool("LIVE_MONITOR_WHISPER_LOCAL", False):
        return analyze_audio_whisper_local(audio_path)
    return analyze_audio_whisper_api(client, audio_path, whisper_model)


def whisper_failure_hint(exc: BaseException) -> str:
    s = str(exc)
    if "503" in s or "502" in s or "429" in s or "无可用渠道" in s:
        return (
            " | 建议: (1) export WHISPER_MODEL=中转台已开通的语音模型名 "
            "(2) 或 export LIVE_MONITOR_WHISPER_LOCAL=1 使用本机 openai-whisper，并 pip install openai-whisper torch"
        )
    return ""


def print_audio_startup_guide(
    *,
    use_mic: bool,
    demo_audio: str,
    whisper_model: str,
    whisper_local: bool,
) -> None:
    """启动时说明听觉链路是否可用，避免误以为「已录音」。"""
    if whisper_local:
        print(
            "\n[听觉] Whisper 模式: 本机 openai-whisper（不经中转台语音接口）。模型规模: %s\n"
            % _env("LIVE_MONITOR_WHISPER_LOCAL_SIZE", "base"),
            flush=True,
        )
    elif use_mic or (demo_audio and os.path.isfile(demo_audio)):
        print(
            "\n[听觉] Whisper 模式: 中转台 API，模型名 WHISPER_MODEL=%s。"
            " 若报 503「无可用渠道」，请向中转台更换可用模型名，或设 LIVE_MONITOR_WHISPER_LOCAL=1 走本机转写。\n"
            % whisper_model,
            flush=True,
        )
    if use_mic:
        print("[听觉] 已启用：每轮将尝试麦克风 → Whisper。", flush=True)
        try:
            import pyaudio  # noqa: F401

            print("      PyAudio: 可导入。", flush=True)
        except Exception as e:
            print(
                "      PyAudio: 不可用（%s）；将依赖 ffmpeg 录 wav。"
                % str(e).replace("\n", " ")[:120],
                flush=True,
            )
        ff = shutil.which("ffmpeg")
        if ff:
            print("      ffmpeg: 已找到（作 PyAudio 失败时的回退）。", flush=True)
        else:
            print("      ffmpeg: 未在 PATH 中；若 PyAudio 失败则无法录音。", flush=True)
        print(
            "      macOS 若无声：系统设置 → 隐私 → 麦克风 → 勾选终端/IDE；"
            "或尝试 export LIVE_MONITOR_AVFOUNDATION=none:1\n",
            flush=True,
        )
    else:
        print(
            "\n[听觉] 未启用实时麦克风。要录音 + Whisper 请使用:\n"
            "      python live_emotion_monitor.py --mic\n"
            "      或 export LIVE_MONITOR_USE_MIC=1\n",
            flush=True,
        )
        if demo_audio and os.path.isfile(demo_audio):
            print(
                "      已发现演示文件「%s」：每轮会用该文件调用 Whisper（非实时麦）。\n"
                % demo_audio,
                flush=True,
            )


def run_mllm_vision_json(
    client,
    *,
    model: str,
    user_speech: str,
    frame_paths: list[str],
    vision_detail: str | None = None,
    max_tokens: int = 384,
    temperature: float = 0.35,
) -> dict:
    content_list = []
    for i, frame_path in enumerate(frame_paths):
        b64 = encode_image_to_base64(frame_path)
        url_obj: dict = {"url": "data:image/jpeg;base64,%s" % b64}
        if vision_detail in ("low", "high", "auto"):
            url_obj["detail"] = vision_detail
        item = {"type": "image_url", "image_url": url_obj}
        content_list.append(item)

    prompt = """You are the vision+affect module of a humanoid robot that will socially interact with the person in the camera.
Transcribed speech from the user (may be empty or a placeholder if no mic): "%s"
Here are sequential camera frames from the same time window.

Output ONLY valid JSON with exactly these keys (no markdown):
"user_action": string, in Chinese — only what the HUMAN is doing with their body (pose, gesture, movement) visible in the frames. Do not describe the robot here.

"emotion": string, in Chinese — infer affect with HIGH sensitivity to the FACE: brows, eyelids, gaze direction, mouth corners, jaw tension, micro-expressions, and blinking patterns; use speech wording as extra evidence if present. Prefer a specific label (e.g. 略带紧张、隐忍的不耐烦、克制的期待、微妙的尴尬) over generic fillers like 平静、放松、中性 unless the face and voice truly show calm neutrality. If two readings are close, pick the slightly stronger or more informative one (still one short phrase, under 28 Chinese characters).

"robot_reaction": string, in English — ONE short HumanML3D-style motion description of how the ROBOT should move as social feedback to this person's emotional state while interacting (e.g. open posture with slow nod to acknowledge tension; gentle side-step to give space; slight forward lean with palms open to invite trust). Describe ONLY the robot's own body from the robot's viewpoint. Do NOT imitate, copy, or replay the user's current pose or action; the robot is responding to emotion and context, not mirroring the user.
""" % (
        user_speech.replace('"', "'")[:800],
    )
    content_list.append({"type": "text", "text": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content_list}],
        response_format={"type": "json_object"},
        temperature=temperature,
        max_tokens=max_tokens,
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
    mllm_model: str,
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
    print("  【周期 %d】  实时感知结果  |  模型: %s" % (cycle, mllm_model or "—"), flush=True)
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
    joints_only: bool,
) -> tuple[int, float]:
    """调用 gen_t2m；返回 (退出码, 子进程墙钟秒数)。"""
    gen_py = os.path.join(_ROOT, "gen_t2m.py")
    if not os.path.isfile(gen_py):
        print("[momask] gen_t2m.py not found at", gen_py)
        return -1, 0.0
    mode = "含 MP4 渲染"
    if joints_only:
        mode = "仅关节坐标 .npy（跳过 BVH/MP4，最快）"
    elif not render_video:
        mode = "跳过 MP4，导出 BVH + joints .npy"
    print("\n[momask] calling gen_t2m … (%s)" % mode, flush=True)
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
    if joints_only:
        cmd.append("--joints_only")
    elif not render_video:
        cmd.append("--no_video_render")
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=_ROOT)
    elapsed = time.perf_counter() - t0
    if r.returncode == 0:
        tail = "joints/*/sample*_*.npy（关节三维坐标 T×22×3）"
        if not joints_only:
            tail += " 与 animations/*/*.bvh"
        print(
            "[momask] 完成 → ./generation/gpt4o_reaction/（或 gpt4o_reaction_runN）\n"
            "        %s" % tail,
            flush=True,
        )
    else:
        print("[momask] 失败，退出码", r.returncode, flush=True)
    return r.returncode, elapsed


def _append_latency_csv(
    path: str,
    row: dict[str, float | int | str],
) -> None:
    if not path:
        return
    abs_path = path if os.path.isabs(path) else os.path.join(_ROOT, path)
    d = os.path.dirname(abs_path)
    if d:
        os.makedirs(d, exist_ok=True)
    # momask_ok: -1=未调用 MoMask, 0=失败, 1=成功
    fieldnames = [
        "ts_iso",
        "cycle",
        "whisper_s",
        "buf_wait_s",
        "vision_s",
        "momask_s",
        "wall_s",
        "vision_ok",
        "momask_ok",
    ]
    new_file = not os.path.isfile(abs_path)
    with open(abs_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def _abspath_under_root(p: str) -> str:
    if not p:
        return ""
    return os.path.abspath(p if os.path.isabs(p) else os.path.join(_ROOT, p))


def _resolve_experiment_log_path(requested: str) -> tuple[str, str | None]:
    """返回 (写入用的路径, 若非空则为首轮提示：因旧文件为窄表头而改用新文件)。"""
    if not requested:
        return "", None
    abs_path = _abspath_under_root(requested)
    if not os.path.isfile(abs_path):
        return requested, None
    with open(abs_path, "r", encoding="utf-8-sig") as f:
        first = f.readline()
    if "user_action" in first:
        return requested, None
    d, base = os.path.split(abs_path)
    stem, ext = os.path.splitext(base)
    alt_abs = os.path.join(d, "%s_full%s" % (stem, ext))
    if not os.path.isabs(requested):
        try:
            alt = os.path.relpath(alt_abs, _ROOT)
        except ValueError:
            alt = alt_abs
    else:
        alt = alt_abs
    msg = (
        "检测到「%s」为旧版「仅延时」列；完整实验列已改写入「%s」。"
        "之后请打开带 _full 的文件查看识别描述与机器人动作建议。"
        % (requested, alt)
    )
    return alt, msg


def _append_experiment_log_csv(path: str, row: dict[str, float | int | str]) -> None:
    """实验用表格日志（UTF-8 CSV，可用 Excel / Numbers 打开）。列含模型、分阶段耗时、识别结果与机器人动作建议。"""
    if not path:
        return
    abs_path = path if os.path.isabs(path) else os.path.join(_ROOT, path)
    d = os.path.dirname(abs_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fieldnames = [
        "ts_iso",
        "cycle",
        "vision_model",
        "whisper_model",
        "num_frames",
        "momask_enabled",
        "whisper_s",
        "buf_wait_s",
        "vision_s",
        "momask_s",
        "wall_s",
        "speech_to_model",
        "user_action",
        "emotion",
        "robot_reaction",
        "vision_ok",
        "momask_ok",
        "error",
    ]
    new_file = not os.path.isfile(abs_path)
    with open(abs_path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


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
    momask_joints_only: bool,
    max_vision_ok: int,
    temp_dir: str,
    latency_log: bool,
    latency_csv: str,
    experiment_log: str,
    vision_max_width: int,
    jpeg_quality: int,
    vision_api_detail: str,
    max_vision_tokens: int,
    min_gap_sec: float,
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
        t_wall0 = time.perf_counter()
        t_whisper_s = 0.0

        # ----- 音频：演示文件直接取路径；麦克风与「攒帧」并行以压缩墙钟 -----
        user_speech = "（本轮无语音输入）"
        audio_ok: bool | None = None
        audio_detail = "未使用麦克风且未找到演示音频文件，本轮仅根据画面分析"
        audio_path: str | None = None
        mic_diag = ""
        mic_thread: threading.Thread | None = None
        mic_holder: dict[str, str | None] = {"path": None, "diag": None}
        wav_path = os.path.join(temp_dir, "temp_live_mic.wav")

        if use_mic:
            print(
                "[音频] 麦克风录制 %.1fs（与下方视频攒帧并行，墙钟约 max(录音, 攒帧)）…" % audio_sec,
                flush=True,
            )

            def _mic_job() -> None:
                try:
                    ok, diag = record_mic_wav(wav_path, audio_sec)
                    if ok:
                        mic_holder["path"] = wav_path
                        mic_holder["diag"] = ""
                    else:
                        mic_holder["path"] = None
                        mic_holder["diag"] = diag
                except Exception as e:
                    mic_holder["path"] = None
                    mic_holder["diag"] = str(e)[:220]

            mic_thread = threading.Thread(target=_mic_job, daemon=True)
            mic_thread.start()
        else:
            audio_path, mic_diag = get_audio_path_for_cycle(
                use_mic=False,
                demo_path=demo_audio,
                audio_sec=audio_sec,
                temp_dir=temp_dir,
            )

        # ----- 等待足够视频帧 -----
        skip_cycle = False
        skip_video_timeout = False
        deadline = time.time() + frame_wait_sec
        t_buf_w0 = time.perf_counter()
        last_n = -1
        while True:
            with buffer_lock:
                nbuf = len(latest_frames_buffer)
            if nbuf >= num_frames:
                break
            if time.time() > deadline:
                skip_video_timeout = True
                skip_cycle = True
                break
            if nbuf != last_n:
                last_n = nbuf
                print(
                    "[视频] 缓冲中… %d / %d 帧" % (nbuf, num_frames),
                    flush=True,
                )
            time.sleep(0.2)

        t_buf_wait_s = time.perf_counter() - t_buf_w0

        if use_mic and mic_thread is not None:
            mic_thread.join(timeout=float(audio_sec) + 45.0)
            if mic_thread.is_alive():
                mic_holder["path"] = None
                mic_holder["diag"] = "麦克风录音线程超时"
            audio_path = mic_holder["path"]
            mic_diag = (mic_holder.get("diag") or "") if not audio_path else ""

        if audio_path:
            tw0 = time.perf_counter()
            try:
                user_speech = transcribe_audio(client, audio_path, whisper_model)
                audio_ok = True
                audio_detail = (
                    "本机 Whisper 转写完成"
                    if _env_bool("LIVE_MONITOR_WHISPER_LOCAL", False)
                    else "Whisper API 转写完成"
                )
            except Exception as e:
                audio_ok = False
                audio_detail = ("Whisper 失败: %s%s" % (str(e)[:400], whisper_failure_hint(e))).strip()
                user_speech = "（语音识别失败，仅根据画面分析）"
            t_whisper_s = time.perf_counter() - tw0
        elif use_mic:
            audio_ok = False
            audio_detail = "麦克风录音失败。%s 提示：系统设置中授予终端/Python 麦克风权限；pip install pyaudio；或安装 ffmpeg 作回退；mac 无声音可试 export LIVE_MONITOR_AVFOUNDATION=none:1" % (
                ("详情: " + mic_diag + "。") if mic_diag else "",
            )

        if skip_video_timeout:
            with buffer_lock:
                nbuf = len(latest_frames_buffer)
            print_cycle_report(
                cycle=cycle,
                mllm_model=model,
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
            time.sleep(max(min_gap_sec, cycle_sec))

        if skip_cycle:
            if experiment_log:
                wall_skip = time.perf_counter() - t_wall0
                _append_experiment_log_csv(
                    experiment_log,
                    {
                        "ts_iso": datetime.now().isoformat(timespec="milliseconds"),
                        "cycle": cycle,
                        "vision_model": model,
                        "whisper_model": whisper_model,
                        "num_frames": num_frames,
                        "momask_enabled": 1 if momask else 0,
                        "whisper_s": round(t_whisper_s, 6),
                        "buf_wait_s": round(t_buf_wait_s, 6),
                        "vision_s": 0.0,
                        "momask_s": 0.0,
                        "wall_s": round(wall_skip, 6),
                        "speech_to_model": user_speech,
                        "user_action": "",
                        "emotion": "",
                        "robot_reaction": "",
                        "vision_ok": 0,
                        "momask_ok": -1,
                        "error": "frame_buffer_timeout",
                    },
                )
            continue

        with buffer_lock:
            frames_to_analyze = list(latest_frames_buffer)
        frame_paths = []
        jq = max(1, min(100, jpeg_quality))
        for i, frame in enumerate(frames_to_analyze[-num_frames:]):
            path = os.path.join(temp_dir, "temp_mllm_frame_%d.jpg" % i)
            small = resize_frame_for_vision(frame, vision_max_width)
            cv2.imwrite(path, small, [cv2.IMWRITE_JPEG_QUALITY, jq])
            frame_paths.append(os.path.abspath(path))

        vw = "原分辨率" if vision_max_width <= 0 else "宽≤%d" % vision_max_width
        video_detail = "已采集 %d 帧（%s，JPEG=%d）→ %s" % (num_frames, vw, jq, model)

        vision_ok = False
        reaction = ""
        ua, em, rr = "", "", ""
        vision_err = ""
        t_vision_s = 0.0
        tv0: float | None = None
        try:
            print("[大脑] 正在分析画面与语境 …", flush=True)
            tv0 = time.perf_counter()
            vd = vision_api_detail.strip().lower() if vision_api_detail else ""
            vd_arg = vd if vd in ("low", "high", "auto") else None
            data = run_mllm_vision_json(
                client,
                model=model,
                user_speech=user_speech,
                frame_paths=frame_paths,
                vision_detail=vd_arg,
                max_tokens=max_vision_tokens,
                temperature=0.35,
            )
            t_vision_s = time.perf_counter() - tv0
            ua, em, rr = _vision_fields(data)
            vision_ok = True
            reaction = rr.strip()
            print_cycle_report(
                cycle=cycle,
                mllm_model=model,
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
            vision_err = str(e)
            t_vision_s = (time.perf_counter() - tv0) if tv0 is not None else 0.0
            print_cycle_report(
                cycle=cycle,
                mllm_model=model,
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

        t_momask_s = 0.0
        momask_ok = -1
        if vision_ok:
            vision_ok_count += 1
            if momask and reaction:
                rc, t_momask_s = trigger_momask_generation(
                    reaction,
                    momask_gpu,
                    render_video=momask_render_video,
                    joints_only=momask_joints_only,
                )
                momask_ok = 1 if rc == 0 else 0

        wall_s = time.perf_counter() - t_wall0
        if latency_log:
            mm_note = "MoMask=关节.npy" if momask_joints_only else "MoMask=BVH/视频等"
            print(
                "\n[延时] cycle=%d  whisper=%.3fs  攒帧等待=%.3fs  MLLM视觉(%s)=%.3fs  momask=%.3fs  本轮墙钟=%.3fs  (%s)\n"
                % (
                    cycle,
                    t_whisper_s,
                    t_buf_wait_s,
                    model,
                    t_vision_s,
                    t_momask_s,
                    wall_s,
                    mm_note,
                ),
                flush=True,
            )
        if latency_csv:
            _append_latency_csv(
                latency_csv,
                {
                    "ts_iso": datetime.now().isoformat(timespec="milliseconds"),
                    "cycle": cycle,
                    "whisper_s": round(t_whisper_s, 6),
                    "buf_wait_s": round(t_buf_wait_s, 6),
                    "vision_s": round(t_vision_s, 6),
                    "momask_s": round(t_momask_s, 6),
                    "wall_s": round(wall_s, 6),
                    "vision_ok": 1 if vision_ok else 0,
                    "momask_ok": momask_ok,
                },
            )

        if experiment_log:
            _append_experiment_log_csv(
                experiment_log,
                {
                    "ts_iso": datetime.now().isoformat(timespec="milliseconds"),
                    "cycle": cycle,
                    "vision_model": model,
                    "whisper_model": whisper_model,
                    "num_frames": num_frames,
                    "momask_enabled": 1 if momask else 0,
                    "whisper_s": round(t_whisper_s, 6),
                    "buf_wait_s": round(t_buf_wait_s, 6),
                    "vision_s": round(t_vision_s, 6),
                    "momask_s": round(t_momask_s, 6),
                    "wall_s": round(wall_s, 6),
                    "speech_to_model": user_speech,
                    "user_action": ua,
                    "emotion": em,
                    "robot_reaction": rr if vision_ok else "",
                    "vision_ok": 1 if vision_ok else 0,
                    "momask_ok": momask_ok,
                    "error": vision_err if not vision_ok else "",
                },
            )

        if max_vision_ok > 0 and vision_ok_count >= max_vision_ok:
            print(
                "\n[实验] 已完成 %d 轮成功多模态视觉分析（达到 LIVE_MONITOR_MAX_VISION_OK），工作线程结束。\n"
                "      摄像头预览仍在运行，按 q 退出窗口。\n" % vision_ok_count,
                flush=True,
            )
            break

        elapsed = time.time() - t0
        gap = max(min_gap_sec, cycle_sec - elapsed)
        if gap > 0:
            time.sleep(gap)


latest_frames_buffer: list = []
buffer_lock = threading.Lock()


def camera_interaction_loop(
    frame_interval: float,
    camera_index: int = 0,
    *,
    window_title: str = "Live emotion monitor (q=quit)",
    frame_buffer_max: int = 24,
) -> None:
    global latest_frames_buffer
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[摄像头] 无法打开设备 %d，请检查权限、占用或换 --camera-index。" % camera_index, flush=True)
        return
    last_capture_time = time.time()
    print("[摄像头] 预览已打开（设备 %d），按键盘 q 退出。" % camera_index, flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_flipped = cv2.flip(frame, 1)
        cv2.imshow(window_title, frame_flipped)

        current_time = time.time()
        if current_time - last_capture_time >= frame_interval:
            with buffer_lock:
                latest_frames_buffer.append(frame_flipped.copy())
                if len(latest_frames_buffer) > frame_buffer_max:
                    latest_frames_buffer[:] = latest_frames_buffer[-frame_buffer_max:]
            last_capture_time = current_time

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    # 本脚本仅使用 OPENAI_*（中转台 OpenAI 兼容）。若终端要求 ZHIPU_API_KEY，说明跑成了 scripts/live_vision_momask.py 或旧副本，请 git pull 后确认路径。
    print(
        "[live_emotion_monitor] 中转台 OpenAI 兼容（OPENAI_API_KEY + OPENAI_BASE_URL），非智谱链路。\n",
        flush=True,
    )
    ap = argparse.ArgumentParser(description="Live vision + Whisper + 多模态 MLLM (OpenAI 兼容中转台)")
    ap.add_argument(
        "--test-api",
        action="store_true",
        help="仅测试中转台文本 API（OPENAI_MODEL），不启动摄像头",
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
        "--momask-export-bvh",
        action="store_true",
        help="生成 BVH（较慢）；默认仅 --joints_only 保存关节坐标 .npy",
    )
    ap.add_argument(
        "--max-vision-ok",
        type=int,
        default=int(_env("LIVE_MONITOR_MAX_VISION_OK", "0")),
        help="成功完成这么多轮多模态视觉后退出工作线程；0=不限制",
    )
    ap.add_argument(
        "--camera-index",
        type=int,
        default=int(_env("LIVE_MONITOR_CAMERA_INDEX", "0")),
        help="OpenCV 摄像头设备编号，默认 0",
    )
    ap.add_argument(
        "--mic",
        action="store_true",
        help="启用麦克风录音 + Whisper（等同 LIVE_MONITOR_USE_MIC=1）",
    )
    ap.add_argument(
        "--experiment-log",
        type=str,
        default="",
        metavar="PATH",
        help="实验表格 CSV 路径（等同 LIVE_MONITOR_EXPERIMENT_LOG；相对路径相对项目根）",
    )
    args = ap.parse_args()

    if args.test_api:
        test_relay_mllm()
        return

    momask = args.momask or _env_bool("LIVE_MONITOR_MOMASK", False)
    momask_render_video = args.momask_render_video or _env_bool(
        "LIVE_MONITOR_MOMASK_RENDER_VIDEO", False
    )
    momask_joints_only = _env_bool("LIVE_MONITOR_MOMASK_JOINTS_ONLY", True)
    if args.momask_export_bvh:
        momask_joints_only = False
    max_vision_ok = int(args.max_vision_ok)
    if max_vision_ok < 0:
        max_vision_ok = 0
    use_mic = args.mic or _env_bool("LIVE_MONITOR_USE_MIC", False)
    demo_audio = _env("LIVE_MONITOR_DEMO_AUDIO", "demo_audio.m4a")
    if not os.path.isabs(demo_audio):
        demo_audio = os.path.join(_ROOT, demo_audio)

    audio_sec = max(0.5, float(_env("LIVE_MONITOR_AUDIO_SEC", "4.5")))
    cycle_sec = max(0.2, float(_env("LIVE_MONITOR_CYCLE_SEC", "2.5")))
    min_gap_sec = max(0.0, float(_env("LIVE_MONITOR_MIN_GAP_SEC", "0.35")))
    warmup_sec = float(_env("LIVE_MONITOR_CAMERA_WARMUP", "3"))
    frame_interval = max(0.12, float(_env("LIVE_MONITOR_FRAME_INTERVAL", "1.0")))
    num_frames = max(1, int(_env("LIVE_MONITOR_NUM_FRAMES", "4")))
    frame_buffer_max = max(12, int(_env("LIVE_MONITOR_FRAME_BUFFER", "24")))
    frame_wait_sec = float(_env("LIVE_MONITOR_FRAME_WAIT_SEC", "15"))

    vision_max_width = int(_env("LIVE_MONITOR_VISION_MAX_WIDTH", "640"))
    if vision_max_width < 0:
        vision_max_width = 0
    jpeg_quality = max(1, min(100, int(_env("LIVE_MONITOR_JPEG_QUALITY", "78"))))
    vision_detail_raw = _env("LIVE_MONITOR_VISION_DETAIL", "")
    max_vision_tokens = max(64, int(_env("LIVE_MONITOR_MAX_VISION_TOKENS", "448")))

    model = _env("OPENAI_MODEL", DEFAULT_MLLM_MODEL)
    whisper_model = _env("WHISPER_MODEL", "whisper-1")
    whisper_local = _env_bool("LIVE_MONITOR_WHISPER_LOCAL", False)
    whisper_display = (
        "本机 openai-whisper(%s)" % _env("LIVE_MONITOR_WHISPER_LOCAL_SIZE", "base")
        if whisper_local
        else "API「%s」" % whisper_model
    )
    temp_dir = os.path.join(_ROOT, ".live_monitor_tmp")
    os.makedirs(temp_dir, exist_ok=True)

    latency_log = _env_bool("LIVE_MONITOR_LATENCY_LOG", True)
    latency_csv_in = _env("LIVE_MONITOR_LATENCY_CSV", "").strip()
    experiment_explicit = (args.experiment_log or _env("LIVE_MONITOR_EXPERIMENT_LOG", "")).strip()
    os.makedirs(os.path.join(_ROOT, "experiment"), exist_ok=True)
    if experiment_explicit:
        experiment_in = experiment_explicit
    else:
        experiment_in = _default_experiment_csv_relative()

    same_csv_target = (
        bool(latency_csv_in)
        and bool(experiment_explicit)
        and _abspath_under_root(latency_csv_in) == _abspath_under_root(experiment_in)
    )
    # 仅当用户显式把「延时」与「实验表」指到同一文件时，只写完整表、不写窄延时
    latency_csv = "" if same_csv_target else latency_csv_in

    experiment_log, experiment_redirect_msg = _resolve_experiment_log_path(experiment_in)

    client = make_openai_client()

    print(
        "\n"
        "========================================================================\n"
        "  Live Emotion Monitor — 配置一览\n"
        "  中转台 API: %s\n"
        "  多模态 MLLM: %s  |  Whisper: %s\n"
        "  麦克风: %s  |  MoMask: %s  |  MoMask 导出: %s\n"
        "  连续感知: 录音 %.1fs | 帧间隔 %.2fs × %d 帧 | 缓冲 %d | 周期 %.1fs | 轮间 ≥ %.2fs\n"
        "  延时日志: %s  |  延时 CSV: %s\n"
        "  实验表格 CSV: %s\n"
        "  视觉输入: %d 帧  |  宽≤%s  JPEG=%d  detail=%s  max_tokens=%d\n"
        "  成功视觉轮数上限: %s  |  每轮等待画面帧: 最长 %.0fs\n"
        "========================================================================\n"
        % (
            _env("OPENAI_BASE_URL"),
            model,
            whisper_display,
            "开" if use_mic else "关",
            "开" if momask else "关",
            (
                "仅关节 .npy（最快）"
                if (momask and momask_joints_only)
                else (
                    ("MP4 开" if momask_render_video else "无 MP4") + " + BVH"
                    if momask
                    else "—"
                )
            ),
            audio_sec,
            frame_interval,
            num_frames,
            frame_buffer_max,
            cycle_sec,
            min_gap_sec,
            "开" if latency_log else "关",
            (latency_csv_in if latency_csv_in else "（未设置）"),
            (
                experiment_log
                if experiment_log
                else "（未设置）"
            ),
            num_frames,
            ("原图" if vision_max_width <= 0 else str(vision_max_width)),
            jpeg_quality,
            (vision_detail_raw if vision_detail_raw else "（未设）"),
            max_vision_tokens,
            str(max_vision_ok) if max_vision_ok else "不限",
            frame_wait_sec,
        ),
        flush=True,
    )

    if not experiment_explicit:
        print(
            "[实验记录] 未设置 LIVE_MONITOR_EXPERIMENT_LOG：完整实验表已自动写入「%s」"
            "（含模型名、user_action、emotion、robot_reaction、各阶段耗时；列 cycle 为轮次）。\n"
            % experiment_log,
            flush=True,
        )
    if experiment_redirect_msg:
        print("[实验记录] %s\n" % experiment_redirect_msg, flush=True)

    print_audio_startup_guide(
        use_mic=use_mic,
        demo_audio=demo_audio,
        whisper_model=whisper_model,
        whisper_local=whisper_local,
    )

    if not use_mic and not os.path.isfile(demo_audio):
        print(
            "[提示] 未检测到演示音频「%s」，且未开麦克风：本轮将只做摄像头画面分析。\n"
            % demo_audio,
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
            momask_joints_only=momask_joints_only,
            max_vision_ok=max_vision_ok,
            temp_dir=temp_dir,
            latency_log=latency_log,
            latency_csv=latency_csv,
            experiment_log=experiment_log,
            vision_max_width=vision_max_width,
            jpeg_quality=jpeg_quality,
            vision_api_detail=vision_detail_raw,
            max_vision_tokens=max_vision_tokens,
            min_gap_sec=min_gap_sec,
        ),
        daemon=True,
    )
    brain.start()
    cam_title = "Live — %s (q=quit)" % model
    camera_interaction_loop(
        frame_interval,
        camera_index=args.camera_index,
        window_title=cam_title,
        frame_buffer_max=frame_buffer_max,
    )


if __name__ == "__main__":
    main()
