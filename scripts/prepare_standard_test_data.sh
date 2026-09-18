#!/usr/bin/env bash
# 在项目根生成最小「合成」音视频，便于立刻跑 pytest / offline_mllm_clip（无需下载大库）。
# 公开标准集（RAVDESS、CREMA-D 等）体积大且许可各异，请按下方链接自行下载到 test_data/external/

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p test_data/synthetic test_data/external

echo "==> 生成合成测试素材到 test_data/synthetic/ ..."
if python3 -c "import cv2, numpy" 2>/dev/null; then
python3 <<'PY'
import os
import math
import wave
import struct

import cv2
import numpy as np

root = os.path.abspath(".")
syn = os.path.join(root, "test_data", "synthetic")
os.makedirs(syn, exist_ok=True)

# 2s 16kHz mono sine wav
wav_path = os.path.join(syn, "synthetic.wav")
fr = 16000
dur = 2
with wave.open(wav_path, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(fr)
    for i in range(fr * dur):
        v = int(32000 * math.sin(2 * math.pi * 440 * i / fr))
        wf.writeframes(struct.pack("<h", max(-32767, min(32767, v))))
print("Wrote", wav_path)

# 3s 640x480 mp4 moving gradient
h, w = 480, 640
fps = 10
n = 3 * fps
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
mp4_path = os.path.join(syn, "synthetic_rgb.mp4")
vw = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))
for t in range(n):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = (t * 7) % 256
    frame[:, :, 1] = (t * 13 + 40) % 256
    frame[:, :, 2] = (255 - t * 5) % 256
    vw.write(frame)
vw.release()
print("Wrote", mp4_path)
PY
else
  echo "（跳过）当前 python 无 cv2/numpy，请在 conda momask 环境中重新运行本脚本以生成 synthetic 素材。"
fi

echo ""
echo "==> 本地自检（不调用网络）:"
python3 -m unittest discover -s tests -p "test_*.py" -v

echo ""
echo "-------------------------------------------------------------------"
echo "检查已下载数据布局与 CREMA 音视频配对:"
echo "  python scripts/verify_benchmark_data.py"
echo ""
echo "公开多模态情绪数据集（请浏览器下载后放入 test_data/external/）："
echo "  RAVDESS: https://smartlaboratory.org/ravdess/"
echo "  CREMA-D: https://www.kaggle.com/datasets/ejlok1/crema-d  （或检索 CREMA-D official）"
echo "  LibriSpeech (语音 WER): https://www.openslr.org/12/"
echo ""
echo "离线跑一条 MLLM（需 OPENAI_*）示例:"
echo "  export OPENAI_API_KEY=... OPENAI_BASE_URL=.../v1"
echo "  python scripts/offline_mllm_clip.py --video test_data/synthetic/synthetic_rgb.mp4 \\"
echo "      --audio test_data/synthetic/synthetic.wav --out-json /tmp/offline_one.json"
echo ""
echo "大规模离线批跑（递归扫描视频；同目录同名 .wav 会自动配对）:"
echo "  python scripts/batch_offline_mllm.py --root test_data/external/RAVDESS \\"
echo "      --out-csv experiment/offline_runs/ravdess_run1.csv \\"
echo "      --out-json-dir experiment/offline_runs/ravdess_run1_json \\"
echo "      --limit 50 --sleep 0.5 --resume"
echo ""
echo "对齐标签 + 计算量化指标（accuracy / macro F1 / weighted F1 / kappa / 混淆矩阵 / 延迟分位）:"
echo "  python scripts/eval_emotion_labels.py \\"
echo "      --csv experiment/offline_runs/ravdess_run1.csv \\"
echo "      --dataset ravdess \\"
echo "      --out-dir experiment/offline_runs/ravdess_run1_eval"
echo ""
echo "  # 输出: summary.json / confusion_matrix.csv / per_class.csv / sample_predictions.csv"
echo "  # CREMA-D 用 --dataset cremad；多数据集混合可用 --dataset auto"
echo "-------------------------------------------------------------------"
