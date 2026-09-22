# 端侧 Decide LLM（树莓派）部署说明

> **正式选型（沿用此前建议）**  
> **Qwen2.5-1.5B-Instruct** + **Q4_K_M GGUF** + **llama.cpp / llama-cpp-python**  
> 中国大陆优先从 **ModelScope（魔搭）** 下载；备选 **HF 镜像**。  
> **端侧不再使用 OpenAI / 云端 Decide。**

| 机型 | 模型 | 备注 |
|------|------|------|
| 测试机 4GB | Qwen2.5-1.5B Q4（或 0.5B） | 与 MoMask 错峰更稳 |
| 真机 16GB | **Qwen2.5-1.5B Q4（默认）**；质量不够再升 3B | 决策约 1s 量级 |

---

## A. Mac 推送代码（新开「本地」终端，不要在 SSH 里）

```bash
cd /Users/emmaliu/Desktop/HKUST/RBM-project/Model/momask-codes
# 重新打包（含 edge decide + 安装脚本）
tar --disable-copyfile --no-xattrs -czf dist/pipeline_edge_pi.tgz \
  pipeline/ \
  scripts/pi_install_edge_llm.sh \
  scripts/pi_download_qwen_gguf.sh \
  scripts/pi_install_cloud_pipeline.sh \
  源码/TonyPi/Functions/FaceExpression.py \
  源码/TonyPi/Functions/EmotionActionScheduler.py \
  源码/TonyPi/Functions/model/geometry_emotion_model.json

scp dist/pipeline_edge_pi.tgz cat@raspberrypi.local:/tmp/
```

密码：`Liushuwen123`

---

## B. 树莓派（已有的 SSH 窗口）

```bash
cd ~/RBM-project/momask-codes
tar -xzf /tmp/pipeline_edge_pi.tgz
source venv_inference/bin/activate

# 一键：装 llama-cpp-python + 下 Qwen GGUF + 冒烟（无 OpenAI）
bash scripts/pi_install_edge_llm.sh
```

若下载慢，可只重试权重：

```bash
bash scripts/pi_download_qwen_gguf.sh
```

权重默认路径：

```text
~/RBM-project/momask-codes/models/edge_llm/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

---

## C. 日常运行（端侧 LLM Decide）

```bash
cd ~/RBM-project/momask-codes
source venv_inference/bin/activate

export DECIDE_BACKEND=edge_llm
export EDGE_LLM_GGUF=$PWD/models/edge_llm/qwen2.5-1.5b-instruct-q4_k_m.gguf
export EDGE_LLM_N_THREADS=4

# 冒烟（假感知 + 端侧 Qwen 写 prompt，不跑 MoMask）
python -m pipeline.run_mac_ab --once --mock-perception --dry-run-b \
  --decide-backend edge_llm \
  --transcript '你好，我有点累，陪我一下'

# 真接 MoMask（需 checkpoints 已在）
python -m pipeline.run_mac_ab --once --mock-perception \
  --decide-backend edge_llm \
  --gpu-id -1
```

`edge_auto`：有 GGUF 就走 Qwen，否则规则兜底。

---

## D. 与云侧关系

| 环境 | Decide |
|------|--------|
| Mac 联调 | 可用 `DECIDE_BACKEND=cloud`（OpenAI 兼容中转） |
| **树莓派正式** | **`edge_llm` = Qwen2.5-1.5B**，禁止依赖 OpenAI |

规则引擎（`edge` / `mock`）仅作无权重时的兜底，**不是**正式分析模型。

---

## E. 第一段闭环（情绪 + 关键词 + ASR → 预设动作，不跑 MoMask）

在 Mac 打包：

```bash
bash scripts/pi_pack_emotion_llm.sh
scp dist/pipeline_emotion_llm_pi.tgz cat@<pi-host>:/tmp/
```

在树莓派：

```bash
cd ~/RBM-project/momask-codes
tar -xzf /tmp/pipeline_emotion_llm_pi.tgz
source venv_inference/bin/activate
# 若尚未装 Qwen：bash scripts/pi_install_edge_llm.sh
# 端侧 ASR：pip install openai-whisper
sudo systemctl stop tonypi    # 必须，否则摄像头被主程序占用

export DECIDE_BACKEND=edge_llm
export EDGE_LLM_GGUF=$PWD/models/edge_llm/qwen2.5-1.5b-instruct-q4_k_m.gguf
export EDGE_LLM_N_THREADS=4

# S0 冒烟（假感知，舵机不动）
python -m pipeline.run_pi_emotion_llm --once --mock-perception --simulate \
  --decide-backend edge_llm --transcript '你好，我有点累，陪我一下'

# S1–S4 真机循环：脸 / 小幻小幻+指令 / 唤醒后说话
python -m pipeline.run_pi_emotion_llm --decide-backend edge_llm
# 或：bash scripts/pi_run_emotion_llm.sh
```

本循环默认 **不** 调用 MoMask。打开开关后，预设动作仍然立刻执行，关节在旁边生成。

开关：

```bash
# 关（默认）
python -m pipeline.run_pi_emotion_llm --no-momask
# 或 ENABLE_MOMASK=0

# 开：ActionGroup + 并发生成 joints.npy
python -m pipeline.run_pi_emotion_llm --momask
# 或 ENABLE_MOMASK=1 bash scripts/pi_run_emotion_llm.sh

# 开但不加载权重（只写 prompt 标记，联调开关用）
python -m pipeline.run_pi_emotion_llm --momask --momask-dry-run --once --mock-perception --simulate
```

JSON 写在 `pipeline_runs/*_emotion_llm.json`，字段 `momask` 为 true/false。
WonderEcho 默认 `/dev/ttyUSB0`（可用 `WONDERECHO_PORT` 改）。未插麦时仍可走视觉。

Track A 保底不再轮播单个 ActionGroup：视觉 + 听觉打分后选一条 2～3 片段短语
（`pipeline/preset_phrases.py`），会话内禁止 8 轮内重复同一 `phrase_id`、3 轮内重复同一 clip。
日志里看 `phrase id=` / `clips=` / `recovery=`。停 / 前进后退转仍走硬规则（短步短语）。


