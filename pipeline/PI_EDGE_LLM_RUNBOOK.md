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
（手写核心 + 组合扩展，见 `pipeline/preset_phrases.py`），会话内禁止 8 轮内重复同一 `phrase_id`、3 轮内重复同一 clip。
日志里看 `phrase id=` / `clips=` / `recovery=`。停 / 前进后退转仍走硬规则（短步短语）。
每个片段在动作名确定之后会补一版 16 路总线脉宽 + 毫米坐标（站立=500，机身 373×186×106 mm，左臂抬手用官方抓取课 14=180/15=260/16=650）。这是按仓库尺寸模拟的坐标层，真机仍播出厂 `.d6a`。日志里看 `coords:` / `coords_peak`。

幻尔自带情绪只有脸（`FaceExpression`），没有听觉情绪。语音情绪要另装开源端侧模型
`iic/emotion2vec_plus_seed`（FunASR / 魔搭，不是中转站）。

**只连 HW 热点时：机器人上不了网，不能在热点里从魔搭/HF 拉权重。**
脸模已经在 `/home/pi/TonyPi/Functions`，778 条短语也不要额外权重，连着 `HW*` 就能跑动作闭环。
Qwen GGUF / emotion2vec 的正确顺序是：**电脑先上网下载 → 再连 HW* / VNC 拷进机器人。**

```bash
# 1) 电脑连家里/学校网（不要连 HW*）
cd /Users/emmaliu/Desktop/HKUST/RBM-project/Model/momask-codes
bash scripts/mac_download_edge_weights.sh --gguf-only
# 得到 dist/edge_weights/edge_llm/qwen2.5-1.5b-instruct-q4_k_m.gguf（约 1GB）

# 2) 电脑改连 HW*，打开 VNC（192.168.149.1）
#    文件管理器：把该 gguf 拷到
#    ~/RBM-project/momask-codes/models/edge_llm/
#    或先拷到 U 盘，插机器人，VNC 打开 /media/pi/* 或 /media/cat/*

# 3) 树莓派 VNC 终端
cd ~/RBM-project/momask-codes
bash scripts/pi_stage_weights_offline.sh ~/Desktop
# 或：bash scripts/pi_stage_weights_offline.sh /media/pi/USB

# 同一热点也可用 scp（不必 VNC 拖文件）
scp dist/edge_weights/edge_llm/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  cat@192.168.149.1:~/RBM-project/momask-codes/models/edge_llm/
```

有局域网时再装运行时和在线下载：

```bash
bash scripts/pi_install_audio_ser.sh
export ENABLE_AUDIO_SER=1
```

说「小幻小幻」后同一段 wav：Whisper 出文本，emotion2vec 出 `audio_emotion`。
未安装时循环仍可跑，只是 `audio_emotion` 为空。

---

## D. 用 WonderPi 启动（装一次，之后不用 VNC）

手机 App 只有 12 个固定按钮，不能加第 13 个。安装脚本把功能 6「人脸识别」换成 778 条短语循环。其他 11 个玩法不变。TonyPi 必须保持开机自启；不要再开 `tonypi-emotion-llm`，那个服务会抢走摄像头。

电脑先连能上网的 Wi-Fi，拉到这条分支，再改连 HW*（192.168.149.1）：

```bash
cd /Users/emmaliu/Desktop/HKUST/RBM-project/Model/momask-codes
git fetch origin cursor/cloud-agent-1790042632488-ttam2
git checkout -B cursor/cloud-agent-1790042632488-ttam2 origin/cursor/cloud-agent-1790042632488-ttam2

scp /Users/emmaliu/Desktop/HKUST/RBM-project/Model/momask-codes/pipeline/wonderpi_face_game.py \
  pi@192.168.149.1:/home/pi/RBM-project/momask-codes/pipeline/wonderpi_face_game.py
scp /Users/emmaliu/Desktop/HKUST/RBM-project/Model/momask-codes/scripts/pi_install_wonderpi_game.sh \
  pi@192.168.149.1:/home/pi/RBM-project/momask-codes/scripts/pi_install_wonderpi_game.sh

ssh pi@192.168.149.1
sudo bash /home/pi/RBM-project/momask-codes/scripts/pi_install_wonderpi_game.sh
```

SSH 密码是树莓派用户 `pi` 的密码，不是热点密码 `hiwonder`。脚本会备份 `Running.py`，写入 `Functions/EmotionPhrase.py`，关掉情绪开机服务，并重启 `tonypi`。

然后只开 WonderPi：进入「人脸识别」，人对着胸口摄像头。画面上会先显示 calibrating，再显示情绪。保持微笑或皱眉大约 1 秒，身体会播一条 2–3 个片段的短语。两条动作之间大约 6–12 秒。停在这个页面里；离开页面约 7 秒后心跳超时，玩法会退出并回到站立。


