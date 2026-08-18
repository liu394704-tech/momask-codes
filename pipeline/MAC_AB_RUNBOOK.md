# Mac A∥B 联调说明（第一期）

默认模式：**A_parallel_B**（先预置 A，再 B 出 prompt/关节）。

基线冻结见：`experiment/baseline/BASELINE_FREEZE.md`  
改任何东西前先跑基线回归。

---

## 0. 基线回归（必须先过）

```bash
cd 源码
source .venv-emotion-arm/bin/activate
python preset_action_tests/test_preset_emotion_actions.py
```

---

## 1. 无 Key、无摄像头 — 验证 Arbiter（现在就能跑）

在项目根目录：

```bash
cd /Users/emmaliu/Desktop/HKUST/RBM-project/Model/momask-codes
python -m pipeline.run_mac_ab --once --mock-perception --mock-decide --dry-run-b
python -m pipeline.run_mac_ab --once --mode A_only --mock-perception --mock-decide --dry-run-b
python -m pipeline.run_mac_ab --once --mode B_only --mock-perception --mock-decide --dry-run-b
```

期望：打印 `effective=...`，并写出 `pipeline_runs/*_round.json`。

---

## 2. 本地人脸 + mock decide（需摄像头权限）

```bash
cd 源码 && source .venv-emotion-arm/bin/activate && cd ..
python -m pipeline.run_mac_ab --once --mock-decide --dry-run-b
```

系统设置 → 隐私 → 摄像头：允许终端。

---

## 3. 云端 Decide（把 Key 给我 / 自行 export）

配置好后需要：

```bash
export OPENAI_API_KEY='sk-...'
export OPENAI_BASE_URL='https://你的中转台/v1'   # 须含 /v1
export OPENAI_MODEL='gpt-4o-mini'              # 或中转台可用名
```

然后：

```bash
python -m pipeline.run_mac_ab --once --dry-run-b
# 或假感知测 API：
python -m pipeline.run_mac_ab --once --mock-perception --dry-run-b
```

**此时再找我拿 Key 即可**（或你本地 export 后告诉我已就绪）。

---

## 4. 接 MoMask（可选，需 momask_env + 权重）

```bash
conda activate momask_env
python -m pipeline.run_mac_ab --once --mock-perception --mock-decide --gpu-id -1
# 去掉 --dry-run-b，真实写 generation/ 下 joints
```

---

## 降级规则（已实现）

- `confidence < 0.45` / `fallback` / 无 prompt / Decide 失败 → 本轮 **只 A**
- B（MoMask）失败 → 标记 degraded，**不撤销已执行的 A**
