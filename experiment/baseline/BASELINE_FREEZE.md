# Baseline Freeze — 可运行最小标准（冻结版）

**冻结日期**: 2026-08-18  
**标签建议**: `baseline/mac-a-preset-momask-v1`  
**原则**: 后续 A∥B / 云端 Decide / 视听 pipeline 优化，**不得破坏本文件列出的回归命令**。

---

## 1. 轨道 A（预置保底）— 必须永远可跑

| 项 | 内容 |
|----|------|
| 权威映射 | `源码/TonyPi/Functions/EmotionActionScheduler.py` |
| 情绪 | `neutral / happy / unhappy / surprised` + mild/strong（阈值 0.55） |
| 离线回归 | `源码/preset_action_tests/test_preset_emotion_actions.py` |
| Mac 策略模拟 | `源码/facedetect_mac_demo.py`（不驱动真机） |

```bash
cd 源码
source .venv-emotion-arm/bin/activate   # 或等价 mediapipe 环境
python preset_action_tests/test_preset_emotion_actions.py
# 期望: ALL PASS
```

映射摘要：

- happy mild: wave, stepping | strong: chest, wave, twist  
- unhappy mild: bow, jugong | strong: squat, bow, jugong  
- surprised mild: twist, stepping | strong: back_fast, twist, stand  
- 禁止: kick / uppercut / wing_chun 等  

---

## 2. 轨道 B 后端（MoMask）— 最小生成标准

| 项 | 内容 |
|----|------|
| 入口 | `gen_t2m.py`（建议 `--no_video_render`） |
| 批量计时 | `scripts/bench_pi_gen.py` |
| 决策→prompt 样例 | `experiment/e2e_spec/`（本地产物，可能被 gitignore） |
| 环境 | 本机 `momask_env` = **Python 3.8**；真机建议 ≤3.10，勿与系统 3.12 混装 |

```bash
# 在已装好权重与 momask_env 的机器上
conda activate momask_env
python gen_t2m.py --gpu_id -1 --ext baseline_smoke \
  --text_prompt "A person waves with the right hand." \
  --no_video_render
```

期望：产出 joints `.npy`，无 traceback。

---

## 3. 架构冻结约定（后续优化必须遵守）

- **A** = 预置保底；**B** = 感知→Decide→MoMask 正式路径  
- **Arbiter** 三模式：`A_only` / `B_only` / `A_parallel_B`（第一期默认 **A_parallel_B**）  
- `confidence < 0.45` 或 `fallback=true` 或 Decide/MoMask 失败 → **本轮只 A**  
- 精细英文 `action_prompt` 由 Decide 生成；粗 4 类情绪只服务 A/仲裁  

---

## 4. 回归清单（改代码后先跑）

1. `python 源码/preset_action_tests/test_preset_emotion_actions.py` → ALL PASS  
2. （可选）`python pipeline/run_mac_ab.py --mode A_only --once --mock-decide`  
3. （有权重时）一条 `gen_t2m.py --no_video_render` smoke  

若 1 失败：**停止优化，先修回基线。**
