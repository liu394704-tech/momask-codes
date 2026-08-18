# 预置情绪 → 动作 测试说明（不含 MoMask）

## 当前映射（权威配置）

文件：`TonyPi/Functions/EmotionActionScheduler.py`

| 情绪 | mild（置信度 < 0.55） | strong（置信度 ≥ 0.55） |
|------|----------------------|-------------------------|
| neutral | 无 | 无 |
| happy | `wave`, `stepping` | `chest`, `wave`, `twist` |
| unhappy | `bow`, `jugong` | `squat`, `bow`, `jugong` |
| surprised | `twist`, `stepping` | `back_fast`, `twist`, `stand` |

- 恢复动作：`stand`
- 已排除攻击性动作：kick / uppercut / wing_chun / shot 等

## 离线自动测试（推荐先跑）

```bash
cd /Users/emmaliu/Desktop/HKUST/RBM-project/Model/momask-codes/源码
source .venv-emotion-arm/bin/activate
python preset_action_tests/test_preset_emotion_actions.py
```

只看人工清单：

```bash
python preset_action_tests/test_preset_emotion_actions.py --checklist
```

## 人工实时测试（Mac，不驱动机器人）

```bash
cd /Users/emmaliu/Desktop/HKUST/RBM-project/Model/momask-codes/源码
source .venv-emotion-arm/bin/activate
python facedetect_mac_demo.py --duration 60
```

按 `scenarios.json` 里 `live_checklist`（L1–L8）做表情，观察终端：

```text
stable happy (mild 0.48) -> would plan wave
stable happy (strong 0.71) -> would plan chest
```

## 真机测试（可选）

在 TonyPi 上启用 FaceDetect 玩法，日志应类似：

```text
[FaceDetect] 稳定情绪 happy (strong, conf=0.71) -> 动作 chest
```

确认不会出现 kick / uppercut 等动作名。
