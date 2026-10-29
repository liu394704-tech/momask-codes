# ForgeBench-E Pilot

家庭陪伴机器人的记忆 / 遗忘初步评测。方案、数据集、指标和可复现结果都在这个目录。

- 完整方案（文献对齐后的步骤）：[完整实验方案.md](完整实验方案.md)
- 两页方案幻灯片：[陪伴机器人记忆遗忘_综述与实验方案.pptx](陪伴机器人记忆遗忘_综述与实验方案.pptx)（方案怎么建立，加上已跑的冒烟和尺子。`python3 experiment/forget_bench/build_review_pptx.py` 会同时写到仓库和 `/home/ubuntu/Desktop/`）
- 试点设计与已跑出的数字：[实验设计_具身记忆遗忘初步方案.md](实验设计_具身记忆遗忘初步方案.md)
- 数据集：[`dataset.py`](dataset.py) 是源，[`dataset/pilot_v0.json`](dataset/pilot_v0.json) 是导出
- 结果：[results/pilot_report.md](results/pilot_report.md)

```bash
python3 experiment/forget_bench/run_bench.py
python3 -m unittest tests/test_forget_bench.py -v
```

这一步不调用大模型，也不接 Habitat / AI2-THOR。它检查的是评测协议能不能把「嘴上说忘了」和「手上还在做、日志里还留着」分开。
