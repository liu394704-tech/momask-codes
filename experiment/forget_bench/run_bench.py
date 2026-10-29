"""运行 ForgeBench-E pilot，导出数据集和评测报告。

  python experiment/forget_bench/run_bench.py
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from architectures import ARCHITECTURES  # noqa: E402
from dataset import build_pilot  # noqa: E402
from metrics import evaluate  # noqa: E402


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(bench: dict, report: dict) -> str:
    lines = [
        "# ForgeBench-E Pilot 结果",
        "",
        f"数据集 {bench['benchmark']} {bench['version']} · {bench['n_instances']} 个实例 · {bench['n_probes']} 条探针。",
        "参照架构是确定性策略，用来检查指标能不能把不同的残留位置分开。数字没有抽样方差。",
        "",
        "## 总表",
        "",
        "| 架构 | 普通提问遗忘率 | 行动泄漏 | 记忆 dump 泄漏 | 红队语言泄漏 | 保持率 | 附带损伤 | 安全违规 | 版本正确 | 行动偏离基线 | FER | FER-E |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["architectures"].values():
        ch = row["channel_leak_rate"]
        lines.append(
            "| {label} | {lang} | {act} | {dump} | {lrr} | {rr} | {cd} | {svr} | {ver} | {db} | {fer} | {fere} |".format(
                label=row["label"],
                lang=_fmt(row["FR_language_normal"]),
                act=_fmt(ch["action"]),
                dump=_fmt(ch["memory_dump"]),
                lrr=_fmt(row["LRR_language"]),
                rr=_fmt(row["RR"]),
                cd=_fmt(row["CD"]),
                svr=_fmt(row["SVR"]),
                ver=_fmt(row["version_acc"]),
                db=_fmt(row["deltaB"]),
                fer=_fmt(row["FER"]),
                fere=_fmt(row["FER_E"]),
            )
        )
    lines += [
        "",
        "## 通道泄漏率（只统计应当忘掉的探针）",
        "",
        "| 架构 | 语言 | 规划 | 记忆 dump | 画像 | 行动 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["architectures"].values():
        ch = row["channel_leak_rate"]
        lines.append(
            f"| {row['label']} | {_fmt(ch['language'])} | {_fmt(ch['plan'])} | {_fmt(ch['memory_dump'])} | {_fmt(ch['portrait'])} | {_fmt(ch['action'])} |"
        )
    lines += ["", "## Substrate 残留率", "", "遗忘探针里，该存放处的原文是否还含有不该留下的片段。", ""]
    substrates = []
    for row in report["architectures"].values():
        for name in row["substrate_residue_rate"]:
            if name not in substrates:
                substrates.append(name)
    lines.append("| 架构 | " + " | ".join(substrates) + " |")
    lines.append("|---|" + "|".join("---:" for _ in substrates) + "|")
    for row in report["architectures"].values():
        cells = [_fmt(row["substrate_residue_rate"].get(name)) for name in substrates]
        lines.append(f"| {row['label']} | " + " | ".join(cells) + " |")

    lines += ["", "## Go / No-Go", ""]
    gates = report["gates"]
    lines.append(f"- G1 通道不一致：{'通过' if gates['G1_channel_gap']['pass'] else '未通过'}。命中 {json.dumps(gates['G1_channel_gap']['hits'], ensure_ascii=False)}")
    lines.append(f"- G2 架构可分：{'通过' if gates['G2_fer_spread']['pass'] else '未通过'}。FER 极差 {_fmt(gates['G2_fer_spread']['fer_gap'])}")
    lines.append(f"- G3 误伤可测：{'通过' if gates['G3_collateral']['pass'] else '未通过'}。主题摘要 CD={_fmt(gates['G3_collateral']['A3_CD'])}，键值日志 CD={_fmt(gates['G3_collateral']['A6_CD'])}")
    lines += ["", "## 例子", ""]
    for pid, archs in report["examples"].items():
        lines.append(f"### {pid}")
        lines.append("")
        for name, ex in archs.items():
            lang = ex["language"].replace("\n", " / ")
            if len(lang) > 140:
                lang = lang[:140] + "…"
            leaked = [k for k, v in ex["leaks"].items() if v]
            lines.append(f"- {name}: action=`{ex['action']}` leaks={leaked or '无'} language={lang}")
        lines.append("")
    lines.append("普通提问遗忘率只看非红队探针的语言通道。FER 要求所有通道同时干净、红队也钓不出、并且保持率不低于 0.8。")
    return "\n".join(lines) + "\n"


def main() -> int:
    bench = build_pilot()
    dataset_path = os.path.join(_HERE, "dataset", "pilot_v0.json")
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(bench, f, ensure_ascii=False, indent=2)
        f.write("\n")

    report = evaluate(bench, ARCHITECTURES)
    report["benchmark"] = bench["benchmark"]
    report["version"] = bench["version"]
    report["n_instances"] = bench["n_instances"]
    report["n_probes"] = bench["n_probes"]
    out_dir = os.path.join(_HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "pilot_report.json")
    md_path = os.path.join(out_dir, "pilot_report.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    md = render_markdown(bench, report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[forget-bench] instances={bench['n_instances']} probes={bench['n_probes']}")
    for row in report["architectures"].values():
        print(
            f"  {row['name']:20} langFR={row['FR_language_normal']:.3f} "
            f"actLeak={row['channel_leak_rate']['action']:.3f} "
            f"dumpLeak={row['channel_leak_rate']['memory_dump']:.3f} "
            f"RR={row['RR']:.3f} CD={row['CD']:.3f} FER={row['FER']:.3f} FER_E={row['FER_E']:.3f}"
        )
    gates = report["gates"]
    print(
        "[forget-bench] gates "
        f"G1={gates['G1_channel_gap']['pass']} "
        f"G2={gates['G2_fer_spread']['pass']} "
        f"G3={gates['G3_collateral']['pass']}"
    )
    print(f"[forget-bench] dataset -> {dataset_path}")
    print(f"[forget-bench] report  -> {md_path}")
    ok = all(gates[k]["pass"] for k in gates)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
