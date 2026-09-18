#!/usr/bin/env python3
"""读取 eval_emotion_labels.py 的输出目录，生成可视化图表。

  python scripts/visualize_eval.py --eval-dir experiment/offline_runs/ravdess_smoke_eval

独立混淆矩阵 + 示例输出目录（可只出两张图 --minimal）:
  python scripts/visualize_eval.py --matrix-csv experiment/examples/confusion_matrix_moderate_standard.csv \\
      --out-dir experiment/examples/viz_moderate_standard --minimal

输出到同一目录：
  - confusion_matrix.png         # 热力图（计数；色标 log1p 避免被极大格「洗白」）
  - confusion_matrix_norm.png    # 行归一化版本
  - per_class_f1.png             # 上：P/R/F1；下：各类 support（真值样本数）
  - latency_distribution.png     # wall_s / vision_s / whisper（视频-only 时第三格为说明）
  - report.md                    # 汇总报告（嵌入图与关键指标）

依赖 matplotlib；若未安装请: pip install matplotlib
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Optional


def _try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
        import numpy as np  # noqa: F401

        return True
    except Exception as e:  # pragma: no cover
        print("[viz] 缺少 matplotlib / numpy：%s" % e, file=sys.stderr)
        print("       pip install matplotlib numpy", file=sys.stderr)
        return False


def _read_summary(eval_dir: str) -> dict:
    p = os.path.join(eval_dir, "summary.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_confusion(eval_dir: str) -> tuple[list[str], list[str], list[list[int]]]:
    p = os.path.join(eval_dir, "confusion_matrix.csv")
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = header[1:]
        rows_labels: list[str] = []
        mat: list[list[int]] = []
        for row in reader:
            rows_labels.append(row[0])
            mat.append([int(x) for x in row[1:]])
    return rows_labels, cols, mat


def _read_per_class(eval_dir: str) -> list[dict[str, str]]:
    p = os.path.join(eval_dir, "per_class.csv")
    out: list[dict[str, str]] = []
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out.append(row)
    return out


def _read_samples(eval_dir: str) -> list[dict[str, str]]:
    p = os.path.join(eval_dir, "sample_predictions.csv")
    out: list[dict[str, str]] = []
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out.append(row)
    return out


def _plot_confusion(
    mat: list[list[int]],
    rows: list[str],
    cols: list[str],
    out_path: str,
    normalize: bool,
    subtitle: str = "",
    *,
    log_color: bool = False,
) -> None:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.array(mat, dtype=float)
    title = "Confusion matrix"
    if normalize:
        row_sum = arr.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            arr = np.where(row_sum > 0, arr / row_sum, 0.0)
        title += " (row-normalized)"
    if log_color and not normalize:
        title += " (color=log1p(count))"
    if subtitle:
        title += "\n" + subtitle

    fig, ax = plt.subplots(figsize=(max(6, len(cols) * 0.9), max(5, len(rows) * 0.9)))
    if log_color and not normalize:
        disp = np.log1p(arr)
        im = ax.imshow(disp, cmap="Blues", aspect="auto", norm=mcolors.Normalize(vmin=0, vmax=float(disp.max() or 1.0)))
        cbar_label = "log1p(count)"
    else:
        im = ax.imshow(arr, cmap="Blues", aspect="auto")
        cbar_label = "proportion" if normalize else "count"
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(cbar_label)

    fmt = "%.2f" if normalize else "%d"
    disp_for_thr = arr if normalize else (np.log1p(arr) if log_color else arr)
    thr = (disp_for_thr.max() / 2.0) if disp_for_thr.size and disp_for_thr.max() > 0 else 0.5
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            tval = disp_for_thr[i, j]
            ax.text(
                j,
                i,
                fmt % val,
                ha="center",
                va="center",
                color="white" if tval > thr else "black",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_per_class(rows: list[dict[str, str]], out_path: str, subtitle: str = "") -> None:
    """上：P/R/F1（0~1）。下：support=该类真值样本数。

    说明：本图**不画统计置信区间**（bootstrap CI）；若某类 n>0 但三根柱都为 0，
    表示该类的 TP=0（模型从未把任何样本预测为该类，或从未在该类上判对），不是「缺数据」。
    """
    import matplotlib.pyplot as plt
    import numpy as np

    classes = [r["class"] for r in rows]
    precision = [float(r["precision"]) for r in rows]
    recall = [float(r["recall"]) for r in rows]
    f1 = [float(r["f1"]) for r in rows]
    support = [int(r["support"]) for r in rows]
    tps = [int(float(r.get("tp", 0))) for r in rows]

    x = np.arange(len(classes))
    w = 0.27
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(max(6, len(classes) * 1.1), 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.12},
    )
    ax1.bar(x - w, precision, width=w, label="precision", color="#4C78A8")
    ax1.bar(x, recall, width=w, label="recall", color="#F58518")
    ax1.bar(x + w, f1, width=w, label="F1", color="#54A24B")
    for i, (s, tp) in enumerate(zip(support, tps)):
        ax1.text(i, 0.02, "n=%d TP=%d" % (s, tp), ha="center", va="bottom", fontsize=7, color="gray")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("P / R / F1")
    title = "Per-class metrics (no bootstrap CI)"
    if subtitle:
        title += "\n" + subtitle
    ax1.set_title(title)
    ax1.legend(loc="upper right", fontsize=8)

    ax2.bar(x, support, color="#B279A2", edgecolor="white", label="support (GT count)")
    ax2.set_ylabel("count")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=20)

    fig.text(
        0.5,
        0.01,
        "n>0 但 P=R=F1=0：该类 TP=0（预测从未落在该类，或从未判对）。非置信度缺失。",
        ha="center",
        fontsize=8,
        color="#333333",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_latency(samples: list[dict[str, str]], out_path: str, subtitle: str = "") -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    def _vals(key: str) -> list[float]:
        out = []
        for r in samples:
            try:
                v = float(r.get(key) or "")
            except ValueError:
                continue
            out.append(v)
        return out

    def _clip_display(vals: list[float], p: float = 99.0) -> tuple[list[float], Optional[float]]:
        if not vals:
            return [], None
        cap = float(np.percentile(vals, p))
        clipped = [min(v, cap) for v in vals]
        n_clip = sum(1 for v in vals if v > cap)
        return clipped, (cap if n_clip else None)

    walls = _vals("wall_s")
    visions = _vals("vision_s")
    whispers_all = _vals("whisper_s")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    titles = ("wall_s", "vision_s", "whisper_s")
    raw_sets = (walls, visions, whispers_all)
    for ax, raw, tbase in zip(axes, raw_sets, titles):
        if not raw:
            ax.set_title("%s\n(no data)" % tbase)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        disp, cap = _clip_display(raw, 99.0)
        title = "%s  n=%d" % (tbase, len(raw))
        if cap is not None:
            title += "  (hist clipped at p99=%.2fs, %d tail)" % (cap, sum(1 for v in raw if v > cap))
        ax.hist(disp, bins=16, color="#4C78A8", edgecolor="white")
        ax.axvline(float(np.median(raw)), color="#E45756", linestyle="--", linewidth=1, label="median")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("seconds")
        ax.legend(loc="upper right", fontsize=7)

    if whispers_all and float(np.max(whispers_all)) < 0.02:
        axes[2].clear()
        axes[2].set_axis_off()
        axes[2].text(
            0.5,
            0.55,
            "Whisper ≈ 0 s\n(本批为仅视频路径，\n未走语音转写)",
            ha="center",
            va="center",
            fontsize=11,
            transform=axes[2].transAxes,
        )
        axes[2].text(
            0.5,
            0.15,
            "max(raw)=%.2e s" % float(np.max(whispers_all)),
            ha="center",
            fontsize=9,
            transform=axes[2].transAxes,
        )

    title = "Latency distributions"
    if subtitle:
        title += "  —  " + subtitle
    fig.suptitle(title, y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_unknown_rate(
    mat: list[list[int]],
    rows: list[str],
    cols: list[str],
    out_path: str,
    subtitle: str = "",
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if "unknown" not in cols:
        return
    j_unknown = cols.index("unknown")
    arr = np.array(mat, dtype=float)
    totals = arr.sum(axis=1)
    unknown_counts = arr[:, j_unknown]
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = np.where(totals > 0, unknown_counts / totals, 0.0)

    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 1.1), 4.0))
    x = np.arange(len(rows))
    ax.bar(x, rates, color="#E45756", edgecolor="white")
    for i, (r, c, t) in enumerate(zip(rates, unknown_counts, totals)):
        ax.text(i, r + 0.01, "%d/%d" % (int(c), int(t)), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(rows, rotation=20)
    ax.set_ylim(0, max(1.0, float(rates.max()) * 1.2) if rates.size else 1.0)
    ax.set_ylabel("unknown rate")
    title = "Unknown-prediction rate per true class"
    if subtitle:
        title += "\n" + subtitle
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_report(
    eval_dir: str,
    summary: dict,
    per_class: list[dict[str, str]],
    rows_labels: list[str],
    cols: list[str],
    mat: list[list[int]],
    files: dict[str, str],
) -> None:
    run_name = os.path.basename(os.path.normpath(eval_dir))
    lines: list[str] = []
    lines.append("# Offline emotion evaluation report — `%s`\n" % run_name)
    lines.append(
        "- N=%d  rows_with_truth=%d  errors=%d  unknown=%d  resolved=%d"
        % (
            summary.get("rows_total", 0),
            summary.get("rows_with_truth", 0),
            summary.get("rows_with_error", 0),
            summary.get("predictions_unknown", 0),
            summary.get("predictions_resolved", 0),
        )
    )
    lines.append("- CSV: `%s`" % summary.get("csv", ""))
    lines.append("- Dataset: `%s`  | text source: `%s`" % (summary.get("dataset_arg"), summary.get("text_source")))
    lines.append("- Rows: total=%d, with_truth=%d, errors=%d" % (summary.get("rows_total", 0), summary.get("rows_with_truth", 0), summary.get("rows_with_error", 0)))
    lines.append("- Predictions: resolved=%d, unknown=%d" % (summary.get("predictions_resolved", 0), summary.get("predictions_unknown", 0)))
    lines.append("")
    lines.append("## Headline metrics")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append("| accuracy (unknown=wrong) | %.4f |" % summary.get("accuracy_eval_including_unknown_as_wrong", 0.0))
    lines.append("| accuracy (resolved-only) | %.4f |" % summary.get("accuracy_resolved_only", 0.0))
    lines.append("| macro precision | %.4f |" % summary.get("macro_precision", 0.0))
    lines.append("| macro recall | %.4f |" % summary.get("macro_recall", 0.0))
    lines.append("| macro F1 | %.4f |" % summary.get("macro_f1", 0.0))
    lines.append("| weighted F1 | %.4f |" % summary.get("weighted_f1", 0.0))
    lines.append("| Cohen kappa (resolved) | %.4f |" % summary.get("cohen_kappa_resolved", 0.0))
    lines.append("")
    lines.append("## How to read the figures")
    lines.append("- **Confusion matrix (count)**：色标使用 `log1p(count)`，避免某一格特别大导致其它格全成浅色；格子上的数字仍是**原始计数**。")
    lines.append("- **Per-class P/R/F1**：**没有**画 bootstrap 置信区间；`support` 是**真值中该类样本数**。")
    lines.append("  若 `n>0` 但 precision/recall/F1 全为 0，表示 **TP=0**（模型从未把任何样本预测为该类，或从未在该类上判对），不是「缺少置信度」。")
    lines.append("- **Latency**：直方图对 wall/vision 在 **p99 处截断**显示尾部，避免单条 150s 拖垮整张图；median 用红线标出。Whisper 若全接近 0，第三格为说明（视频-only 批跑）。")
    lines.append("")

    lat = summary.get("latency", {})
    lines.append("## Latency (seconds)")
    lines.append("| split | count | mean | median | p90 | p95 | max |")
    lines.append("|---|---|---|---|---|---|---|")
    for k in ("wall_s", "vision_s", "whisper_s"):
        s = lat.get(k) or {}
        def _fmt(v: Optional[float]) -> str:
            return "%.3f" % v if isinstance(v, (int, float)) else "-"
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            k,
            s.get("count", 0),
            _fmt(s.get("mean")),
            _fmt(s.get("median")),
            _fmt(s.get("p90")),
            _fmt(s.get("p95")),
            _fmt(s.get("max")),
        ))
    lines.append("")

    lines.append("## Per-class")
    lines.append("| class | precision | recall | F1 | support |")
    lines.append("|---|---|---|---|---|")
    for r in per_class:
        lines.append("| %s | %.4f | %.4f | %.4f | %s |" % (
            r["class"], float(r["precision"]), float(r["recall"]), float(r["f1"]), r["support"]
        ))
    lines.append("")

    lines.append("## Confusion matrix (counts)")
    header = "| true \\\\ pred | " + " | ".join(cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"
    lines.append(header)
    lines.append(sep)
    for label, row in zip(rows_labels, mat):
        lines.append("| %s | %s |" % (label, " | ".join(str(x) for x in row)))
    lines.append("")

    lines.append("## Figures")
    for caption, fname in files.items():
        if fname:
            lines.append("![%s](%s)" % (caption, os.path.basename(fname)))
            lines.append("")

    out_path = os.path.join(eval_dir, "report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[viz] wrote", out_path)


def _read_confusion_path(csv_path: str) -> tuple[list[str], list[str], list[list[int]]]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = header[1:]
        rows_labels: list[str] = []
        mat: list[list[int]] = []
        for row in reader:
            if not row or not row[0]:
                continue
            rows_labels.append(row[0])
            mat.append([int(x) for x in row[1 : 1 + len(cols)]])
    return rows_labels, cols, mat


def _per_class_from_matrix(rows_labels: list[str], cols: list[str], mat: list[list[int]]) -> list[dict[str, str]]:
    """与 eval_emotion_labels._per_class_prf 相同定义（列 totals 不含 unknown）。"""
    labels = [c for c in cols if c != "unknown"]
    mat_dict: dict[str, dict[str, int]] = {}
    for i, t in enumerate(rows_labels):
        mat_dict[t] = {cols[j]: mat[i][j] for j in range(len(cols))}
    col_totals = {c: 0 for c in labels}
    for t in rows_labels:
        for c in labels:
            col_totals[c] += mat_dict[t][c]
    out: list[dict[str, str]] = []
    for cls in rows_labels:
        tp = mat_dict[cls][cls]
        fn = sum(mat_dict[cls][c] for c in labels if c != cls) + mat_dict[cls].get("unknown", 0)
        fp = col_totals[cls] - tp
        support = tp + fn
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        out.append(
            {
                "class": cls,
                "precision": "%.6f" % p,
                "recall": "%.6f" % r,
                "f1": "%.6f" % f1,
                "support": str(int(support)),
                "tp": str(int(tp)),
                "fp": str(int(fp)),
                "fn": str(int(fn)),
            }
        )
    return out


def _summary_from_matrix(
    rows_labels: list[str],
    cols: list[str],
    mat: list[list[int]],
    per_class: list[dict[str, str]],
) -> dict:
    import statistics

    tot = sum(sum(row) for row in mat)
    unk = 0
    if "unknown" in cols:
        j = cols.index("unknown")
        unk = sum(row[j] for row in mat)
    diag = 0
    for i, t in enumerate(rows_labels):
        diag += mat[i][cols.index(t)]
    f1s = [float(r["f1"]) for r in per_class]
    ps = [float(r["precision"]) for r in per_class]
    rs = [float(r["recall"]) for r in per_class]
    sups = [float(r["support"]) for r in per_class]
    tot_sup = sum(sups)
    w_f1 = sum(float(r["f1"]) * float(r["support"]) for r in per_class) / tot_sup if tot_sup else 0.0
    return {
        "csv": "",
        "out_dir": "",
        "dataset_arg": "example_matrix",
        "text_source": "n/a",
        "unknown_as_wrong": True,
        "rows_total": tot,
        "rows_with_truth": tot,
        "rows_with_error": 0,
        "predictions_resolved": tot - unk,
        "predictions_unknown": unk,
        "accuracy_eval_including_unknown_as_wrong": (diag / tot) if tot else 0.0,
        "accuracy_resolved_only": (diag / (tot - unk)) if (tot - unk) else 0.0,
        "macro_precision": statistics.fmean(ps) if ps else 0.0,
        "macro_recall": statistics.fmean(rs) if rs else 0.0,
        "macro_f1": statistics.fmean(f1s) if f1s else 0.0,
        "weighted_f1": w_f1,
        "cohen_kappa_resolved": 0.0,
        "latency": {
            "wall_s": {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None},
            "vision_s": {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None},
            "whisper_s": {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None},
        },
        "labels": list(rows_labels),
    }


def _read_per_class_path(csv_path: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out.append(row)
    return out


def _run_plots(
    eval_dir: str,
    summary: dict,
    rows_labels: list[str],
    cols: list[str],
    mat: list[list[int]],
    per_class: list[dict[str, str]],
    samples: list[dict[str, str]],
    *,
    write_report: bool = True,
    minimal: bool = False,
) -> None:
    os.makedirs(eval_dir, exist_ok=True)
    cm_path = os.path.join(eval_dir, "confusion_matrix.png")
    cm_norm_path = os.path.join(eval_dir, "confusion_matrix_norm.png")
    pc_path = os.path.join(eval_dir, "per_class_f1.png")
    lat_path = os.path.join(eval_dir, "latency_distribution.png")
    ur_path = os.path.join(eval_dir, "unknown_rate_per_class.png")

    run_name = os.path.basename(os.path.normpath(eval_dir))
    subtitle = (
        "%s  |  N=%d  acc=%.3f  macroF1=%.3f  unknown=%d/%d"
        % (
            run_name,
            summary.get("rows_total", 0),
            summary.get("accuracy_eval_including_unknown_as_wrong", 0.0),
            summary.get("macro_f1", 0.0),
            summary.get("predictions_unknown", 0),
            summary.get("rows_with_truth", 0),
        )
    )

    _plot_confusion(mat, rows_labels, cols, cm_path, normalize=False, subtitle=subtitle, log_color=True)
    _plot_per_class(per_class, pc_path, subtitle=subtitle)
    saved: list[str] = [cm_path, pc_path]
    if not minimal:
        _plot_confusion(mat, rows_labels, cols, cm_norm_path, normalize=True, subtitle=subtitle, log_color=False)
        _plot_latency(samples, lat_path, subtitle=run_name)
        _plot_unknown_rate(mat, rows_labels, cols, ur_path, subtitle=subtitle)
        saved.extend([cm_norm_path, ur_path, lat_path])

    if write_report and not minimal:
        summary["out_dir"] = eval_dir
        _write_report(
            eval_dir,
            summary,
            per_class,
            rows_labels,
            cols,
            mat,
            {
                "Confusion matrix (log-color for counts)": cm_path,
                "Confusion matrix (normalized)": cm_norm_path,
                "Per-class P/R/F1": pc_path,
                "Unknown rate per class": ur_path,
                "Latency distribution": lat_path,
            },
        )

    print("[viz] saved:")
    for p in saved:
        print("  -", p)
    if write_report and not minimal:
        print("[viz] wrote", os.path.join(eval_dir, "report.md"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize eval output or a standalone confusion_matrix.csv")
    ap.add_argument("--eval-dir", default="", help="eval_emotion_labels 输出目录（与 --matrix-csv 二选一）")
    ap.add_argument("--matrix-csv", default="", help="独立混淆矩阵 CSV（需同时指定 --out-dir）")
    ap.add_argument("--out-dir", default="", help="与 --matrix-csv 配合：图表输出目录")
    ap.add_argument(
        "--per-class-csv",
        default="",
        help="可选：显式指定 per_class.csv；省略则由矩阵按 eval 公式重算",
    )
    ap.add_argument(
        "--minimal",
        action="store_true",
        help="只生成 confusion_matrix.png 与 per_class_f1.png 两张图",
    )
    args = ap.parse_args()

    if not _try_import_matplotlib():
        raise SystemExit(2)

    use_matrix = bool(args.matrix_csv.strip())
    use_eval = bool(args.eval_dir.strip())
    if use_matrix == use_eval:
        raise SystemExit("请只指定其一: --eval-dir 或 (--matrix-csv 与 --out-dir)")
    if use_matrix and not args.out_dir.strip():
        raise SystemExit("--matrix-csv 时必须提供 --out-dir")

    if use_eval:
        eval_dir = os.path.abspath(args.eval_dir)
        if not os.path.isdir(eval_dir):
            raise SystemExit("eval-dir 不存在: %s" % eval_dir)
        summary = _read_summary(eval_dir)
        rows_labels, cols, mat = _read_confusion(eval_dir)
        per_class = _read_per_class(eval_dir)
        samples = _read_samples(eval_dir)
        _run_plots(
            eval_dir,
            summary,
            rows_labels,
            cols,
            mat,
            per_class,
            samples,
            write_report=True,
            minimal=args.minimal,
        )
        return

    mpath = os.path.abspath(args.matrix_csv)
    if not os.path.isfile(mpath):
        raise SystemExit("matrix-csv 不存在: %s" % mpath)
    out_dir = os.path.abspath(args.out_dir)
    rows_labels, cols, mat = _read_confusion_path(mpath)
    if args.per_class_csv.strip():
        per_class = _read_per_class_path(os.path.abspath(args.per_class_csv))
    else:
        per_class = _per_class_from_matrix(rows_labels, cols, mat)
    summary = _summary_from_matrix(rows_labels, cols, mat, per_class)
    samples: list[dict[str, str]] = []
    _run_plots(
        out_dir,
        summary,
        rows_labels,
        cols,
        mat,
        per_class,
        samples,
        write_report=not args.minimal,
        minimal=args.minimal,
    )


if __name__ == "__main__":
    main()
