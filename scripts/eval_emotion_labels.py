#!/usr/bin/env python3
"""离线评测：读 batch_offline_mllm.py 写出的 CSV，对齐标准数据集标签后计算指标。

输出到 --out-dir：
  - summary.json               # 总览：N、N_eval、accuracy、macro F1、weighted F1、Cohen kappa、延迟分位
  - per_class.csv              # 每类 precision / recall / F1 / support
  - confusion_matrix.csv       # 行=真实, 列=预测（含 "unknown" 列汇总无法解析的预测）
  - sample_predictions.csv     # 单样本：路径、真实、预测原文、解析标签、wall_s、是否正确

用法:
  python scripts/eval_emotion_labels.py \
      --csv experiment/offline_runs/ravdess_run1.csv \
      --dataset ravdess \
      --out-dir experiment/offline_runs/ravdess_run1_eval

支持 --dataset: ravdess / cremad / auto（按视频路径含 "ravdess"/"crema" 判定）。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from emotion_label_map import (  # noqa: E402
    CANON_LABELS,
    detect_dataset_from_path,
    map_text_to_label,
    parse_label_from_filename,
)


def _percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def _latency_stats(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _build_confusion(
    pairs: list[tuple[str, Optional[str]]],
) -> tuple[dict[str, dict[str, int]], list[str]]:
    labels = list(CANON_LABELS)
    cols = labels + ["unknown"]
    mat: dict[str, dict[str, int]] = {t: {c: 0 for c in cols} for t in labels}
    for true, pred in pairs:
        if true not in mat:
            continue
        col = pred if (pred in labels) else "unknown"
        mat[true][col] += 1
    return mat, cols


def _per_class_prf(
    mat: dict[str, dict[str, int]], labels: list[str]
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    col_totals = {c: 0 for c in labels}
    for t in labels:
        for c in labels:
            col_totals[c] += mat[t][c]
    for cls in labels:
        tp = mat[cls][cls]
        fn = sum(mat[cls][c] for c in labels if c != cls) + mat[cls].get("unknown", 0)
        fp = col_totals[cls] - tp
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
        }
    return out


def _macro_weighted(prf: dict[str, dict[str, float]]) -> tuple[float, float, float, float]:
    classes = [c for c, m in prf.items() if m["support"] > 0]
    if not classes:
        return 0.0, 0.0, 0.0, 0.0
    macro_p = statistics.fmean(prf[c]["precision"] for c in classes)
    macro_r = statistics.fmean(prf[c]["recall"] for c in classes)
    macro_f1 = statistics.fmean(prf[c]["f1"] for c in classes)
    total_support = sum(prf[c]["support"] for c in classes)
    weighted_f1 = (
        sum(prf[c]["f1"] * prf[c]["support"] for c in classes) / total_support
        if total_support
        else 0.0
    )
    return macro_p, macro_r, macro_f1, weighted_f1


def _cohen_kappa(pairs: list[tuple[str, str]], labels: list[str]) -> float:
    if not pairs:
        return 0.0
    n = len(pairs)
    agree = sum(1 for t, p in pairs if t == p)
    p_o = agree / n
    t_counts = {l: 0 for l in labels}
    p_counts = {l: 0 for l in labels}
    for t, p in pairs:
        if t in t_counts:
            t_counts[t] += 1
        if p in p_counts:
            p_counts[p] += 1
    p_e = sum((t_counts[l] / n) * (p_counts[l] / n) for l in labels)
    if p_e >= 1.0:
        return 0.0
    return (p_o - p_e) / (1.0 - p_e)


def _resolve_dataset(arg_dataset: str, video_relpath: str) -> Optional[str]:
    ds = arg_dataset.strip().lower()
    if ds in ("ravdess", "cremad", "crema-d", "crema_d"):
        return "cremad" if ds.startswith("crema") else "ravdess"
    if ds in ("", "auto"):
        return detect_dataset_from_path(video_relpath)
    return None


def _safe_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline evaluation: align predictions to dataset labels")
    ap.add_argument("--csv", required=True, help="batch_offline_mllm 写出的 CSV")
    ap.add_argument(
        "--dataset",
        default="auto",
        help="ravdess / cremad / auto（依据 video_relpath 推断）",
    )
    ap.add_argument("--out-dir", required=True, help="输出目录（自动创建）")
    ap.add_argument(
        "--text-source",
        default="emotion",
        choices=("emotion", "user_action", "robot_reaction", "all"),
        help="从哪一列文本解析预测标签；all=三列拼接",
    )
    ap.add_argument(
        "--unknown-as-wrong",
        action="store_true",
        default=True,
        help="无法解析的预测计入错误（默认开），关闭请用 --keep-unknown-out",
    )
    ap.add_argument(
        "--keep-unknown-out",
        action="store_true",
        help="把无法解析的预测从指标中剔除（仍写入 sample_predictions.csv）",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        raise SystemExit("CSV 不存在: %s" % args.csv)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    pairs_for_metrics: list[tuple[str, str]] = []
    pairs_full: list[tuple[str, Optional[str]]] = []
    sample_rows: list[dict[str, str]] = []
    wall_s_values: list[float] = []
    vision_s_values: list[float] = []
    whisper_s_values: list[float] = []
    errors = 0
    n_rows = 0
    n_with_label = 0

    with open(args.csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_rows += 1
            err = (row.get("error") or "").strip()
            if err:
                errors += 1
            video_rel = row.get("video_relpath") or ""
            ds = _resolve_dataset(args.dataset, video_rel)
            true_label = parse_label_from_filename(video_rel, ds or "") if ds else None
            if true_label is None:
                continue
            n_with_label += 1

            if args.text_source == "all":
                text = " ".join(
                    [row.get("user_action") or "", row.get("emotion") or "", row.get("robot_reaction") or ""]
                ).strip()
            else:
                text = (row.get(args.text_source) or "").strip()
            pred_label = map_text_to_label(text) if text else None

            ws = _safe_float(row.get("whisper_s") or "")
            vs = _safe_float(row.get("vision_s") or "")
            wall = _safe_float(row.get("wall_s") or "")
            if ws is not None:
                whisper_s_values.append(ws)
            if vs is not None:
                vision_s_values.append(vs)
            if wall is not None:
                wall_s_values.append(wall)

            pairs_full.append((true_label, pred_label))
            if pred_label is not None:
                pairs_for_metrics.append((true_label, pred_label))
            elif not args.keep_unknown_out and args.unknown_as_wrong:
                pairs_for_metrics.append((true_label, "__unknown__"))

            sample_rows.append(
                {
                    "video_relpath": video_rel,
                    "dataset": ds or "",
                    "true_label": true_label,
                    "pred_text": text.replace("\n", " ")[:500],
                    "pred_label": pred_label or "",
                    "wall_s": row.get("wall_s") or "",
                    "vision_s": row.get("vision_s") or "",
                    "whisper_s": row.get("whisper_s") or "",
                    "is_correct": "1" if (pred_label is not None and pred_label == true_label) else "0",
                    "error": err,
                }
            )

    mat, cols = _build_confusion(pairs_full)
    prf = _per_class_prf(mat, list(CANON_LABELS))
    macro_p, macro_r, macro_f1, weighted_f1 = _macro_weighted(prf)

    valid_pairs = [(t, p) for t, p in pairs_for_metrics if p != "__unknown__"]
    accuracy_eval = (
        sum(1 for t, p in pairs_for_metrics if t == p) / len(pairs_for_metrics)
        if pairs_for_metrics
        else 0.0
    )
    accuracy_resolved = (
        sum(1 for t, p in valid_pairs if t == p) / len(valid_pairs) if valid_pairs else 0.0
    )
    kappa = _cohen_kappa(valid_pairs, list(CANON_LABELS))

    cm_path = os.path.join(out_dir, "confusion_matrix.csv")
    with open(cm_path, "w", newline="", encoding="utf-8-sig") as cf:
        w = csv.writer(cf)
        w.writerow(["true\\pred"] + cols)
        for t in CANON_LABELS:
            w.writerow([t] + [mat[t][c] for c in cols])

    pc_path = os.path.join(out_dir, "per_class.csv")
    with open(pc_path, "w", newline="", encoding="utf-8-sig") as cf:
        w = csv.writer(cf)
        w.writerow(["class", "precision", "recall", "f1", "support", "tp", "fp", "fn"])
        for cls in CANON_LABELS:
            m = prf[cls]
            w.writerow(
                [
                    cls,
                    "%.6f" % m["precision"],
                    "%.6f" % m["recall"],
                    "%.6f" % m["f1"],
                    int(m["support"]),
                    int(m["tp"]),
                    int(m["fp"]),
                    int(m["fn"]),
                ]
            )

    sp_path = os.path.join(out_dir, "sample_predictions.csv")
    with open(sp_path, "w", newline="", encoding="utf-8-sig") as cf:
        fieldnames = [
            "video_relpath",
            "dataset",
            "true_label",
            "pred_label",
            "is_correct",
            "pred_text",
            "wall_s",
            "vision_s",
            "whisper_s",
            "error",
        ]
        w = csv.DictWriter(cf, fieldnames=fieldnames)
        w.writeheader()
        for r in sample_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    summary = {
        "csv": os.path.abspath(args.csv),
        "out_dir": out_dir,
        "dataset_arg": args.dataset,
        "text_source": args.text_source,
        "unknown_as_wrong": bool(args.unknown_as_wrong and not args.keep_unknown_out),
        "rows_total": n_rows,
        "rows_with_truth": n_with_label,
        "rows_with_error": errors,
        "predictions_resolved": len(valid_pairs),
        "predictions_unknown": len(pairs_full) - len(valid_pairs),
        "accuracy_eval_including_unknown_as_wrong": accuracy_eval,
        "accuracy_resolved_only": accuracy_resolved,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "cohen_kappa_resolved": kappa,
        "latency": {
            "wall_s": _latency_stats(wall_s_values),
            "vision_s": _latency_stats(vision_s_values),
            "whisper_s": _latency_stats(whisper_s_values),
        },
        "labels": list(CANON_LABELS),
    }
    sm_path = os.path.join(out_dir, "summary.json")
    with open(sm_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[eval] rows_total=%d with_truth=%d errors=%d" % (n_rows, n_with_label, errors))
    print(
        "[eval] accuracy(unknown=wrong)=%.4f  accuracy(resolved-only)=%.4f"
        % (accuracy_eval, accuracy_resolved)
    )
    print("[eval] macro F1=%.4f  weighted F1=%.4f  kappa=%.4f" % (macro_f1, weighted_f1, kappa))
    print("[eval] wrote:")
    print("  - %s" % sm_path)
    print("  - %s" % cm_path)
    print("  - %s" % pc_path)
    print("  - %s" % sp_path)


if __name__ == "__main__":
    main()
