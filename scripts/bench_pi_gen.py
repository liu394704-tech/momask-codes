"""树莓派端侧「文本 -> 动作」推理耗时 + 动作质量基准。

在树莓派上运行：
  cd /home/cat/RBM-project/momask-codes
  source venv_inference/bin/activate
  python scripts/bench_pi_gen.py \\
      --text_path assets/bench_prompts_100.txt \\
      --out_csv experiment/pi_bench/bench_results.csv \\
      --time_steps 18

计时：
  gen_s / recover_s / total_s；前 --warmup 条不计入统计。

动作质量（无 GT，基于生成关节的可复现代理指标）：
  mean_speed   平均关节速度（越大动作越剧烈）
  max_root_disp 根节点最大位移
  mean_jerk    平均加加速度（越小通常越平滑）
  valid_ratio  有限数值关节比例（异常检测）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from argparse import Namespace
from collections import defaultdict
from os.path import join as pjoin

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils.motion_process import recover_from_ric
from gen_t2m import load_vq_model, load_trans_model, load_res_model, load_len_estimator


def build_opts(args, device):
    opt = Namespace()
    opt.checkpoints_dir = args.checkpoints_dir
    opt.dataset_name = args.dataset_name
    opt.name = args.name
    opt.res_name = args.res_name
    opt.device = device
    opt.gpu_id = -1

    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_opt = get_opt(pjoin(root_dir, "opt.txt"), device)

    vq_opt = get_opt(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, "opt.txt"), device)
    vq_opt.dim_pose = 251 if opt.dataset_name == "kit" else 263

    res_opt = get_opt(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, "opt.txt"), device)
    return opt, model_opt, vq_opt, res_opt


def load_all(opt, model_opt, vq_opt, res_opt):
    vq_model, vq_opt = load_vq_model(vq_opt)
    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    res_model = load_res_model(res_opt, vq_opt, opt)
    assert res_opt.vq_name == model_opt.vq_name

    t2m_transformer = load_trans_model(model_opt, opt, "latest.tar")
    length_estimator = load_len_estimator(model_opt)

    for m in (vq_model, res_model, t2m_transformer, length_estimator):
        m.eval()
        m.to(opt.device)
    return vq_model, res_model, t2m_transformer, length_estimator


def read_prompts(path):
    """支持: 'text' | 'emotion|text' | 'text #len' | 'emotion|text #len'"""
    prompts, lengths, emotions = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            emotion = "unknown"
            if "|" in line:
                emotion, line = line.split("|", 1)
                emotion = emotion.strip().lower()
                line = line.strip()
            fixed_len = 0
            if "#" in line:
                text, _, ln = line.partition("#")
                line = text.strip()
                if ln.strip().isdigit():
                    fixed_len = int(ln.strip())
            prompts.append(line)
            lengths.append(fixed_len)
            emotions.append(emotion)
    return prompts, lengths, emotions


def motion_quality_metrics(joints: np.ndarray) -> dict:
    """joints: (T, J, 3)"""
    if joints is None or joints.size == 0 or joints.shape[0] < 2:
        return {
            "mean_speed": 0.0,
            "max_root_disp": 0.0,
            "mean_jerk": 0.0,
            "valid_ratio": 0.0,
        }
    finite = np.isfinite(joints)
    valid_ratio = float(finite.mean())
    x = np.nan_to_num(joints, nan=0.0, posinf=0.0, neginf=0.0)
    vel = np.diff(x, axis=0)
    speed = np.linalg.norm(vel, axis=-1)  # (T-1, J)
    mean_speed = float(speed.mean())
    root = x[:, 0, :]
    max_root_disp = float(np.linalg.norm(root - root[0], axis=-1).max())
    if x.shape[0] >= 4:
        acc = np.diff(vel, axis=0)
        jerk = np.diff(acc, axis=0)
        mean_jerk = float(np.linalg.norm(jerk, axis=-1).mean())
    else:
        mean_jerk = 0.0
    return {
        "mean_speed": round(mean_speed, 6),
        "max_root_disp": round(max_root_disp, 6),
        "mean_jerk": round(mean_jerk, 6),
        "valid_ratio": round(valid_ratio, 6),
    }


@torch.no_grad()
def gen_one(models, opt, args, caption, fixed_len, mean, std):
    vq_model, res_model, t2m_transformer, length_estimator = models

    if fixed_len and fixed_len >= 20:
        token_lens = torch.LongTensor([fixed_len // 4]).to(opt.device).long()
    else:
        text_emb = t2m_transformer.encode_text([caption])
        pred_dis = length_estimator(text_emb)
        probs = F.softmax(pred_dis, dim=-1)
        token_lens = Categorical(probs).sample()
    m_length = int(token_lens[0].item()) * 4

    t0 = time.perf_counter()
    mids = t2m_transformer.generate(
        [caption], token_lens,
        timesteps=args.time_steps,
        cond_scale=args.cond_scale,
        temperature=args.temperature,
        topk_filter_thres=args.topkr,
        gsample=False,
    )
    mids = res_model.generate(mids, [caption], token_lens, temperature=1, cond_scale=5)
    pred_motions = vq_model.forward_decoder(mids)
    pred_motions = pred_motions.detach().cpu().numpy()
    gen_s = time.perf_counter() - t0

    recover_s = 0.0
    joints = None
    feat = None
    if not args.skip_recover:
        data = pred_motions * std + mean
        feat = data[0][:max(m_length, 1)].astype(np.float32)
        t1 = time.perf_counter()
        joints = recover_from_ric(torch.from_numpy(feat).float(), 22).numpy().astype(np.float32)
        recover_s = time.perf_counter() - t1

    q = motion_quality_metrics(joints)
    return gen_s, recover_s, m_length, q, joints, feat


def summarize(name, values):
    if not values:
        return f"{name}: (no data)"
    vals = sorted(values)
    p95 = vals[min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))]
    return (
        f"{name}: mean={statistics.mean(vals):.3f}s  median={statistics.median(vals):.3f}s  "
        f"p95={p95:.3f}s  min={vals[0]:.3f}s  max={vals[-1]:.3f}s"
    )


def main():
    ap = argparse.ArgumentParser(description="树莓派端侧文本到动作耗时/质量基准")
    ap.add_argument("--text_path", default="assets/bench_prompts_100.txt")
    ap.add_argument("--out_csv", default="experiment/pi_bench/bench_results.csv")
    ap.add_argument("--checkpoints_dir", default="./checkpoints")
    ap.add_argument("--dataset_name", default="t2m")
    ap.add_argument("--name", default="t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns")
    ap.add_argument("--res_name", default="tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw")
    ap.add_argument("--vq_meta", default="rvq_nq6_dc512_nc512_noshare_qdp0.2")
    ap.add_argument("--time_steps", type=int, default=18, help="与 gen_t2m 默认一致")
    ap.add_argument("--cond_scale", type=float, default=4.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--topkr", type=float, default=0.9)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_recover", action="store_true")
    ap.add_argument("--save_joints", action="store_true",
                    help="保存每条 (T,22,3) 关节坐标 npy（不渲染）")
    ap.add_argument("--save_features", action="store_true",
                    help="同时保存反归一化后的 263 维运动特征 npy")
    ap.add_argument("--joints_dir", default="",
                    help="关节输出目录；默认 out_csv 同级 joints/")
    ap.add_argument("--seed", type=int, default=10107)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    fixseed(args.seed)
    device = torch.device("cpu")

    print(f"[bench] device=CPU threads={args.threads} time_steps={args.time_steps} skip_recover={args.skip_recover}", flush=True)
    print("[bench] 加载模型中（首次较慢）...", flush=True)
    t_load0 = time.perf_counter()
    opt, model_opt, vq_opt, res_opt = build_opts(args, device)
    models = load_all(opt, model_opt, vq_opt, res_opt)
    load_s = time.perf_counter() - t_load0
    print(f"[bench] 模型加载完成，用时 {load_s:.1f}s", flush=True)

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, args.vq_meta, "meta", "mean.npy"))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, args.vq_meta, "meta", "std.npy"))

    prompts, lengths, emotions = read_prompts(args.text_path)
    if args.limit > 0:
        prompts = prompts[:args.limit]
        lengths = lengths[:args.limit]
        emotions = emotions[:args.limit]
    print(f"[bench] 共 {len(prompts)} 条 prompt，其中前 {args.warmup} 条为预热", flush=True)
    emo_counts = defaultdict(int)
    for e in emotions:
        emo_counts[e] += 1
    print("[bench] 情绪分布: " + ", ".join(f"{k}={v}" for k, v in sorted(emo_counts.items())), flush=True)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    joints_dir = args.joints_dir or pjoin(os.path.dirname(args.out_csv) or ".", "joints")
    feat_dir = pjoin(os.path.dirname(args.out_csv) or ".", "features")
    if args.save_joints:
        os.makedirs(joints_dir, exist_ok=True)
        print(f"[bench] 关节坐标将保存到: {joints_dir}", flush=True)
    if args.save_features:
        os.makedirs(feat_dir, exist_ok=True)
        print(f"[bench] 运动特征将保存到: {feat_dir}", flush=True)

    fieldnames = [
        "idx", "phase", "emotion", "prompt", "motion_length",
        "gen_s", "recover_s", "total_s",
        "mean_speed", "max_root_disp", "mean_jerk", "valid_ratio",
        "joints_npy", "features_npy",
    ]
    rows = []
    gen_list, recover_list, total_list = [], [], []
    by_emotion = defaultdict(lambda: {"gen": [], "total": [], "speed": [], "jerk": [], "valid": []})

    for i, (caption, flen, emotion) in enumerate(zip(prompts, lengths, emotions)):
        gen_s, recover_s, m_len, q, joints, feat = gen_one(models, opt, args, caption, flen, mean, std)
        total_s = gen_s + recover_s
        is_warm = i < args.warmup
        tag = "warmup" if is_warm else "count"
        joints_path = ""
        feat_path = ""
        if args.save_joints and joints is not None:
            joints_path = pjoin(joints_dir, f"{i:03d}_{emotion}.npy")
            np.save(joints_path, joints)
        if args.save_features and feat is not None:
            feat_path = pjoin(feat_dir, f"{i:03d}_{emotion}_feat263.npy")
            np.save(feat_path, feat)
        row = {
            "idx": i,
            "phase": tag,
            "emotion": emotion,
            "prompt": caption,
            "motion_length": m_len,
            "gen_s": round(gen_s, 4),
            "recover_s": round(recover_s, 4),
            "total_s": round(total_s, 4),
            **q,
            "joints_npy": joints_path,
            "features_npy": feat_path,
        }
        rows.append(row)
        if not is_warm:
            gen_list.append(gen_s)
            recover_list.append(recover_s)
            total_list.append(total_s)
            by_emotion[emotion]["gen"].append(gen_s)
            by_emotion[emotion]["total"].append(total_s)
            by_emotion[emotion]["speed"].append(q["mean_speed"])
            by_emotion[emotion]["jerk"].append(q["mean_jerk"])
            by_emotion[emotion]["valid"].append(q["valid_ratio"])

        # 增量写 CSV，便于中途查看
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        done = i + 1
        counted = len(total_list)
        avg = statistics.mean(total_list) if total_list else 0.0
        print(
            f"  [{done:3d}/{len(prompts)}] ({tag}) emo={emotion:9s} len={m_len:3d} "
            f"gen={gen_s:.3f}s total={total_s:.3f}s "
            f"speed={q['mean_speed']:.4f} jerk={q['mean_jerk']:.4f} valid={q['valid_ratio']:.3f} "
            f"| avg_total={avg:.3f}s (n={counted})  {caption[:36]}",
            flush=True,
        )

    emotion_summary = {}
    for emo, d in sorted(by_emotion.items()):
        emotion_summary[emo] = {
            "n": len(d["total"]),
            "gen_mean_s": round(statistics.mean(d["gen"]), 4) if d["gen"] else None,
            "total_mean_s": round(statistics.mean(d["total"]), 4) if d["total"] else None,
            "mean_speed": round(statistics.mean(d["speed"]), 6) if d["speed"] else None,
            "mean_jerk": round(statistics.mean(d["jerk"]), 6) if d["jerk"] else None,
            "valid_ratio_mean": round(statistics.mean(d["valid"]), 6) if d["valid"] else None,
        }

    summary = {
        "device": "cpu",
        "threads": args.threads,
        "time_steps": args.time_steps,
        "skip_recover": args.skip_recover,
        "model_load_s": round(load_s, 2),
        "n_prompts_total": len(prompts),
        "n_warmup": args.warmup,
        "n_counted": len(total_list),
        "gen_mean_s": round(statistics.mean(gen_list), 4) if gen_list else None,
        "recover_mean_s": round(statistics.mean(recover_list), 4) if recover_list else None,
        "total_mean_s": round(statistics.mean(total_list), 4) if total_list else None,
        "total_median_s": round(statistics.median(total_list), 4) if total_list else None,
        "total_p95_s": round(sorted(total_list)[min(len(total_list) - 1, int(round(0.95 * (len(total_list) - 1))))], 4) if total_list else None,
        "by_emotion": emotion_summary,
    }
    summary_path = os.path.splitext(args.out_csv)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n==== 树莓派端侧耗时统计（已排除预热） ====", flush=True)
    print(summarize("gen_s    (神经网络生成)", gen_list), flush=True)
    if not args.skip_recover:
        print(summarize("recover_s(关节恢复)", recover_list), flush=True)
    print(summarize("total_s  (端到端单条)", total_list), flush=True)
    print(f"模型加载一次: {load_s:.1f}s（不计入单条平均）", flush=True)
    print("\n==== 按情绪分层耗时 ====", flush=True)
    for emo, s in emotion_summary.items():
        print(
            f"  {emo:9s} n={s['n']:2d} total_mean={s['total_mean_s']}s "
            f"speed={s['mean_speed']} jerk={s['mean_jerk']} valid={s['valid_ratio_mean']}",
            flush=True,
        )
    print(f"\n明细 CSV : {args.out_csv}", flush=True)
    print(f"摘要 JSON: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
