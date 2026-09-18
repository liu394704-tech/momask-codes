#!/usr/bin/env python3
"""In-process MoMask: load weights once, then prompt -> joints.npy (no BVH IK).

This is the interactive path. Do not use gen_t2m.py per request: that script
reloads CLIP/checkpoints and runs IK 100+100 iterations (~20–30 s).
Resident generate on Pi 5 is ~2.2 s (time_steps=18), matching bench_pi_gen.
"""
from __future__ import annotations

import os
import threading
import time
from argparse import Namespace
from os.path import join as pjoin
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

_LOCK = threading.Lock()
_STATE: Optional[Dict[str, Any]] = None
_LOAD_S = 0.0


def _build_opts(device):
    from utils.get_opt import get_opt

    opt = Namespace()
    opt.checkpoints_dir = str(ROOT / "checkpoints")
    opt.dataset_name = "t2m"
    opt.name = "t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns"
    opt.res_name = "tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw"
    opt.device = device
    opt.gpu_id = -1

    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_opt = get_opt(pjoin(root_dir, "opt.txt"), device)
    vq_opt = get_opt(
        pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, "opt.txt"),
        device,
    )
    vq_opt.dim_pose = 263
    res_opt = get_opt(
        pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, "opt.txt"),
        device,
    )
    return opt, model_opt, vq_opt, res_opt


def ensure_loaded(threads: Optional[int] = None) -> float:
    """Load CLIP + MoMask once. Returns load seconds (0 if already loaded)."""
    global _STATE, _LOAD_S
    if _STATE is not None:
        return 0.0
    with _LOCK:
        if _STATE is not None:
            return 0.0
        import numpy as np
        import torch
        from gen_t2m import (
            load_len_estimator,
            load_res_model,
            load_trans_model,
            load_vq_model,
        )
        from utils.fixseed import fixseed

        n_threads = int(threads or os.environ.get("MOMASK_THREADS", "4") or "4")
        torch.set_num_threads(max(1, n_threads))
        fixseed(int(os.environ.get("MOMASK_SEED", "10107") or "10107"))
        device = torch.device("cpu")

        t0 = time.perf_counter()
        opt, model_opt, vq_opt, res_opt = _build_opts(device)
        vq_model, vq_opt = load_vq_model(vq_opt)
        model_opt.num_tokens = vq_opt.nb_code
        model_opt.num_quantizers = vq_opt.num_quantizers
        model_opt.code_dim = vq_opt.code_dim
        res_model = load_res_model(res_opt, vq_opt, opt)
        t2m_transformer = load_trans_model(model_opt, opt, "latest.tar")
        length_estimator = load_len_estimator(model_opt)
        for m in (vq_model, res_model, t2m_transformer, length_estimator):
            m.eval()
            m.to(device)
        vq_meta = "rvq_nq6_dc512_nc512_noshare_qdp0.2"
        mean = np.load(
            pjoin(opt.checkpoints_dir, opt.dataset_name, vq_meta, "meta", "mean.npy")
        )
        std = np.load(
            pjoin(opt.checkpoints_dir, opt.dataset_name, vq_meta, "meta", "std.npy")
        )
        _LOAD_S = time.perf_counter() - t0
        _STATE = {
            "opt": opt,
            "models": (vq_model, res_model, t2m_transformer, length_estimator),
            "mean": mean,
            "std": std,
            "device": device,
        }
        return _LOAD_S


def generate(
    prompt: str,
    out_npy: Path,
    time_steps: int = 18,
    cond_scale: float = 4.0,
    motion_length_hint: int = 0,
) -> Tuple[Path, float, float, int]:
    """Return (joints_path, load_s, gen_s, T). load_s is 0 after the first call."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.distributions.categorical import Categorical
    from utils.motion_process import recover_from_ric

    load_s = ensure_loaded()
    assert _STATE is not None
    vq_model, res_model, t2m_transformer, length_estimator = _STATE["models"]
    mean, std = _STATE["mean"], _STATE["std"]
    device = _STATE["device"]
    caption = (prompt or "").strip()
    if not caption:
        raise ValueError("empty prompt")

    with torch.no_grad():
        if motion_length_hint and motion_length_hint >= 20:
            token_lens = torch.LongTensor([motion_length_hint // 4]).to(device).long()
        else:
            text_emb = t2m_transformer.encode_text([caption])
            pred_dis = length_estimator(text_emb)
            probs = F.softmax(pred_dis, dim=-1)
            token_lens = Categorical(probs).sample()
        m_length = int(token_lens[0].item()) * 4

        t0 = time.perf_counter()
        mids = t2m_transformer.generate(
            [caption],
            token_lens,
            timesteps=time_steps,
            cond_scale=cond_scale,
            temperature=1.0,
            topk_filter_thres=0.9,
            gsample=False,
        )
        mids = res_model.generate(
            mids, [caption], token_lens, temperature=1, cond_scale=5
        )
        pred_motions = vq_model.forward_decoder(mids).detach().cpu().numpy()
        data = pred_motions * std + mean
        feat = data[0][: max(m_length, 1)].astype(np.float32)
        joints = recover_from_ric(torch.from_numpy(feat).float(), 22).numpy().astype(
            np.float32
        )
        gen_s = time.perf_counter() - t0

    out_npy = Path(out_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_npy), joints)
    return out_npy, load_s, gen_s, int(joints.shape[0])
