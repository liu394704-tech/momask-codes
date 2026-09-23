#!/usr/bin/env python3
"""Lightweight motion quality metrics for generated (T, J, 3) joints."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def motion_quality_metrics(joints: Optional[np.ndarray]) -> Dict[str, float]:
    """joints: (T, J, 3). Used for the Motion Package quality block."""
    if joints is None or getattr(joints, "size", 0) == 0 or joints.shape[0] < 2:
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
    speed = np.linalg.norm(vel, axis=-1)
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


def quality_from_joints_path(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path or not str(path).endswith(".npy"):
        return None
    try:
        joints = np.load(path)
    except Exception:
        return None
    q = motion_quality_metrics(joints)
    q["joints_shape"] = list(joints.shape)
    return q
