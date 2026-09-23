#!/usr/bin/env python3
"""TonyPi body, servo IDs, and a size-matched forward-kinematics stand-in.

Numbers come from this repo and the official Hiwonder TonyPi kit, not Mixamo:

* PC software Reset writes pulse 500 to bus IDs 1-16 (optional 17/18 hands).
* Sliders / .d6a rows are pulse 0-1000; frame time 20-9999 ms.
* Official envelope: 373 x 186 x 106 mm, about 1800 g, 16 body bus + 2 head PWM.
* Left/right mirror in the PC software is ``1000 - other`` for pairs 1↔9 … 8↔16.
* Vision-grip lesson: left raise ``14=180, 15=260, 16=650``; lower
  ``14=460, 15=200, 16=275``; hands 17/18 center 500, splay 760.
* PWM head (LAB_Tool / board demo): center 1500, legal 500-2500.

This module does **not** replace factory ``.d6a`` playback. It authors the
pulse/xyz layer that sits on top of the current ActionGroup names.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Official Hiwonder size / mass (standing envelope).
HEIGHT_MM = 373.0
WIDTH_MM = 186.0
THICKNESS_MM = 106.0
MASS_G = 1800.0
SIZE_MM = (HEIGHT_MM, WIDTH_MM, THICKNESS_MM)

BUS_COUNT = 16
HAND_COUNT = 2
ALL_BUS_COUNT = 18
PULSE_MIN = 0
PULSE_MAX = 1000
PULSE_CENTER = 500
# LX-824HV travel used by the PC sliders: 0-1000 ≈ ±120° about stand.
DEG_AT_LIMIT = 120.0
TIME_MIN_MS = 20
TIME_MAX_MS = 9999

PWM_CENTER = 1500
PWM_MIN = 500
PWM_MAX = 2500

# Segment lengths (mm) chosen so a zero-angle stand is the official 373 mm
# height and the hanging-arm envelope stays near 186 x 106.
FOOT_H = 28.0
SHANK = 72.0
THIGH = 78.0
PELVIS_H = 22.0
TORSO = 118.0
NECK = 25.0
HEAD = 30.0
# 28+72+78+22+118+25+30 = 373
HIP_HALF = 36.0
SHOULDER_HALF = 72.0
UPPER_ARM = 62.0
FOREARM = 58.0
FOOT_LEN = 42.0
FOOT_HALF = 18.0

# World: +X forward, +Y left, +Z up. Origin under the pelvis on the floor.
SERVO_NAMES: Dict[int, str] = {
    1: "right_foot_roll",
    2: "right_ankle_pitch",
    3: "right_knee",
    4: "right_hip_pitch",
    5: "right_hip_roll",
    6: "right_shoulder",
    7: "right_arm_roll",
    8: "right_elbow",
    9: "left_foot_roll",
    10: "left_ankle_pitch",
    11: "left_knee",
    12: "left_hip_pitch",
    13: "left_hip_roll",
    14: "left_shoulder",
    15: "left_arm_roll",
    16: "left_elbow",
    17: "right_hand",
    18: "left_hand",
}

# Pulse>500 sign that produces the +FK angle used below.
# Right = +1 on most hinges; left pitch/roll is opposite because the PC
# software mirrors with 1000-x. Elbows follow the official up_hand numbers
# (left 16=650 is flexion, so left elbow sign is +1).
SERVO_SIGN: Dict[int, int] = {
    1: +1, 2: +1, 3: +1, 4: +1, 5: +1,
    6: +1, 7: +1, 8: -1,
    9: -1, 10: -1, 11: -1, 12: -1, 13: -1,
    14: -1, 15: -1, 16: +1,
}

# Official grip-lesson left-arm poses (absolute pulses).
UP_HAND_LEFT = {14: 180, 15: 260, 16: 650}
DOWN_HAND_LEFT = {14: 460, 15: 200, 16: 275}
HAND_SPLAY = 760
HAND_GRASP = 500

Vec3 = Tuple[float, float, float]


def clamp_pulse(value: int) -> int:
    return max(PULSE_MIN, min(PULSE_MAX, int(round(value))))


def clamp_pwm(value: int) -> int:
    return max(PWM_MIN, min(PWM_MAX, int(round(value))))


def clamp_time_ms(value: int) -> int:
    return max(TIME_MIN_MS, min(TIME_MAX_MS, int(round(value))))


def stand_pulses(n: int = BUS_COUNT) -> List[int]:
    return [PULSE_CENTER] * n


def stand_hands() -> List[int]:
    return [PULSE_CENTER, PULSE_CENTER]


def stand_pwm() -> List[int]:
    return [PWM_CENTER, PWM_CENTER]


def pulse_to_rad(servo_id: int, pulse: int) -> float:
    delta = clamp_pulse(pulse) - PULSE_CENTER
    deg = SERVO_SIGN.get(servo_id, 1) * delta * (DEG_AT_LIMIT / float(PULSE_CENTER))
    return math.radians(deg)


def mirror_pulses(pulses: Sequence[int]) -> List[int]:
    """Official PC-software mirror: pair 1↔9 … 8↔16 via 1000-x; 17↔18 the same."""
    out = [clamp_pulse(p) for p in pulses]
    while len(out) < ALL_BUS_COUNT:
        out.append(PULSE_CENTER)
    for i in range(8):
        right, left = out[i], out[i + 8]
        out[i] = clamp_pulse(PULSE_MAX - left)
        out[i + 8] = clamp_pulse(PULSE_MAX - right)
    if len(out) >= 18:
        out[16], out[17] = clamp_pulse(PULSE_MAX - out[17]), clamp_pulse(PULSE_MAX - out[16])
    return out[: len(pulses)]


def apply_absolute(
    base: Optional[Sequence[int]] = None,
    absolute: Optional[Mapping[int, int]] = None,
    delta: Optional[Mapping[int, int]] = None,
) -> List[int]:
    out = list(base) if base is not None else stand_pulses()
    if len(out) < BUS_COUNT:
        out.extend([PULSE_CENTER] * (BUS_COUNT - len(out)))
    if absolute:
        for sid, pulse in absolute.items():
            if 1 <= sid <= len(out):
                out[sid - 1] = clamp_pulse(pulse)
    if delta:
        for sid, dlt in delta.items():
            if 1 <= sid <= len(out):
                out[sid - 1] = clamp_pulse(out[sid - 1] + int(dlt))
    return out[:BUS_COUNT]


def mirror_left_abs_to_right(left_abs: Mapping[int, int]) -> Dict[int, int]:
    """Map left IDs 9-16 (and 14-16 arms) onto right 1-8 via 1000-x."""
    out: Dict[int, int] = {}
    for sid, pulse in left_abs.items():
        if 9 <= sid <= 16:
            out[sid - 8] = clamp_pulse(PULSE_MAX - pulse)
        elif 17 <= sid <= 18:
            out[35 - sid] = clamp_pulse(PULSE_MAX - pulse)
    return out


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _rot_x(p: Vec3, ang: float) -> Vec3:
    c, s = math.cos(ang), math.sin(ang)
    return (p[0], c * p[1] - s * p[2], s * p[1] + c * p[2])


def _rot_y(p: Vec3, ang: float) -> Vec3:
    c, s = math.cos(ang), math.sin(ang)
    return (c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2])


def _rot_z(p: Vec3, ang: float) -> Vec3:
    c, s = math.cos(ang), math.sin(ang)
    return (c * p[0] - s * p[1], s * p[0] + c * p[1], p[2])


def _yaw_offset(p: Vec3, yaw: float, origin: Vec3) -> Vec3:
    local = (p[0] - origin[0], p[1] - origin[1], p[2] - origin[2])
    spun = _rot_z(local, yaw)
    return _add(origin, spun)


class Pose:
    """Named 3D points in millimetres plus the 16-bus pulse row that made them."""

    def __init__(
        self,
        pulses: Sequence[int],
        points: Dict[str, Vec3],
        root: Vec3,
        yaw_rad: float,
        hands: Optional[Sequence[int]] = None,
        pwm: Optional[Sequence[int]] = None,
        time_ms: int = 0,
    ):
        self.pulses = [clamp_pulse(p) for p in pulses[:BUS_COUNT]]
        self.points = dict(points)
        self.root = root
        self.yaw_rad = float(yaw_rad)
        self.hands = [clamp_pulse(p) for p in (hands or stand_hands())][:HAND_COUNT]
        self.pwm = [clamp_pwm(p) for p in (pwm or stand_pwm())][:2]
        self.time_ms = clamp_time_ms(time_ms) if time_ms else 0

    @property
    def com(self) -> Vec3:
        return self.points["com"]

    def compact(self) -> Dict[str, object]:
        keys = (
            "head", "neck", "pelvis", "com",
            "left_hand", "right_hand",
            "left_foot", "right_foot",
            "left_shoulder", "right_shoulder",
        )
        return {
            "t_ms": self.time_ms,
            "servos": list(self.pulses),
            "hands": list(self.hands),
            "pwm": list(self.pwm),
            "root_mm": [round(v, 2) for v in self.root],
            "yaw_deg": round(math.degrees(self.yaw_rad), 2),
            "xyz_mm": {k: [round(c, 2) for c in self.points[k]] for k in keys if k in self.points},
        }


def forward_kinematics(
    pulses: Sequence[int],
    root: Vec3 = (0.0, 0.0, 0.0),
    yaw_rad: float = 0.0,
    hands: Optional[Sequence[int]] = None,
    pwm: Optional[Sequence[int]] = None,
    time_ms: int = 0,
    plant_feet: bool = True,
) -> Pose:
    """Simple 16-DOF FK sized to 373 x 186 x 106 mm.

    Joint zeros are the official Reset/stand pose (all pulses 500). Arms hang
    along -Z; positive shoulder swing raises the hand forward/up.
    """
    p = apply_absolute(pulses)
    ang = {sid: pulse_to_rad(sid, p[sid - 1]) for sid in range(1, BUS_COUNT + 1)}
    pwm = list(pwm or stand_pwm())
    head_pan = math.radians((pwm[1] - PWM_CENTER) * (60.0 / 400.0))
    head_tilt = math.radians((pwm[0] - PWM_CENTER) * (40.0 / 400.0))

    hip_c = (0.0, 0.0, FOOT_H + SHANK + THIGH + PELVIS_H)

    def _leg(side: str) -> Dict[str, Vec3]:
        sign = 1.0 if side == "left" else -1.0
        if side == "left":
            roll_f, pitch_a, knee, hip_p, hip_r = (ang[9], ang[10], ang[11], ang[12], ang[13])
        else:
            roll_f, pitch_a, knee, hip_p, hip_r = (ang[1], ang[2], ang[3], ang[4], ang[5])
        hip = (0.0, sign * HIP_HALF, hip_c[2])
        # thigh along -Z, flex hip (pitch) toward +X, roll about X.
        thigh = _rot_x(_rot_y((0.0, 0.0, -THIGH), hip_p), hip_r)
        knee_p = _add(hip, thigh)
        shank = _rot_x(_rot_y((0.0, 0.0, -SHANK), hip_p + knee), hip_r)
        ankle = _add(knee_p, shank)
        # Stand (all zeros): ankle sits PELVIS_H+FOOT_H above the floor, so the
        # foot offset must drop that far or official height 373 mm is short.
        foot_drop = FOOT_H + PELVIS_H
        foot = _rot_x(_rot_y(_rot_x((FOOT_LEN * 0.35, 0.0, -foot_drop), roll_f), hip_p + knee + pitch_a), hip_r)
        toe = _add(ankle, foot)
        heel = _add(ankle, _rot_x(_rot_y(_rot_x((-FOOT_LEN * 0.25, 0.0, -foot_drop), roll_f), hip_p + knee + pitch_a), hip_r))
        return {"hip": hip, "knee": knee_p, "ankle": ankle, "foot": toe, "heel": heel}

    left_leg = _leg("left")
    right_leg = _leg("right")

    def _arm(side: str) -> Dict[str, Vec3]:
        sign = 1.0 if side == "left" else -1.0
        if side == "left":
            sh, roll, elb = ang[14], ang[15], ang[16]
        else:
            sh, roll, elb = ang[6], ang[7], ang[8]
        shoulder = (8.0, sign * SHOULDER_HALF, hip_c[2] + TORSO)
        # Hang along -Z; +shoulder swings the arm forward (toward +X) and up.
        upper = _rot_y(_rot_x((0.0, 0.0, -UPPER_ARM), roll), -sh)
        elbow = _add(shoulder, upper)
        fore = _rot_y(_rot_x((0.0, 0.0, -FOREARM), roll), -(sh + elb * 0.85))
        hand = _add(elbow, fore)
        return {"shoulder": shoulder, "elbow": elbow, "hand": hand}

    left_arm = _arm("left")
    right_arm = _arm("right")
    neck = (6.0, 0.0, hip_c[2] + TORSO + NECK)
    head = _add(neck, _rot_y(_rot_z((0.0, 0.0, HEAD), head_pan), -head_tilt))

    pts: Dict[str, Vec3] = {
        "pelvis": hip_c,
        "neck": neck,
        "head": head,
        "left_hip": left_leg["hip"],
        "right_hip": right_leg["hip"],
        "left_knee": left_leg["knee"],
        "right_knee": right_leg["knee"],
        "left_ankle": left_leg["ankle"],
        "right_ankle": right_leg["ankle"],
        "left_foot": left_leg["foot"],
        "right_foot": right_leg["foot"],
        "left_heel": left_leg["heel"],
        "right_heel": right_leg["heel"],
        "left_shoulder": left_arm["shoulder"],
        "right_shoulder": right_arm["shoulder"],
        "left_elbow": left_arm["elbow"],
        "right_elbow": right_arm["elbow"],
        "left_hand": left_arm["hand"],
        "right_hand": right_arm["hand"],
    }

    if plant_feet:
        ground = min(pts["left_foot"][2], pts["right_foot"][2], pts["left_heel"][2], pts["right_heel"][2])
        if abs(ground) > 1e-6:
            shift = (0.0, 0.0, -ground)
            pts = {k: _add(v, shift) for k, v in pts.items()}

    # Pelvis / torso / thigh dominate the 1800 g mass for a coarse CoM.
    weights = {
        "head": 0.10,
        "pelvis": 0.22,
        "left_shoulder": 0.06,
        "right_shoulder": 0.06,
        "left_hand": 0.03,
        "right_hand": 0.03,
        "left_knee": 0.12,
        "right_knee": 0.12,
        "left_foot": 0.13,
        "right_foot": 0.13,
    }
    com = (0.0, 0.0, 0.0)
    wsum = 0.0
    for name, w in weights.items():
        com = _add(com, _scale(pts[name], w))
        wsum += w
    pts["com"] = _scale(com, 1.0 / wsum)

    origin = (0.0, 0.0, 0.0)
    if abs(yaw_rad) > 1e-9 or root != (0.0, 0.0, 0.0):
        moved: Dict[str, Vec3] = {}
        for key, val in pts.items():
            spun = _yaw_offset(val, yaw_rad, origin)
            moved[key] = _add(spun, root)
        pts = moved

    return Pose(
        pulses=p,
        points=pts,
        root=root,
        yaw_rad=yaw_rad,
        hands=hands,
        pwm=pwm,
        time_ms=time_ms,
    )


def pose_extent_mm(pose: Pose) -> Tuple[float, float, float]:
    xs = [v[0] for v in pose.points.values()]
    ys = [v[1] for v in pose.points.values()]
    zs = [v[2] for v in pose.points.values()]
    return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))


def validate_pulses(pulses: Iterable[int]) -> List[int]:
    out = [clamp_pulse(p) for p in pulses]
    if len(out) != BUS_COUNT:
        raise ValueError("expected %d bus pulses, got %d" % (BUS_COUNT, len(out)))
    return out
