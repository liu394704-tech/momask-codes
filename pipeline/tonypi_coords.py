#!/usr/bin/env python3
"""Pulse + millimetre coordinates for every current social ActionGroup.

Each allow-listed clip is a short keyframe table (PC-software / .d6a shape:
Time + Servo1-18). Pulses start from official stand=500 and only move the
joints that the clip name implies. Sidestep distances come from the clip
suffixes ``_10`` / ``_20`` (mm). Arm raises reuse the grip-lesson numbers
already documented for this robot (left 14/15/16 = 180/260/650).

Track A still plays factory ``.d6a`` on the robot. These rows are the
authored coordinates that sit on top of the current action settings so a
phrase can be inspected, logged, or later imported into the PC software.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .actions import ACTION_ALLOWLIST, BLOCKED_ACTIONS
from .preset_catalog import CATALOG
from .tonypi_body import (
    ALL_BUS_COUNT,
    BUS_COUNT,
    DOWN_HAND_LEFT,
    HAND_GRASP,
    HEIGHT_MM,
    PULSE_CENTER,
    PULSE_MAX,
    SIZE_MM,
    THICKNESS_MM,
    TIME_MIN_MS,
    UP_HAND_LEFT,
    WIDTH_MM,
    Pose,
    apply_absolute,
    clamp_pulse,
    clamp_time_ms,
    forward_kinematics,
    mirror_left_abs_to_right,
    stand_hands,
    stand_pulses,
    stand_pwm,
)

World = Tuple[float, float, float, float]  # x_mm, y_mm, z_mm, yaw_rad


@dataclass(frozen=True)
class Keyframe:
    time_ms: int
    pulses: Tuple[int, ...]
    hands: Tuple[int, ...] = (PULSE_CENTER, PULSE_CENTER)
    pwm: Tuple[int, ...] = (1500, 1500)
    world: World = (0.0, 0.0, 0.0, 0.0)

    def as_d6a_row(self) -> List[int]:
        hands = list(self.hands) + [PULSE_CENTER] * 2
        return [clamp_time_ms(self.time_ms), *list(self.pulses)[:BUS_COUNT], hands[0], hands[1]]


@dataclass
class ClipCoords:
    name: str
    frames: List[Keyframe]
    note: str = ""
    source: str = "repo_sized_fk"

    def peak(self) -> Keyframe:
        if len(self.frames) == 1:
            return self.frames[0]
        return self.frames[min(len(self.frames) - 1, max(1, len(self.frames) // 2))]


@dataclass
class SimulatedMotion:
    clips: List[str]
    poses: List[Pose] = field(default_factory=list)
    frames: List[dict] = field(default_factory=list)
    size_mm: Tuple[float, float, float] = SIZE_MM
    source: str = "repo_sized_fk"

    def compact(self) -> Dict[str, object]:
        peak = self.poses[len(self.poses) // 2] if self.poses else None
        return {
            "clips": list(self.clips),
            "size_mm": list(self.size_mm),
            "source": self.source,
            "n_frames": len(self.poses),
            "total_ms": sum(p.time_ms for p in self.poses),
            "frames": [p.compact() for p in self.poses],
            "peak_xyz_mm": None if peak is None else peak.compact()["xyz_mm"],
        }


def _kf(
    time_ms: int,
    absolute: Optional[Mapping[int, int]] = None,
    delta: Optional[Mapping[int, int]] = None,
    world: World = (0.0, 0.0, 0.0, 0.0),
    hands: Optional[Sequence[int]] = None,
    pwm: Optional[Sequence[int]] = None,
    base: Optional[Sequence[int]] = None,
) -> Keyframe:
    pulses = apply_absolute(base, absolute=absolute, delta=delta)
    hh = tuple(hands or stand_hands())
    pp = tuple(pwm or stand_pwm())
    return Keyframe(
        time_ms=clamp_time_ms(time_ms),
        pulses=tuple(pulses),
        hands=(clamp_pulse(hh[0]), clamp_pulse(hh[1])),
        pwm=(int(pp[0]), int(pp[1])),
        world=world,
    )


def _stand_frame(time_ms: int = 400, world: World = (0.0, 0.0, 0.0, 0.0)) -> Keyframe:
    return _kf(time_ms, world=world)


def _hold(frame: Keyframe, time_ms: int) -> Keyframe:
    return Keyframe(
        time_ms=clamp_time_ms(time_ms),
        pulses=frame.pulses,
        hands=frame.hands,
        pwm=frame.pwm,
        world=frame.world,
    )


def _left_arm(abs_left: Mapping[int, int], also_right: bool = False) -> Dict[int, int]:
    out = dict(abs_left)
    if also_right:
        out.update(mirror_left_abs_to_right(abs_left))
    return out


def _right_arm_from_left(abs_left: Mapping[int, int]) -> Dict[int, int]:
    return mirror_left_abs_to_right(abs_left)


def _gait_legs(phase: str, amount: int = 70) -> Dict[int, int]:
    """Small planted gait. phase: L_up, R_up, L_push, R_push, center."""
    a = int(amount)
    table = {
        "center": {},
        "L_up": {11: a, 12: -a // 2, 10: a // 3, 3: a // 4, 4: a // 5},
        "R_up": {3: a, 4: -a // 2, 2: a // 3, 11: a // 4, 12: a // 5},
        "L_push": {12: a // 2, 4: -a // 3, 11: a // 3, 2: -a // 4},
        "R_push": {4: a // 2, 12: -a // 3, 3: a // 3, 10: -a // 4},
        "L_side": {5: -a // 2, 13: a // 2, 1: a // 3, 9: -a // 3},
        "R_side": {5: a // 2, 13: -a // 2, 1: -a // 3, 9: a // 3},
    }
    raw = table.get(phase, {})
    # Convert signed "flexion amount" into left/right absolute-friendly deltas
    # that respect 1000-x: positive a on a left pitch id means pulse 500+a.
    return {sid: int(val) for sid, val in raw.items()}


def _world(x: float = 0.0, y: float = 0.0, yaw_deg: float = 0.0) -> World:
    import math
    return (float(x), float(y), 0.0, math.radians(float(yaw_deg)))


def _walk(distance_mm: float, n_steps: int, side: str, time_ms: int) -> List[Keyframe]:
    """Translate the root while cycling a tiny gait. +X forward, +Y left."""
    frames = [_stand_frame(220)]
    sign_x = 0.0
    sign_y = 0.0
    yaw = 0.0
    if side == "fwd":
        sign_x = 1.0
    elif side == "back":
        sign_x = -1.0
    elif side == "left":
        sign_y = 1.0
    elif side == "right":
        sign_y = -1.0
    elif side == "turn_left":
        yaw = 1.0
    elif side == "turn_right":
        yaw = -1.0
    for i in range(1, n_steps + 1):
        frac = i / float(n_steps)
        phase = "L_up" if i % 2 else "R_up"
        if side in ("left", "right"):
            phase = "L_side" if side == "left" else "R_side"
        world = _world(sign_x * distance_mm * frac, sign_y * distance_mm * frac, yaw * distance_mm * frac)
        # For turns, distance_mm is degrees.
        if side.startswith("turn"):
            world = _world(0.0, 0.0, yaw * distance_mm * frac)
        frames.append(_kf(time_ms, delta=_gait_legs(phase, 55 if abs(distance_mm) < 25 else 80), world=world))
    frames.append(_stand_frame(280, world=frames[-1].world))
    return frames


def _build_library() -> Dict[str, ClipCoords]:
    lib: Dict[str, ClipCoords] = {}

    def add(name: str, frames: List[Keyframe], note: str) -> None:
        lib[name] = ClipCoords(name=name, frames=frames, note=note)

    add("stand", [_stand_frame(800)], "official Reset: all 16 bus servos at 500")
    add("stand_slow", [_stand_frame(1400)], "same 500 stand, longer hold")

    # Official 8-frame bow course: stand → straighten → raise → prep → bow →
    # hold → unbow keep arms → stand. Pulses reconstructed from stand=500,
    # 1000-x mirror, and the grip-lesson arm numbers (figures are not tables).
    bow_raise = _left_arm({14: 400, 15: 320, 16: 420}, also_right=True)
    bow_prep = dict(bow_raise)
    bow_prep.update({4: 430, 12: 570, 2: 530, 10: 470, 3: 470, 11: 530})
    bow_down = dict(bow_raise)
    bow_down.update({4: 360, 12: 640, 2: 560, 10: 440, 3: 430, 11: 570, 6: 620, 14: 380})
    add(
        "bow",
        [
            _stand_frame(400),
            _kf(350, delta={3: -20, 11: 20, 8: -15, 16: 15}),
            _kf(400, absolute=bow_raise),
            _kf(400, absolute=bow_prep),
            _kf(500, absolute=bow_down),
            _kf(500, absolute=bow_down),
            _kf(450, absolute=bow_prep),
            _stand_frame(450),
        ],
        "official 8-frame bow lesson, pulses simulated from stand + hip lean",
    )
    jugong_down = dict(bow_down)
    jugong_down.update({4: 330, 12: 670, 2: 580, 10: 420})
    add(
        "jugong",
        [
            _stand_frame(450),
            _kf(400, absolute=bow_raise),
            _kf(500, absolute=bow_prep),
            _kf(600, absolute=jugong_down),
            _kf(600, absolute=jugong_down),
            _kf(500, absolute=bow_prep),
            _stand_frame(500),
        ],
        "deeper bow / 鞠躬 from the same hip-pitch family",
    )

    squat_down = {3: 360, 11: 640, 4: 380, 12: 620, 2: 560, 10: 440, 6: 540, 14: 460}
    add(
        "squat",
        [_stand_frame(300), _kf(500, absolute=squat_down), _hold(_kf(400, absolute=squat_down), 400), _stand_frame(500)],
        "both knees/hips flex, ankles compensate so feet stay planted",
    )
    add("squat_down", [_stand_frame(250), _kf(700, absolute=squat_down)], "down half of squat")
    add("squat_up", [_kf(250, absolute=squat_down), _stand_frame(700)], "up half of squat")

    # Hands: official up_hand / down_hand on the left, 1000-x on the right.
    up_l = _left_arm(UP_HAND_LEFT)
    up_r = _right_arm_from_left(UP_HAND_LEFT)
    up_both = _left_arm(UP_HAND_LEFT, also_right=True)
    mid_l = _left_arm({14: 340, 15: 380, 16: 580})
    mid_r = _right_arm_from_left({14: 340, 15: 380, 16: 580})
    down_l = _left_arm(DOWN_HAND_LEFT)
    add(
        "lift_left_hand",
        [_stand_frame(250), _kf(700, absolute=up_l), _hold(_kf(400, absolute=up_l), 350), _stand_frame(400)],
        "official up_hand: 14=180 15=260 16=650",
    )
    add(
        "left_hand",
        [_stand_frame(250), _kf(600, absolute=mid_l), _stand_frame(400)],
        "half-raise of the official left-arm pose",
    )
    add(
        "right_hand",
        [_stand_frame(250), _kf(600, absolute=mid_r), _stand_frame(400)],
        "mirror of left_hand via 1000-x",
    )
    add(
        "go_hand_up",
        [_stand_frame(250), _kf(700, absolute=up_both), _hold(_kf(400, absolute=up_both), 350), _stand_frame(400)],
        "both arms at official up_hand / mirrored up_hand",
    )
    add(
        "go_hand_up1",
        [_stand_frame(250), _kf(500, absolute=up_l), _kf(500, absolute=up_both), _stand_frame(400)],
        "left official raise, then both",
    )
    wave_pose = dict(up_r)
    wave_pose.update({8: 320})
    wave_hi = dict(wave_pose)
    wave_hi[8] = 220
    wave_lo = dict(wave_pose)
    wave_lo[8] = 420
    add(
        "wave",
        [
            _stand_frame(250),
            _kf(450, absolute=wave_pose),
            _kf(280, absolute=wave_hi),
            _kf(280, absolute=wave_lo),
            _kf(280, absolute=wave_hi),
            _stand_frame(400),
        ],
        "right arm mirrored from up_hand, elbow oscillates",
    )

    chest = {6: 680, 7: 620, 8: 280, 14: 320, 15: 380, 16: 720}
    add(
        "chest",
        [_stand_frame(250), _kf(500, absolute=chest), _hold(_kf(400, absolute=chest), 400), _stand_frame(450)],
        "both elbows in, celebrate / 击掌胸前",
    )
    twist = {5: 420, 13: 580, 6: 560, 14: 440, 1: 530, 9: 470}
    twist_other = {5: 580, 13: 420, 6: 440, 14: 560, 1: 470, 9: 530}
    add(
        "twist",
        [_stand_frame(250), _kf(400, absolute=twist), _kf(400, absolute=twist_other), _kf(350, absolute=twist), _stand_frame(400)],
        "hip-roll + opposite shoulders, startle / idle",
    )
    add(
        "stepping",
        [
            _stand_frame(220),
            _kf(320, delta=_gait_legs("L_up", 90), world=_world(8, 0)),
            _kf(320, delta=_gait_legs("R_up", 90), world=_world(4, 0)),
            _kf(320, delta=_gait_legs("L_up", 70), world=_world(10, 0)),
            _stand_frame(350, world=_world(0, 0)),
        ],
        "in-place march, root stays inside a 10 mm box",
    )

    # Locomotion: names already encode 10 mm / 20 mm. Small / full steps are
    # sized against the 373 mm height (≈1/15, 1/8, 1/5 of stature).
    add("left_move_10", _walk(10, 2, "left", 320), "named 10 mm left sidestep")
    add("right_move_10", _walk(10, 2, "right", 320), "named 10 mm right sidestep")
    add("left_move_20", _walk(20, 2, "left", 380), "named 20 mm left sidestep")
    add("right_move_20", _walk(20, 2, "right", 380), "named 20 mm right sidestep")
    add("left_move", _walk(15, 2, "left", 360), "generic left sidestep ≈15 mm")
    add("right_move", _walk(15, 2, "right", 360), "generic right sidestep ≈15 mm")
    add("go_forward_one_small_step", _walk(25, 2, "fwd", 360), "≈25 mm, ~1/15 of 373 mm height")
    add("go_forward_one_step", _walk(45, 2, "fwd", 400), "≈45 mm one step")
    add("go_forward", _walk(80, 3, "fwd", 380), "full-cycle forward ≈80 mm")
    add("back_one_step", _walk(45, 2, "back", 360), "≈45 mm back")
    add("back_fast", _walk(70, 3, "back", 300), "faster / longer retreat")
    add("turn_left_small_step", _walk(15, 2, "turn_left", 360), "≈15° left")
    add("turn_right_small_step", _walk(15, 2, "turn_right", 360), "≈15° right")
    add("turn_left_small_step_a", _walk(10, 2, "turn_left", 320), "≈10° left")
    add("turn_right_small_step_a", _walk(10, 2, "turn_right", 320), "≈10° right")
    add("turn_left", _walk(30, 3, "turn_left", 380), "≈30° left")
    add("turn_right", _walk(30, 3, "turn_right", 380), "≈30° right")

    missing = [n for n in ACTION_ALLOWLIST if n not in lib]
    extra = [n for n in lib if n not in ACTION_ALLOWLIST]
    if missing or extra:
        raise RuntimeError("coord library mismatch missing=%s extra=%s" % (missing, extra))
    for name in lib:
        if name in BLOCKED_ACTIONS:
            raise RuntimeError("blocked action has coordinates: %s" % name)
    return lib


CLIP_COORDS: Dict[str, ClipCoords] = _build_library()


def clip_coords(name: str) -> ClipCoords:
    if name not in CLIP_COORDS:
        raise KeyError("no coordinates for action %r" % name)
    return CLIP_COORDS[name]


def simulate_clip(name: str, world_offset: World = (0.0, 0.0, 0.0, 0.0)) -> List[Pose]:
    clip = clip_coords(name)
    poses: List[Pose] = []
    ox, oy, oz, oyaw = world_offset
    for frame in clip.frames:
        wx, wy, wz, wyaw = frame.world
        poses.append(
            forward_kinematics(
                frame.pulses,
                root=(ox + wx, oy + wy, oz + wz),
                yaw_rad=oyaw + wyaw,
                hands=frame.hands,
                pwm=frame.pwm,
                time_ms=frame.time_ms,
            )
        )
    return poses


def simulate_clips(
    names: Sequence[str],
    recovery: Optional[str] = None,
    pause_ms: int = 0,
) -> SimulatedMotion:
    clips = [n for n in names if n]
    if recovery and recovery not in clips[-1:]:
        clips = list(clips) + [recovery]
    poses: List[Pose] = []
    ox = oy = oyaw = 0.0
    for index, name in enumerate(clips):
        part = simulate_clip(name, world_offset=(ox, oy, 0.0, oyaw))
        if part:
            last = part[-1]
            ox, oy, oyaw = last.root[0], last.root[1], last.yaw_rad
            poses.extend(part)
        if pause_ms and index + 1 < len(clips):
            poses.append(
                forward_kinematics(
                    stand_pulses(),
                    root=(ox, oy, 0.0),
                    yaw_rad=oyaw,
                    time_ms=max(TIME_MIN_MS, pause_ms),
                )
            )
    return SimulatedMotion(clips=list(clips), poses=poses, frames=[p.compact() for p in poses])


def d6a_rows(name: str) -> List[List[int]]:
    return [frame.as_d6a_row() for frame in clip_coords(name).frames]


def export_tables() -> Dict[str, object]:
    clips = {}
    for name in ACTION_ALLOWLIST:
        item = clip_coords(name)
        peak = simulate_clip(name)
        mid = peak[len(peak) // 2]
        clips[name] = {
            "note": item.note,
            "duration_hint_s": getattr(CATALOG.get(name), "duration_hint_s", None),
            "d6a": {
                "columns": ["Time"] + ["Servo%d" % i for i in range(1, ALL_BUS_COUNT + 1)],
                "rows": d6a_rows(name),
            },
            "peak_xyz_mm": mid.compact()["xyz_mm"],
            "peak_servos": list(mid.pulses),
        }
    return {
        "robot": {
            "size_mm": {"height": HEIGHT_MM, "width": WIDTH_MM, "thickness": THICKNESS_MM},
            "pulse_center": PULSE_CENTER,
            "pulse_range": [0, PULSE_MAX],
            "bus_servos": BUS_COUNT,
            "hands_optional": 2,
            "source": "Hiwonder TonyPi + repo PC software / grip lesson",
        },
        "clips": clips,
    }


def summarize_motion(motion: SimulatedMotion) -> str:
    if not motion.poses:
        return "coords: empty"
    peak = motion.poses[len(motion.poses) // 2]
    xyz = peak.points
    return (
        "coords: size=%.0fx%.0fx%.0fmm frames=%d total=%dms "
        "head_z=%.0f lh_z=%.0f rh_z=%.0f com=(%.0f,%.0f,%.0f)"
        % (
            SIZE_MM[0], SIZE_MM[1], SIZE_MM[2],
            len(motion.poses),
            sum(p.time_ms for p in motion.poses),
            xyz["head"][2],
            xyz["left_hand"][2],
            xyz["right_hand"][2],
            xyz["com"][0], xyz["com"][1], xyz["com"][2],
        )
    )


if __name__ == "__main__":
    import json
    import sys

    names = [a for a in sys.argv[1:] if not a.startswith("-")]
    if names:
        payload = {name: export_tables()["clips"][name] for name in names}
    else:
        payload = export_tables()
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
