#!/usr/bin/env python3
"""TonyPi pulse / millimetre coordinates for the current social actions."""
from __future__ import annotations

import math
import unittest

from pipeline.actions import ACTION_ALLOWLIST, BLOCKED_ACTIONS
from pipeline.tonypi_body import (
    BUS_COUNT,
    HEIGHT_MM,
    PULSE_CENTER,
    PULSE_MAX,
    PULSE_MIN,
    SIZE_MM,
    THICKNESS_MM,
    UP_HAND_LEFT,
    WIDTH_MM,
    apply_absolute,
    forward_kinematics,
    mirror_left_abs_to_right,
    mirror_pulses,
    stand_pulses,
)
from pipeline.tonypi_coords import (
    CLIP_COORDS,
    clip_coords,
    d6a_rows,
    export_tables,
    simulate_clip,
    simulate_clips,
)


class HardwareContractTests(unittest.TestCase):
    def test_official_envelope(self):
        self.assertEqual(SIZE_MM, (373.0, 186.0, 106.0))
        self.assertEqual((HEIGHT_MM, WIDTH_MM, THICKNESS_MM), SIZE_MM)

    def test_stand_is_all_500(self):
        self.assertEqual(stand_pulses(), [PULSE_CENTER] * BUS_COUNT)
        self.assertEqual(list(clip_coords("stand").frames[0].pulses), [500] * 16)
        self.assertEqual(list(clip_coords("stand_slow").frames[0].pulses), [500] * 16)

    def test_mirror_1000_minus(self):
        row = apply_absolute(absolute={1: 400, 6: 820, 8: 350})
        mirrored = mirror_pulses(row)
        self.assertEqual(mirrored[8], 1000 - 400)  # ID9
        self.assertEqual(mirrored[13], 1000 - 820)  # ID14
        self.assertEqual(mirrored[15], 1000 - 350)  # ID16

    def test_official_up_hand_mirror(self):
        right = mirror_left_abs_to_right(UP_HAND_LEFT)
        self.assertEqual(right, {6: 820, 7: 740, 8: 350})


class LibraryCoverageTests(unittest.TestCase):
    def test_every_allowlisted_clip_has_coords(self):
        self.assertEqual(set(CLIP_COORDS), set(ACTION_ALLOWLIST))
        for name in ACTION_ALLOWLIST:
            frames = clip_coords(name).frames
            self.assertGreaterEqual(len(frames), 1, name)
            for frame in frames:
                self.assertEqual(len(frame.pulses), 16, name)
                self.assertTrue(all(PULSE_MIN <= p <= PULSE_MAX for p in frame.pulses), name)
                self.assertGreaterEqual(frame.time_ms, 20, name)
                self.assertLessEqual(frame.time_ms, 9999, name)

    def test_blocked_actions_absent(self):
        for name in BLOCKED_ACTIONS:
            self.assertNotIn(name, CLIP_COORDS)

    def test_d6a_shape(self):
        rows = d6a_rows("wave")
        self.assertGreaterEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(len(row), 19)  # Time + 18 servos
            self.assertGreaterEqual(row[0], 20)


class ForwardKinematicsTests(unittest.TestCase):
    def test_stand_height_near_official(self):
        pose = forward_kinematics(stand_pulses())
        self.assertGreater(pose.points["head"][2], 330)
        self.assertLess(pose.points["head"][2], 400)
        self.assertAlmostEqual(pose.points["head"][2], HEIGHT_MM, delta=8)
        self.assertLess(abs(pose.points["left_foot"][2]), 8)
        self.assertLess(abs(pose.points["right_foot"][2]), 8)

    def test_stand_envelope_near_width_thickness(self):
        pose = forward_kinematics(stand_pulses())
        ys = [v[1] for v in pose.points.values()]
        xs = [v[0] for v in pose.points.values()]
        width = max(ys) - min(ys)
        thick = max(xs) - min(xs)
        self.assertGreater(width, 100)
        self.assertLess(width, 240)
        self.assertLess(thick, 160)

    def test_up_hand_raises_left(self):
        stand = forward_kinematics(stand_pulses())
        raised = forward_kinematics(apply_absolute(absolute=UP_HAND_LEFT))
        self.assertGreater(
            raised.points["left_hand"][2] - stand.points["left_hand"][2],
            40,
        )

    def test_mirrored_up_hand_raises_right(self):
        stand = forward_kinematics(stand_pulses())
        raised = forward_kinematics(apply_absolute(absolute=mirror_left_abs_to_right(UP_HAND_LEFT)))
        self.assertGreater(
            raised.points["right_hand"][2] - stand.points["right_hand"][2],
            40,
        )


class ActionSemanticsTests(unittest.TestCase):
    def _peak(self, name: str):
        poses = simulate_clip(name)
        return max(poses, key=lambda p: abs(p.points["head"][2] - HEIGHT_MM) + abs(p.points["com"][2]))

    def test_bow_lowers_head(self):
        stand_z = simulate_clip("stand")[0].points["head"][2]
        bow_min = min(p.points["head"][2] for p in simulate_clip("bow"))
        self.assertLess(bow_min, stand_z - 15)

    def test_jugong_at_least_as_deep_as_bow(self):
        bow_min = min(p.points["head"][2] for p in simulate_clip("bow"))
        jugong_min = min(p.points["head"][2] for p in simulate_clip("jugong"))
        self.assertLessEqual(jugong_min, bow_min + 5)

    def test_squat_lowers_com(self):
        stand_z = simulate_clip("stand")[0].points["com"][2]
        squat_min = min(p.points["com"][2] for p in simulate_clip("squat"))
        self.assertLess(squat_min, stand_z - 20)

    def test_wave_uses_right_hand(self):
        stand = simulate_clip("stand")[0]
        wave = max(simulate_clip("wave"), key=lambda p: p.points["right_hand"][2])
        self.assertGreater(wave.points["right_hand"][2], stand.points["right_hand"][2] + 30)
        self.assertGreater(wave.points["right_hand"][2], wave.points["left_hand"][2] - 5)

    def test_lift_left_hand_matches_official_pulses(self):
        peak = max(clip_coords("lift_left_hand").frames, key=lambda f: abs(f.pulses[13] - 500))
        self.assertEqual(peak.pulses[13], 180)
        self.assertEqual(peak.pulses[14], 260)
        self.assertEqual(peak.pulses[15], 650)

    def test_sidestep_distances(self):
        def last_y(name):
            return simulate_clip(name)[-1].root[1]

        self.assertAlmostEqual(last_y("left_move_10"), 10.0, delta=0.2)
        self.assertAlmostEqual(last_y("left_move_20"), 20.0, delta=0.2)
        self.assertAlmostEqual(last_y("right_move_10"), -10.0, delta=0.2)
        self.assertAlmostEqual(last_y("right_move_20"), -20.0, delta=0.2)
        self.assertGreater(last_y("left_move"), 10.0)
        self.assertLess(last_y("left_move"), 20.0)

    def test_forward_and_back(self):
        fwd = simulate_clip("go_forward_one_small_step")[-1].root[0]
        step = simulate_clip("go_forward_one_step")[-1].root[0]
        full = simulate_clip("go_forward")[-1].root[0]
        back = simulate_clip("back_one_step")[-1].root[0]
        self.assertAlmostEqual(fwd, 25.0, delta=0.2)
        self.assertAlmostEqual(step, 45.0, delta=0.2)
        self.assertAlmostEqual(full, 80.0, delta=0.2)
        self.assertAlmostEqual(back, -45.0, delta=0.2)
        self.assertLess(fwd, step)
        self.assertLess(step, full)

    def test_turns_yaw(self):
        left = simulate_clip("turn_left_small_step")[-1].yaw_rad
        right = simulate_clip("turn_right_small_step")[-1].yaw_rad
        self.assertGreater(left, 0)
        self.assertLess(right, 0)
        self.assertAlmostEqual(abs(math.degrees(left)), 15.0, delta=0.2)

    def test_phrase_concatenates_and_keeps_world(self):
        motion = simulate_clips(["left_move_10", "go_forward_one_small_step"], recovery="stand")
        self.assertGreaterEqual(len(motion.poses), 3)
        last = motion.poses[-1]
        self.assertAlmostEqual(last.root[1], 10.0, delta=0.2)
        self.assertAlmostEqual(last.root[0], 25.0, delta=0.2)
        self.assertEqual(motion.clips[-1], "stand")

    def test_export_tables_covers_allowlist(self):
        tables = export_tables()
        self.assertEqual(tables["robot"]["size_mm"]["height"], 373.0)
        self.assertEqual(set(tables["clips"]), set(ACTION_ALLOWLIST))
        wave = tables["clips"]["wave"]
        self.assertEqual(len(wave["d6a"]["columns"]), 19)
        self.assertTrue(wave["peak_xyz_mm"]["right_hand"])


if __name__ == "__main__":
    unittest.main()
