#!/usr/bin/env python3
# coding=utf8
"""Offline tests for preset emotion -> ActionGroup mapping (no MoMask, no robot).

Usage:
    cd 源码
    source .venv-emotion-arm/bin/activate   # or any env with Python 3
    python preset_action_tests/test_preset_emotion_actions.py

    # print live checklist only
    python preset_action_tests/test_preset_emotion_actions.py --checklist

    # optional Mac policy monitor after offline pass
    python facedetect_mac_demo.py --duration 60
"""
from __future__ import print_function

import argparse
import json
import os
import sys
import time


ROOT = os.path.dirname(os.path.abspath(__file__))
FUNCTIONS_DIR = os.path.join(os.path.dirname(ROOT), 'TonyPi', 'Functions')
if FUNCTIONS_DIR not in sys.path:
    sys.path.insert(0, FUNCTIONS_DIR)

from EmotionActionScheduler import (  # noqa: E402
    BLOCKED_ACTION_GROUPS,
    EMOTION_ACTIONS,
    INTENSITY_STRONG_THRESHOLD,
    EmotionActionScheduler,
    flatten_action_map,
    intensity_from_confidence,
    normalize_action_map,
)


def load_scenarios():
    path = os.path.join(ROOT, 'scenarios.json')
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def feed_stable(scheduler, emotion, confidence, now, votes=None):
    votes = votes if votes is not None else scheduler.voter.min_votes
    for _ in range(votes):
        scheduler.observe(emotion, confidence)
    return scheduler.plan(now)


def advance_through_action(scheduler, now):
    """Simulate action + recovery so the next plan can run."""
    action = scheduler.mark_action_started()
    if action is None:
        return None
    scheduler.mark_action_finished(now + 0.1)
    scheduler.mark_recovery_finished(now + 0.2)
    # Jump past cooldown for synthetic batch tests.
    scheduler.cooldown_until = now
    scheduler.state = scheduler.IDLE
    return action


def test_mapping_table(scenarios, failures):
    expected = scenarios['mapping_expectations']
    actual = normalize_action_map(EMOTION_ACTIONS)
    print('\n[1] Mapping table')
    for emotion, pools in expected.items():
        for intensity in ('mild', 'strong'):
            want = list(pools.get(intensity) or [])
            got = list(actual.get(emotion, {}).get(intensity) or [])
            ok = want == got
            status = 'PASS' if ok else 'FAIL'
            print('  %s %s/%s: %s' % (status, emotion, intensity, got))
            if not ok:
                failures.append('%s/%s expected %s got %s' % (
                    emotion, intensity, want, got))


def test_no_blocked_actions(scenarios, failures):
    blocked = set(scenarios.get('blocked_actions') or BLOCKED_ACTION_GROUPS)
    flat = flatten_action_map(EMOTION_ACTIONS)
    print('\n[2] Safety: no aggressive ActionGroups')
    offenders = []
    for emotion, actions in flat.items():
        bad = [name for name in actions if name in blocked]
        if bad:
            offenders.append((emotion, bad))
    if offenders:
        for emotion, bad in offenders:
            print('  FAIL %s uses blocked %s' % (emotion, bad))
            failures.append('%s blocked actions %s' % (emotion, bad))
    else:
        print('  PASS all mapped actions are non-aggressive')


def test_synthetic_cases(scenarios, failures):
    print('\n[3] Synthetic stable-emotion planning')
    threshold = float(scenarios.get('strong_threshold', INTENSITY_STRONG_THRESHOLD))
    scheduler = EmotionActionScheduler(strong_threshold=threshold)
    now = time.time()

    for case in scenarios.get('synthetic_cases') or []:
        case_id = case['id']
        emotion = case['emotion']
        confidence = float(case['confidence'])
        expect_intensity = case['expect_intensity']
        expect_in = list(case.get('expect_action_in') or [])
        repeats = int(case.get('repeat') or 1)

        got_intensity = intensity_from_confidence(confidence, threshold)
        if got_intensity != expect_intensity:
            msg = '%s intensity %s != %s' % (case_id, got_intensity, expect_intensity)
            print('  FAIL %s' % msg)
            failures.append(msg)
            continue

        planned = []
        # Bypass same-emotion interval between repeats for rotation checks.
        scheduler.voter._last_label = None
        scheduler.voter._last_time = 0.0
        for index in range(repeats):
            tick = now + index * 100.0
            scheduler.voter.reset()
            scheduler.voter._last_label = None
            scheduler.voter._last_time = 0.0
            if scheduler.state != scheduler.IDLE:
                scheduler.state = scheduler.IDLE
                scheduler.cooldown_until = tick
            stable, action = feed_stable(scheduler, emotion, confidence, tick)
            snap = scheduler.snapshot(tick)
            planned.append((stable, action, snap.get('last_stable_intensity')))
            if action:
                advance_through_action(scheduler, tick)

        actions = [item[1] for item in planned]
        intensities = [item[2] for item in planned]
        if emotion == 'neutral':
            ok = all(action is None for action in actions)
        else:
            ok = all(action in expect_in for action in actions if action is not None)
            ok = ok and all(intensity == expect_intensity for intensity in intensities)
            if case.get('expect_rotation') and repeats > 1:
                ok = ok and len(set(actions)) >= min(2, len(expect_in))

        status = 'PASS' if ok else 'FAIL'
        print('  %s %s emotion=%s conf=%.2f intensity=%s actions=%s' % (
            status, case_id, emotion, confidence, expect_intensity, actions))
        if not ok:
            failures.append('%s planned %s' % (case_id, planned))


def test_same_emotion_hold(failures):
    print('\n[4] Same-emotion interval hold')
    scheduler = EmotionActionScheduler()
    now = time.time()
    stable1, action1 = feed_stable(scheduler, 'happy', 0.7, now)
    if action1:
        advance_through_action(scheduler, now)
        # Keep cooldown cleared but same-emotion interval active.
        scheduler.state = scheduler.IDLE
        scheduler.cooldown_until = now
    stable2, action2 = feed_stable(scheduler, 'happy', 0.7, now + 1.0)
    ok = stable1 == 'happy' and action1 is not None and stable2 is None and action2 is None
    print('  %s first=%s/%s second=%s/%s' % (
        'PASS' if ok else 'FAIL', stable1, action1, stable2, action2))
    if not ok:
        failures.append('same-emotion hold failed')


def print_checklist(scenarios):
    print('\n=== Live checklist (Mac / robot) ===')
    for item in scenarios.get('live_checklist') or []:
        print('[%s] emotion=%s intensity=%s' % (
            item['id'], item.get('emotion'), item.get('intensity', '-')))
        print('     how: %s' % item['how'])
        print('     expect: %s' % item['expect'])
    print('\nMac monitor command:')
    print('  python facedetect_mac_demo.py --duration 60')


def main():
    parser = argparse.ArgumentParser(description='Preset emotion-action offline tests')
    parser.add_argument('--checklist', action='store_true',
                        help='only print live manual checklist')
    args = parser.parse_args()
    scenarios = load_scenarios()

    if args.checklist:
        print_checklist(scenarios)
        return 0

    failures = []
    print('Preset emotion-action offline tests')
    print('strong_threshold=%.2f' % float(
        scenarios.get('strong_threshold', INTENSITY_STRONG_THRESHOLD)))
    print('flat map:', flatten_action_map(EMOTION_ACTIONS))

    test_mapping_table(scenarios, failures)
    test_no_blocked_actions(scenarios, failures)
    test_synthetic_cases(scenarios, failures)
    test_same_emotion_hold(failures)
    print_checklist(scenarios)

    print('\n=== Summary ===')
    if failures:
        print('FAILED (%d)' % len(failures))
        for item in failures:
            print(' - %s' % item)
        return 1
    print('ALL PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
