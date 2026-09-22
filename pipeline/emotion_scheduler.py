#!/usr/bin/python3
# coding=utf8
"""Pipeline-local copy of TonyPi EmotionActionScheduler (submodule may be empty).

This module owns the policy rather than the actual robot movement:
    stable emotion + intensity -> rotating action -> recovery stand -> cooldown -> idle

It deliberately has no OpenCV, MediaPipe, or hiwonder dependency, so desktop tools can
exercise exactly the same decision flow without controlling the robot.
"""
from __future__ import print_function

from collections import deque
import threading
import time


# ==================================================================================
#                        Shared action configuration
# ==================================================================================
# Intensity tiers (driven by classifier confidence):
#   mild   : softer / smaller responses
#   strong : clearer / larger responses
#
# Only non-aggressive ActionGroups from ActionGroupDict are used.
# Excluded on purpose: kick / uppercut / wing_chun / shot variants.
EMOTION_ACTIONS = {
    'neutral': {
        'mild': [],
        'strong': [],
    },
    'happy': {
        'mild': ['wave', 'stepping'],
        'strong': ['chest', 'wave', 'twist'],
    },
    'unhappy': {
        'mild': ['bow', 'jugong'],
        'strong': ['squat', 'bow', 'jugong'],
    },
    'surprised': {
        'mild': ['twist', 'stepping'],
        'strong': ['back_fast', 'twist', 'stand'],
    },
}

INTENSITY_STRONG_THRESHOLD = 0.55

RECOVERY_ACTION = 'stand'

ACTION_COOLDOWN = 8.0

SAME_EMOTION_INTERVAL = 12.0

VOTE_WINDOW = 8
VOTE_MIN = 6
VOTE_MIN_CONFIDENCE = 0.35

SAFE_ACTION_GROUPS = (
    'stand', 'stand_slow', 'wave', 'bow', 'jugong', 'squat', 'squat_down',
    'squat_up', 'chest', 'twist', 'stepping', 'back_fast', 'go_forward',
    'turn_left', 'turn_right', 'left_move_fast', 'right_move_fast',
    'left_hand', 'right_hand', 'lift_left_hand', 'go_hand_up', 'go_hand_up1',
    'back_one_step', 'go_forward_one_small_step', 'go_forward_one_step',
    'turn_left_small_step', 'turn_right_small_step', 'left_move_10',
    'right_move_10',
)

BLOCKED_ACTION_GROUPS = (
    'left_shot_fast', 'right_shot_fast', 'left_uppercut', 'right_uppercut',
    'left_kick', 'right_kick', 'wing_chun',
)


def intensity_from_confidence(confidence, strong_threshold=None):
    """Map classifier confidence to ``mild`` or ``strong``."""
    threshold = (
        INTENSITY_STRONG_THRESHOLD
        if strong_threshold is None else float(strong_threshold)
    )
    try:
        value = float(confidence or 0.0)
    except (TypeError, ValueError):
        return 'mild'
    return 'strong' if value >= threshold else 'mild'


def normalize_action_map(action_map):
    """Return a copy with list values and only known emotion keys."""
    out = {}
    source = action_map or {}
    for emotion in ('neutral', 'happy', 'unhappy', 'surprised'):
        pools = source.get(emotion) or {}
        out[emotion] = {
            'mild': list(pools.get('mild') or []),
            'strong': list(pools.get('strong') or []),
        }
    return out


def flatten_action_map(action_map):
    """Emotion -> unique action names across mild/strong."""
    flat = {}
    for emotion, pools in normalize_action_map(action_map).items():
        names = []
        for intensity in ('mild', 'strong'):
            for name in pools.get(intensity) or []:
                if name not in names:
                    names.append(name)
        flat[emotion] = names
    return flat


class EmotionVoter(object):
    def __init__(self, window_size=VOTE_WINDOW, min_votes=VOTE_MIN,
                 min_confidence=VOTE_MIN_CONFIDENCE, cooldown=SAME_EMOTION_INTERVAL):
        self.window_size = int(window_size)
        self.min_votes = int(min_votes)
        self.min_confidence = float(min_confidence)
        self.cooldown = float(cooldown)
        self._window = deque(maxlen=self.window_size)
        self._last_label = None
        self._last_time = 0.0

    def reset(self):
        self._window.clear()

    def push(self, emotion, confidence):
        label = (emotion or '').strip().lower() or 'neutral'
        try:
            conf = float(confidence or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        self._window.append((label, conf))

    def vote(self, now=None):
        if now is None:
            now = time.time()
        if self._last_label and (now - self._last_time) < self.cooldown:
            return None, 0.0
        counts = {}
        confs = {}
        for label, conf in self._window:
            if conf < self.min_confidence:
                continue
            counts[label] = counts.get(label, 0) + 1
            confs.setdefault(label, []).append(conf)
        if not counts:
            return None, 0.0
        label = max(
            counts,
            key=lambda key: (counts[key], sum(confs[key]) / float(len(confs[key]))),
        )
        if counts[label] < self.min_votes:
            return None, 0.0
        mean_conf = sum(confs[label]) / float(len(confs[label]))
        self._last_label = label
        self._last_time = now
        return label, mean_conf


class EmotionActionScheduler(object):
    """stable emotion + intensity -> rotating action -> recovery -> cooldown -> idle."""

    IDLE = 'IDLE'
    ACTION = 'ACTION'
    RECOVERY = 'RECOVERY'
    COOLDOWN = 'COOLDOWN'

    def __init__(self, action_map=None, recovery_action=RECOVERY_ACTION,
                 action_cooldown=ACTION_COOLDOWN,
                 same_emotion_interval=SAME_EMOTION_INTERVAL,
                 vote_window=VOTE_WINDOW, vote_min=VOTE_MIN,
                 vote_min_confidence=VOTE_MIN_CONFIDENCE,
                 strong_threshold=INTENSITY_STRONG_THRESHOLD):
        self.action_map = normalize_action_map(
            EMOTION_ACTIONS if action_map is None else action_map)
        self.recovery_action = str(recovery_action)
        self.action_cooldown = float(action_cooldown)
        self.same_emotion_interval = float(same_emotion_interval)
        self.strong_threshold = float(strong_threshold)
        self.voter = EmotionVoter(
            window_size=vote_window, min_votes=vote_min,
            min_confidence=vote_min_confidence,
            cooldown=self.same_emotion_interval,
        )
        self._lock = threading.RLock()
        self.reset()

    def reset(self):
        with self._lock:
            self.action_index = {}
            self.pending_action = None
            self.current_action = None
            self.current_emotion = None
            self.current_intensity = None
            self.current_confidence = 0.0
            self.state = self.IDLE
            self.cooldown_until = 0.0
            self.last_stable_emotion = None
            self.last_stable_intensity = None
            self.last_stable_confidence = 0.0
            self.last_stable_at = 0.0
            self.voter.reset()
            self.voter._last_label = None
            self.voter._last_time = 0.0

    def reset_votes(self):
        with self._lock:
            self.voter.reset()

    def observe(self, emotion, confidence):
        with self._lock:
            self.voter.push(emotion, confidence)

    def update(self, now=None):
        if now is None:
            now = time.time()
        with self._lock:
            if self.state == self.COOLDOWN and now >= self.cooldown_until:
                self.state = self.IDLE
                self.cooldown_until = 0.0
                self.current_action = None
                self.current_emotion = None
                self.current_intensity = None
                self.current_confidence = 0.0
            return self.state

    def cooldown_remaining(self, now=None):
        if now is None:
            now = time.time()
        with self._lock:
            self.update(now)
            return max(0.0, self.cooldown_until - now)

    def same_emotion_remaining(self, now=None):
        if now is None:
            now = time.time()
        with self._lock:
            if self.last_stable_emotion is None:
                return 0.0
            return max(0.0, self.same_emotion_interval - (now - self.last_stable_at))

    def can_schedule(self, now=None):
        if now is None:
            now = time.time()
        with self._lock:
            self.update(now)
            return self.state == self.IDLE

    def pick_action(self, emotion, intensity='mild'):
        with self._lock:
            pools = self.action_map.get(emotion) or {'mild': [], 'strong': []}
            intensity = 'strong' if intensity == 'strong' else 'mild'
            actions = list(pools.get(intensity) or [])
            if not actions and intensity == 'strong':
                actions = list(pools.get('mild') or [])
            if not actions:
                return None
            index_key = '%s:%s' % (emotion, intensity)
            index = self.action_index.get(index_key, 0)
            self.action_index[index_key] = (index + 1) % len(actions)
            return actions[index % len(actions)]

    def queue_action(self, action, emotion=None, intensity=None, confidence=None,
                     now=None):
        if now is None:
            now = time.time()
        if not action:
            return None
        with self._lock:
            if not self.can_schedule(now):
                return None
            self.pending_action = str(action)
            self.current_action = str(action)
            self.current_emotion = emotion
            self.current_intensity = intensity
            self.current_confidence = float(confidence or 0.0)
            self.state = self.ACTION
            return self.pending_action

    def plan(self, now=None):
        if now is None:
            now = time.time()
        with self._lock:
            if not self.can_schedule(now):
                return None, None

            stable, confidence = self.voter.vote(now)
            if stable is None:
                return None, None

            intensity = intensity_from_confidence(
                confidence, strong_threshold=self.strong_threshold)
            self.last_stable_emotion = stable
            self.last_stable_intensity = intensity
            self.last_stable_confidence = float(confidence or 0.0)
            self.last_stable_at = now

            if stable == 'neutral':
                return 'neutral', None

            action = self.pick_action(stable, intensity=intensity)
            if action:
                self.pending_action = str(action)
                self.current_action = str(action)
                self.current_emotion = stable
                self.current_intensity = intensity
                self.current_confidence = float(confidence or 0.0)
                self.state = self.ACTION
            return stable, action

    def mark_action_started(self):
        with self._lock:
            action = self.pending_action
            if action:
                self.state = self.ACTION
                self.current_action = action
            return action

    def mark_action_finished(self, now=None):
        if now is None:
            now = time.time()
        with self._lock:
            self.state = self.RECOVERY
            self.pending_action = None
            return self.recovery_action

    def mark_recovery_finished(self, now=None):
        if now is None:
            now = time.time()
        with self._lock:
            self.state = self.COOLDOWN
            self.cooldown_until = float(now) + float(self.action_cooldown)
            return self.cooldown_until

    def snapshot(self, now=None):
        if now is None:
            now = time.time()
        with self._lock:
            self.update(now)
            return {
                'state': self.state,
                'pending_action': self.pending_action,
                'current_action': self.current_action,
                'current_emotion': self.current_emotion,
                'current_intensity': self.current_intensity,
                'current_confidence': self.current_confidence,
                'last_stable_emotion': self.last_stable_emotion,
                'last_stable_intensity': self.last_stable_intensity,
                'last_stable_confidence': self.last_stable_confidence,
                'cooldown_remaining': max(0.0, self.cooldown_until - now),
            }
