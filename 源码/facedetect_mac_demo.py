#!/usr/bin/env python3
# coding=utf8
"""Mac monitor for the FaceDetect emotion-feedback policy.

This program uses the same geometry emotion model and EmotionActionScheduler as
``TonyPi/Functions/FaceDetect.py``.  It NEVER imports robot SDK modules and NEVER
calls ActionGroupControl: action and recovery durations are simulated solely to
make the normal action -> stand -> cooldown state flow observable.
"""
from __future__ import print_function

import argparse
import os
import sys
import time

FUNCTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'TonyPi', 'Functions')
if FUNCTIONS_DIR not in sys.path:
    sys.path.insert(0, FUNCTIONS_DIR)

import cv2
import numpy as np

import FaceExpression as FX
from EmotionActionScheduler import EmotionActionScheduler, flatten_action_map


WINDOW = 'FaceDetect Mac Monitor'
POINT_COLORS = [(0, 0, 255), (0, 128, 255), (0, 255, 255), (255, 0, 255)]


def build_parser():
    parser = argparse.ArgumentParser(
        description='Mac monitor for the FaceDetect emotion action policy (no robot motion).')
    parser.add_argument('--camera-index', type=int, default=0,
                        help='preferred local camera index (default: 0)')
    parser.add_argument('--duration', type=float, default=0.0,
                        help='run duration in seconds; 0 means until ESC/Ctrl+C')
    parser.add_argument('--no-show', dest='show', action='store_false',
                        help='do not show a window; print monitor status only')
    parser.add_argument('--no-mirror', dest='mirror', action='store_false',
                        help='do not mirror the local camera image')
    parser.add_argument('--action-duration', type=float, default=2.0,
                        help='simulated duration of an emotion action in seconds')
    parser.add_argument('--recovery-duration', type=float, default=1.0,
                        help='simulated duration of the stand recovery in seconds')
    parser.add_argument('--neutral-bias', type=float, default=None,
                        help='temporary geometry-model neutral bias for this run')
    parser.add_argument('--boost', action='append', default=[], metavar='EMOTION=VALUE',
                        help='temporary class bias, repeatable (for example: --boost unhappy=0.2)')
    parser.add_argument('--calibration-frames', type=int, default=25,
                        help='stable neutral frames required before classification')
    parser.set_defaults(show=True, mirror=True)
    return parser


def parse_boosts(items):
    boosts = {}
    for item in items:
        if '=' not in item:
            raise ValueError('--boost must use EMOTION=VALUE, got %r' % item)
        emotion, value = item.split('=', 1)
        emotion = emotion.strip()
        if emotion not in FX.EMOTIONS:
            raise ValueError('unknown emotion %r; available: %s'
                             % (emotion, ', '.join(FX.EMOTIONS)))
        boosts[emotion] = float(value)
    return boosts


def try_open_camera(index):
    """Prefer the macOS-native AVFoundation backend, then try OpenCV default."""
    attempts = []
    if hasattr(cv2, 'CAP_AVFOUNDATION'):
        attempts.append((cv2.CAP_AVFOUNDATION, 'AVFOUNDATION'))
    attempts.append((cv2.CAP_ANY, 'ANY'))
    for backend, backend_name in attempts:
        camera = cv2.VideoCapture(index, backend)
        if camera.isOpened():
            ok, frame = camera.read()
            if ok and frame is not None:
                return camera, backend_name
        camera.release()
    return None, None


def open_camera(preferred_index):
    order = [preferred_index] + [index for index in range(4) if index != preferred_index]
    for index in order:
        camera, backend = try_open_camera(index)
        if camera is not None:
            print('camera opened: index=%d backend=%s' % (index, backend))
            return camera
        print('camera unavailable: index=%d' % index)
    print('Unable to open a local camera. On macOS, grant Camera access to Kiro or your terminal:')
    print('System Settings > Privacy & Security > Camera')
    return None


def draw_text(frame, text, origin, color=(230, 230, 230), scale=0.48, thickness=1):
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                cv2.LINE_AA)


def draw_monitor(frame, result, emotion, probabilities, scheduler, fps, banner):
    """Draw landmarks plus policy state, never commanding any robot hardware."""
    height, width = frame.shape[:2]
    snapshot = scheduler.snapshot()

    if result.found and result.bbox is not None:
        x, y, box_w, box_h = result.bbox
        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 235, 0), 2)
    if result.found and result.points5 is not None:
        for index, (point_x, point_y) in enumerate(result.points5):
            cv2.circle(frame, (int(point_x), int(point_y)), 3,
                       POINT_COLORS[index % len(POINT_COLORS)], -1)

    panel_width = min(430, width)
    panel_height = min(height, 298)
    shaded = frame[:panel_height, :panel_width].copy()
    frame[:panel_height, :panel_width] = cv2.addWeighted(
        shaded, 0.30, np.zeros_like(shaded), 0.70, 0)

    selected = emotion or '-'
    selected_color = (0, 235, 0) if selected not in ('-', 'neutral') else (205, 205, 205)
    draw_text(frame, 'EMOTION: %s' % selected.upper(), (10, 27), selected_color, 0.72, 2)

    row_y = 52
    if probabilities:
        for label in sorted(probabilities, key=lambda item: -probabilities[item]):
            probability = probabilities[label]
            color = (0, 235, 0) if label == emotion else (165, 165, 165)
            draw_text(frame, '%-10s %.2f' % (label, probability), (10, row_y), color)
            cv2.rectangle(frame, (142, row_y - 10),
                          (142 + int(100 * probability), row_y), color, -1)
            row_y += 20
    else:
        draw_text(frame, 'waiting for a calibrated face', (10, row_y), (165, 165, 165))
        row_y += 20

    vote_label = snapshot['vote_emotion'] or '-'
    draw_text(frame, 'VOTE: %s %.2f (%d/%d)' % (
        vote_label, snapshot['vote_confidence'], snapshot['vote_count'],
        scheduler.voter.window_size), (10, row_y + 8), (0, 215, 255))
    row_y += 31

    state_color = {
        scheduler.IDLE: (205, 205, 205),
        scheduler.ACTION: (0, 165, 255),
        scheduler.RECOVERY: (255, 180, 0),
        scheduler.COOLDOWN: (0, 215, 255),
    }[snapshot['state']]
    planned = snapshot['current_action'] or '-'
    draw_text(frame, 'STATE: %s   PLAN: %s' % (snapshot['state'], planned),
              (10, row_y), state_color, 0.53, 2)
    row_y += 25

    if snapshot['state'] == scheduler.RECOVERY:
        activity = 'SIMULATING RECOVERY: %s' % snapshot['recovery_action']
    elif snapshot['state'] == scheduler.ACTION:
        activity = 'SIMULATING ACTION: %s' % planned
    else:
        activity = 'NO ROBOT ACTION IS EXECUTED'
    draw_text(frame, activity, (10, row_y), (180, 230, 180), 0.47)
    row_y += 23

    draw_text(frame, 'COOLDOWN: %.1fs   SAME-EMOTION HOLD: %.1fs' % (
        snapshot['cooldown_remaining'], snapshot['same_emotion_remaining']),
              (10, row_y), (0, 215, 255), 0.46)
    row_y += 24
    draw_text(frame, 'FPS %.1f   ESC/Q quit   C recalibrate' % fps,
              (10, row_y), (200, 200, 200), 0.43)

    if result.calibrating:
        cv2.rectangle(frame, (0, height - 40), (width, height), (0, 0, 0), -1)
        draw_text(frame, 'CALIBRATING %d%% - hold a relaxed face still' %
                  int(result.calibration_progress * 100), (10, height - 14),
                  (0, 215, 255), 0.58, 2)
    elif banner:
        cv2.rectangle(frame, (0, height - 40), (width, height), (0, 0, 0), -1)
        draw_text(frame, banner, (10, height - 14), (0, 235, 0), 0.55, 2)
    return frame


def drive_simulated_motion(scheduler, simulation, now, action_duration, recovery_duration):
    """Advance simulated timing while using the scheduler's real state transitions."""
    scheduler.update(now)
    state = scheduler.state
    events = []

    if state == scheduler.ACTION:
        if simulation['phase'] != 'action':
            action = scheduler.mark_action_started()
            if action is not None:
                simulation['phase'] = 'action'
                simulation['deadline'] = now + max(0.0, action_duration)
                events.append('would execute action: %s' % action)
        elif now >= simulation['deadline']:
            if scheduler.mark_action_finished(now):
                simulation['phase'] = 'recovery'
                simulation['deadline'] = now + max(0.0, recovery_duration)
                events.append('would execute recovery: %s' % scheduler.recovery_action)

    elif state == scheduler.RECOVERY:
        if simulation['phase'] != 'recovery':
            simulation['phase'] = 'recovery'
            simulation['deadline'] = now + max(0.0, recovery_duration)
        elif now >= simulation['deadline']:
            if scheduler.mark_recovery_finished(now):
                simulation['phase'] = 'cooldown'
                events.append('recovery complete; cooldown started')

    elif state == scheduler.COOLDOWN:
        simulation['phase'] = 'cooldown'
    else:
        simulation['phase'] = None
        simulation['deadline'] = 0.0

    return events


def format_status(elapsed, now, emotion, score, scheduler):
    """Render elapsed time while querying the policy with its absolute clock."""
    snapshot = scheduler.snapshot(now)
    return ('[%6.1fs] emotion=%-10s score=%.2f state=%-8s plan=%-14s '
            'cooldown=%.1fs same_hold=%.1fs' % (
                elapsed, emotion or '-', score or 0.0, snapshot['state'],
                snapshot['current_action'] or '-', snapshot['cooldown_remaining'],
                snapshot['same_emotion_remaining']))


def run(args):
    try:
        boosts = parse_boosts(args.boost)
        analyzer = FX.ExpressionAnalyzer(
            calibration_frames=args.calibration_frames,
            neutral_bias=args.neutral_bias,
            boosts=boosts,
        )
    except (RuntimeError, ValueError) as error:
        print('Unable to start FaceDetect Mac monitor: %s' % error)
        return 1

    scheduler = EmotionActionScheduler()
    info = analyzer.model.info
    print('FaceDetect Mac monitor: model classes=%s' % ', '.join(analyzer.model.classes))
    print('Shared actions: %s' % flatten_action_map(scheduler.action_map))
    print('Intensity map: %s; strong_threshold=%.2f' % (
        scheduler.action_map, scheduler.strong_threshold))
    print('recovery=%s; cooldown=%.1fs; same-emotion=%.1fs' % (
        scheduler.recovery_action, scheduler.action_cooldown,
        scheduler.same_emotion_interval))
    print('Simulation only: no hiwonder import and no ActionGroupControl call.')

    camera = open_camera(args.camera_index)
    if camera is None:
        analyzer.close()
        return 1

    if args.show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    started_at = time.time()
    last_console_at = 0.0
    fps_started_at = started_at
    fps_frames = 0
    fps = 0.0
    frames = 0
    classified = 0
    simulation = {'phase': None, 'deadline': 0.0}
    banner = ''
    banner_until = 0.0

    print('Look at the camera with a relaxed face while calibration completes. Press C to recalibrate.')
    try:
        while True:
            now = time.time()
            if args.duration > 0 and now - started_at >= args.duration:
                break

            ok, frame = camera.read()
            if not ok or frame is None:
                if now - started_at > 5.0 and frames == 0:
                    print('The camera opened but did not deliver frames.')
                    break
                time.sleep(0.02)
                continue

            frames += 1
            fps_frames += 1
            if now - fps_started_at >= 0.5:
                fps = fps_frames / (now - fps_started_at)
                fps_started_at, fps_frames = now, 0
            if args.mirror:
                frame = cv2.flip(frame, 1)

            result = analyzer.process(frame)
            emotion = None
            score = 0.0
            probabilities = None

            if result.found and not result.calibrating and result.emotion:
                emotion = result.emotion
                score = result.emotion_score
                probabilities = result.emotion_scores
                classified += 1
                scheduler.observe(emotion, score)
                stable, action = scheduler.plan(now)
                if stable is not None:
                    snap = scheduler.snapshot(now)
                    intensity = snap.get('last_stable_intensity') or '-'
                    conf = snap.get('last_stable_confidence') or 0.0
                    if action is None:
                        message = 'stable %s (%s %.2f) -> no action' % (
                            stable, intensity, conf)
                    else:
                        message = 'stable %s (%s %.2f) -> would plan %s' % (
                            stable, intensity, conf, action)
                    print('  >>> %s' % message)
                    banner = message.upper()
                    banner_until = now + 2.0
            elif not result.found:
                # Match FaceDetect: losing a face discards stale classifier votes but
                # does not interrupt an action already in progress.
                scheduler.reset_votes()

            for event in drive_simulated_motion(
                    scheduler, simulation, now, args.action_duration, args.recovery_duration):
                print('  >>> %s' % event)
                banner = event.upper()
                banner_until = now + 2.0

            if now - last_console_at >= 0.5:
                last_console_at = now
                elapsed = now - started_at
                print(format_status(elapsed, now, emotion, score, scheduler))

            if args.show:
                frame = draw_monitor(frame, result, emotion, probabilities, scheduler, fps,
                                     banner if now < banner_until else '')
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
                if key == ord('c'):
                    analyzer.recalibrate()
                    scheduler.reset_votes()
                    print('Recalibrating neutral baseline.')

    except KeyboardInterrupt:
        print('\nStopped with Ctrl+C.')
    finally:
        camera.release()
        analyzer.close()
        if args.show:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    elapsed = max(time.time() - started_at, 1e-6)
    print('Summary: %.1fs, %d frames, %.1f fps, %d classified frames.' %
          (elapsed, frames, frames / elapsed, classified))
    if frames == 0:
        return 1
    print('FaceDetect monitor finished; zero robot actions were executed.')
    return 0


def main():
    return run(build_parser().parse_args())


if __name__ == '__main__':
    sys.exit(main())
