#!/usr/bin/env python3
# coding=utf8
"""
表情识别实时验证程序(live expression recognition test)

识别 4 种情绪：中性 / 高兴 / 不开心 / 惊讶
(recognises four emotions: neutral, happy, unhappy, surprised)

识别链路(pipeline):
    摄像头 -> FaceMesh(468 点) -> 21 个几何特征 -> 减去个人中性基线
           -> 多人数据训练的线性模型 -> 4 类情绪
                                    -> 面部动作标签(用于看清判断依据)

类别是逐步收敛到 4 类的，每一步都有实测依据：
  7 类 -> 5 类：disgust 跨人召回始终 0.00，fearful 公开数据只有 10 个样本
  5 类 -> 4 类：sad 与 angry 无法可靠区分(617 帧 angry 有 317 帧判成 sad)，合并为 unhappy
(the class set converged to four in measured steps: disgust recall stayed at 0.00 across people and the
 public dataset holds only 10 fearful samples, so seven became five; then sad and angry proved
 inseparable, with 317 of 617 angry frames labelled sad, so five became four with the two merged into
 unhappy)

用法(usage):
    python3 emotion_mac_test.py                     # 默认弹窗
    python3 emotion_mac_test.py --no-show           # 只打印
    python3 emotion_mac_test.py --diagnose          # 排查弹窗与摄像头
    python3 emotion_mac_test.py --neutral-bias 0.0  # 更灵敏，表情更容易触发
    python3 emotion_mac_test.py --neutral-bias 1.0  # 更保守，中性更多
    python3 emotion_mac_test.py --boost sad=0.5     # 单独提高某个情绪

窗口内按键(keys):
    ESC / q  退出        c  重新标定中性基线
"""
import os
import sys
import time
import argparse

FUNCTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TonyPi', 'Functions')
if FUNCTIONS_DIR not in sys.path:
    sys.path.insert(0, FUNCTIONS_DIR)

import cv2
import numpy as np

import FaceExpression as FX
import EmotionRecognition as ER

WINDOW = 'emotion test'
POINT_COLORS = [(0, 0, 255), (0, 128, 255), (0, 255, 255), (255, 0, 255), (255, 128, 0)]


def build_parser():
    p = argparse.ArgumentParser(description='live expression recognition test')
    p.add_argument('--camera-index', type=int, default=0)
    p.add_argument('--duration', type=float, default=0.0,
                   help='运行秒数，0 表示一直运行(seconds to run, 0 means forever)')
    p.add_argument('--no-show', dest='show', action='store_false',
                   help='不弹窗，只打印(no window, print only)')
    p.add_argument('--diagnose', action='store_true',
                   help='排查弹窗与摄像头问题(diagnose window and camera problems)')
    p.add_argument('--neutral-bias', type=float, default=None,
                   help='中性偏置，调大更保守、调小更灵敏'
                        '(neutral bias; larger is more conservative, smaller more sensitive)')
    p.add_argument('--boost', action='append', default=[], metavar='EMOTION=VALUE',
                   help='单独提高某个情绪的倾向，可重复，例如 --boost sad=0.5'
                        '(bias one emotion upward, repeatable)')
    p.add_argument('--sensitivity', type=float, default=1.0,
                   help='动作标签的灵敏度，只影响显示(sensitivity of the action tags, display only)')
    p.add_argument('--calibration-frames', type=int, default=25)
    p.add_argument('--no-mirror', dest='mirror', action='store_false')
    p.set_defaults(show=True, mirror=True)
    return p


def parse_boosts(items):
    out = {}
    for item in items:
        if '=' not in item:
            raise SystemExit('--boost 需要写成 情绪=数值，例如 sad=0.5 (got %r)' % item)
        name, value = item.split('=', 1)
        name = name.strip()
        if name not in FX.EMOTIONS:
            raise SystemExit('未知情绪 %r，可用: %s' % (name, ', '.join(FX.EMOTIONS)))
        out[name] = float(value)
    return out


# ---------------- 摄像头(camera) ----------------

def try_open(index):
    for backend, name in ((cv2.CAP_AVFOUNDATION, 'AVFOUNDATION'), (cv2.CAP_ANY, 'ANY')):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, name
        cap.release()
    return None, None


def open_camera(preferred):
    for index in [preferred] + [i for i in range(4) if i != preferred]:
        cap, backend = try_open(index)
        if cap is not None:
            print('摄像头已打开(camera opened): index=%d backend=%s' % (index, backend))
            return cap
        print('摄像头编号 %d 打不开(camera index %d unavailable)' % (index, index))
    return None


def permission_hint():
    print('')
    print('摄像头无法使用，最常见原因是 macOS 摄像头权限没给运行本脚本的程序。')
    print('请打开：系统设置 > 隐私与安全性 > 摄像头，允许 Kiro（或你使用的终端），然后重新运行。')
    print('(grant access in System Settings > Privacy & Security > Camera, then rerun)')
    print('')


def run_diagnose():
    print('=' * 70)
    print('第 1 步：OpenCV 的 GUI 后端')
    for line in cv2.getBuildInformation().splitlines():
        if 'GUI' in line or 'Cocoa' in line or 'AVFoundation' in line:
            print('   ' + line.strip())

    print('')
    print('第 2 步：弹一个测试窗口，持续 4 秒。看到彩色窗口说明弹窗正常。')
    canvas = np.zeros((260, 620, 3), dtype=np.uint8)
    for y in range(260):
        canvas[y, :] = (int(255 * y / 260), 90, 255 - int(255 * y / 260))
    cv2.putText(canvas, 'WINDOW TEST OK', (95, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    window_ok = False
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        deadline = time.time() + 4.0
        while time.time() < deadline:
            cv2.imshow(WINDOW, canvas)
            if cv2.waitKey(30) != -1:
                break
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        window_ok = True
        print('   imshow 调用成功')
    except Exception as e:
        print('   imshow 失败: %s: %s' % (type(e).__name__, e))

    print('')
    print('第 3 步：逐个尝试摄像头编号')
    found = []
    for index in range(4):
        cap, backend = try_open(index)
        if cap is not None:
            print('   index=%d 可用 backend=%s  %dx%d'
                  % (index, backend, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            found.append(index)
            cap.release()
        else:
            print('   index=%d 不可用' % index)

    print('')
    print('=' * 70)
    print('窗口功能: %s     可用摄像头: %s'
          % ('正常' if window_ok else '异常', found if found else '无'))
    if not found:
        permission_hint()
        return 1
    return 0


# ---------------- 画面(overlay) ----------------

def draw(frame, result, emotion, probs, actions, fps, banner):
    h, w = frame.shape[:2]

    if result.bbox is not None:
        x, y, bw, bh = result.bbox
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 235, 0), 2)
    if result.points5 is not None:
        for i, (px, py) in enumerate(result.points5):
            cv2.circle(frame, (int(px), int(py)), 3, POINT_COLORS[i], -1)

    panel_w = 236
    panel_h = 40 + len(FX.EMOTIONS) * 22 + 24
    sub = frame[0:panel_h, 0:panel_w].copy()
    frame[0:panel_h, 0:panel_w] = cv2.addWeighted(sub, 0.3, np.zeros_like(sub), 0.7, 0)

    head = emotion or '-'
    color = (0, 235, 0) if head not in ('-', 'neutral') else (200, 200, 200)
    cv2.putText(frame, head.upper(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    if probs:
        for row, name in enumerate(sorted(probs, key=lambda k: -probs[k])):
            value = probs[name]
            y0 = 54 + row * 22
            c = (0, 235, 0) if name == emotion else (155, 155, 155)
            cv2.rectangle(frame, (132, y0 - 9), (132 + int(90 * value), y0 + 1), c, -1)
            cv2.putText(frame, '%-10s' % name, (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
            cv2.putText(frame, '%.2f' % value, (92, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1)
    else:
        cv2.putText(frame, 'no face', (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.putText(frame, 'fps %.1f   ESC quit   c recalibrate' % fps,
                (10, panel_h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    if result.calibrating:
        cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, 'calibrating %d%% - hold still, relaxed face'
                    % int(result.calibration_progress * 100),
                    (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
        return frame

    for row, key in enumerate(actions):
        text = FX.describe_action(key, chinese=False)
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        x0 = w - size[0] - 20
        y0 = 30 + row * 26
        cv2.rectangle(frame, (x0 - 8, y0 - 18), (w - 8, y0 + 8), (40, 110, 40), -1)
        cv2.putText(frame, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    if banner:
        cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, 'STABLE: %s' % banner, (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 235, 0), 2)
    return frame


# ---------------- 主循环(main loop) ----------------

def run(args):
    try:
        analyzer = FX.ExpressionAnalyzer(calibration_frames=args.calibration_frames,
                                        sensitivity=args.sensitivity,
                                        neutral_bias=args.neutral_bias,
                                        boosts=parse_boosts(args.boost))
    except RuntimeError as e:
        print(e)
        print('请先训练模型: cd emotion_train && '
              'PYTHONPATH=../TonyPi/Functions ../.venv-emotion/bin/python train_model.py')
        return 1

    info = analyzer.model.info
    print('情绪类别(emotions): %s' % ', '.join(FX.EMOTIONS))
    print('模型(model): %s 人 / %s 样本   评估方式 %s   中性偏置 %.2f'
          % (info.get('people', '?'), info.get('samples', '?'),
             info.get('evaluation', '?'), analyzer.model.neutral_bias))
    if analyzer.model.boosts:
        print('情绪偏置(boosts): %s' % analyzer.model.boosts)

    cap = open_camera(args.camera_index)
    if cap is None:
        permission_hint()
        return 1

    voter = ER.EmotionVoter(window_size=8, min_votes=6, min_confidence=0.45, cooldown=4.0)

    if args.show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    start = time.time()
    last_print = 0.0
    frames = inferences = stable_hits = 0
    fps = 0.0
    fps_t0 = time.time()
    fps_n = 0
    banner, banner_until = '', 0.0
    last_actions_printed = None

    print('-' * 70)
    print('先正对镜头、保持自然表情不要动，等基线标定完成。标定不理想按 c 重来。')
    sys.stdout.flush()

    try:
        while True:
            now = time.time()
            if args.duration > 0 and now - start >= args.duration:
                break

            ok, frame = cap.read()
            if not ok or frame is None:
                if now - start > 5.0 and frames == 0:
                    print('摄像头打开了但读不到画面')
                    permission_hint()
                    break
                time.sleep(0.02)
                continue

            frames += 1
            fps_n += 1
            if now - fps_t0 >= 0.5:
                fps = fps_n / (now - fps_t0)
                fps_t0, fps_n = now, 0

            if args.mirror:
                frame = cv2.flip(frame, 1)

            result = analyzer.process(frame)
            elapsed = now - start

            emotion, probs, actions = None, None, []
            if result.found and not result.calibrating and result.emotion:
                emotion = result.emotion
                probs = result.emotion_scores
                actions = result.actions
                inferences += 1
                voter.push(emotion, result.emotion_score)

                # 每 0.3 秒打印一行，避免刷屏
                if now - last_print >= 0.3:
                    last_print = now
                    top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
                    print('[%6.1fs] %-10s %.2f   %s   动作: %s'
                          % (elapsed, emotion, result.emotion_score,
                             '  '.join('%s=%.2f' % (k, v) for k, v in top),
                             ' | '.join(FX.describe_action(a) for a in actions) or '无'))

                stable = voter.vote(now)
                if stable is not None:
                    stable_hits += 1
                    mapped = ER.DEFAULT_EMOTION_ACTIONS.get(stable)
                    banner = stable + (' -> %s' % mapped if mapped else ' (no action)')
                    banner_until = now + 2.0
                    print('  >>> 稳定情绪(stable): %s   机器人上会执行(would run): %s'
                          % (stable, mapped or '无(none)'))
            elif not result.found:
                voter.reset()
                if last_actions_printed != 'noface' and elapsed > 2.0:
                    last_actions_printed = 'noface'
                    print('[%6.1fs] 未检测到人脸(no face)' % elapsed)

            if result.found:
                last_actions_printed = None

            if args.show:
                frame = draw(frame, result, emotion, probs, actions, fps,
                             banner if now < banner_until else '')
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
                if key == ord('c'):
                    analyzer.recalibrate()
                    voter.reset()
                    print('重新标定中性基线(recalibrating)')

    except KeyboardInterrupt:
        print('\n收到 Ctrl+C')
    finally:
        cap.release()
        analyzer.close()
        if args.show:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    total = max(time.time() - start, 1e-6)
    print('-' * 70)
    print('统计: %.1fs  帧数 %d  帧率 %.1f  判定次数 %d  稳定触发 %d'
          % (total, frames, frames / total, inferences, stable_hits))
    if frames == 0:
        print('结果: 失败，没有读到任何画面')
        return 1
    if inferences == 0:
        print('结果: 摄像头正常，但没完成判定。请正对镜头，等基线标定完成。')
        return 1
    print('结果: 链路跑通(PIPELINE OK)')
    return 0


def main():
    args = build_parser().parse_args()
    if args.diagnose:
        return run_diagnose()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
