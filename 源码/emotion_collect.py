#!/usr/bin/env python3
# coding=utf8
"""
表情样本采集程序（自标注）(expression sample collector with self-labelling)

用途：让每个人对着摄像头做出指定表情并自己确认标签，把几何特征存成数据集，
      然后用这批真实数据重新训练识别模型。
(purpose: let each person perform a given expression in front of the camera and confirm the label
 themselves, store the geometric features as a dataset, and retrain the recogniser on this real data)

为什么需要它：公开数据集里 sad 只有 194 个样本，且多为名人抓拍照，表情微弱、标签噪声大，
              导致 sad 召回只有 0.12。用你自己的摄像头、光照和表情习惯采集，效果会好得多。
(why this is needed: the public dataset has only 194 sad samples, mostly faint candid celebrity photos
 with noisy labels, which limits sad recall to 0.12. Collecting with your own camera, lighting and
 expression habits works considerably better)

用法(usage):
    python3 emotion_collect.py --person zhangsan     # 采集某个人的样本
    python3 emotion_collect.py --stats               # 只看已有数据统计
    python3 emotion_collect.py --person lisi --seconds 1.5

窗口内按键(keys in the window):
    n  中性 neutral    h  高兴 happy    s  悲伤 sad
    a  生气 angry      u  惊讶 surprised
    按一下即录制一段，不需要按住(press once to record one burst, no need to hold)

    c        重新标定中性基线(recalibrate the neutral baseline)
    DELETE   删除上一段录制(delete the last burst)
    q / ESC  退出(quit)

采集流程(how a burst works):
    按下情绪键 -> 1 秒准备时间（做出表情）-> 录制约 1 秒 -> 自动保存
    (press an emotion key, get 1 second to prepare, then about 1 second of recording is saved)
"""
import os
import sys
import json
import time
import argparse
import collections

FUNCTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TonyPi', 'Functions')
if FUNCTIONS_DIR not in sys.path:
    sys.path.insert(0, FUNCTIONS_DIR)

import cv2
import numpy as np

import FaceExpression as FX

WINDOW = 'emotion collector'
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'emotion_train', 'data')
DATA_FILE = os.path.join(DATA_DIR, 'samples.jsonl')

# 按键 -> 情绪标签(key to emotion label)
# 采集时仍然分别记录悲伤和生气。它们在训练时会合并成"不开心"，
# 但分开采集的数据更细，将来想改回五类不需要重新采。
# (collection still records sadness and anger separately. They merge into unhappy at training time, but
#  keeping them apart in the data means returning to five classes later needs no re-collection)
KEYMAP = {
    'n': 'neutral', 'h': 'happy', 's': 'sad',
    'a': 'angry', 'u': 'surprised',
}
LABEL_ZH = dict(FX.COLLECT_LABEL_ZH)
ORDER = list(FX.COLLECT_LABELS)

# 这两个标签训练时会被合并(these two labels are merged at training time)
MERGED_NOTE = {'sad': '合并为不开心', 'angry': '合并为不开心'}

# 每种情绪建议采够多少样本(suggested number of samples per emotion)
TARGET_PER_LABEL = 60


def build_parser():
    p = argparse.ArgumentParser(description='expression sample collector')
    p.add_argument('--person', help='采集者标识，同一个人请保持一致'
                                    '(subject id, keep it consistent for the same person)')
    p.add_argument('--camera-index', type=int, default=0)
    p.add_argument('--seconds', type=float, default=1.0,
                   help='每段录制时长秒数(seconds recorded per burst)')
    p.add_argument('--prepare', type=float, default=1.0,
                   help='按键后的准备时间秒数(seconds to prepare after pressing a key)')
    p.add_argument('--calibration-frames', type=int, default=25)
    p.add_argument('--stats', action='store_true', help='只打印数据集统计(print dataset stats only)')
    p.add_argument('--data', default=DATA_FILE)
    p.add_argument('--no-mirror', dest='mirror', action='store_false')
    p.set_defaults(mirror=True)
    return p


# ---------------- 数据集读写(dataset I/O) ----------------

def load_samples(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def append_samples(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def rewrite_samples(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def print_stats(path):
    samples = load_samples(path)
    if not samples:
        print('还没有采集任何样本(no samples collected yet):', path)
        return
    by_label = collections.Counter(s['label'] for s in samples)
    by_person = collections.Counter(s['person'] for s in samples)
    print('数据文件(data file): %s' % path)
    print('样本总数(total samples): %d    人数(people): %d' % (len(samples), len(by_person)))
    print()
    print('%-10s %-8s %8s   %s' % ('情绪', 'emotion', '样本数', '进度(建议每人每种 %d)' % TARGET_PER_LABEL))
    for lab in ORDER:
        n = by_label.get(lab, 0)
        target = TARGET_PER_LABEL * max(1, len(by_person))
        bar = '#' * min(30, int(30.0 * n / target)) if target else ''
        note = MERGED_NOTE.get(lab, '')
        print('%-10s %-8s %8d   %-30s %s' % (LABEL_ZH[lab], lab, n, bar, note))
    merged = sum(by_label.get(l, 0) for l in MERGED_NOTE)
    if merged:
        print('%-10s %-8s %8d   (悲伤+生气，训练时作为一类)' % ('不开心', 'unhappy', merged))
    print()
    print('按人统计(per person):')
    for person, n in by_person.most_common():
        labs = collections.Counter(s['label'] for s in samples if s['person'] == person)
        missing = [LABEL_ZH[l] for l in ORDER if labs.get(l, 0) < 10]
        note = ('  缺少: ' + ' '.join(missing)) if missing else '  齐全'
        print('  %-14s %5d 样本%s' % (person, n, note))
    people_with_all = sum(1 for p in by_person
                          if all(sum(1 for s in samples if s['person'] == p and s['label'] == l) >= 10
                                 for l in ORDER[:5]))
    print()
    if len(by_person) < 2:
        print('提示：目前只有 %d 个人的数据。跨人泛化至少需要 3~5 个人，'
              '请让不同的人各采一轮。' % len(by_person))
    else:
        print('已有 %d 个人的数据，其中 %d 人主要情绪齐全。' % (len(by_person), people_with_all))


# ---------------- 摄像头(camera) ----------------

def open_camera(index):
    for backend, name in ((cv2.CAP_AVFOUNDATION, 'AVFOUNDATION'), (cv2.CAP_ANY, 'ANY')):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                print('摄像头已打开(camera opened): index=%d backend=%s' % (index, name))
                return cap
        cap.release()
    return None


# ---------------- 画面(overlay) ----------------

def draw(frame, result, counts, person, state, state_left, last_label, live_pred, total):
    h, w = frame.shape[:2]

    if result.bbox is not None:
        color = (0, 235, 0) if state == 'recording' else (200, 200, 200)
        x, y, bw, bh = result.bbox
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
    if result.points5 is not None:
        for px, py in result.points5:
            cv2.circle(frame, (int(px), int(py)), 3, (0, 200, 255), -1)

    # 左侧面板：各情绪已采样本数(left panel: sample count per emotion)
    panel_w, panel_h = 232, 40 + len(ORDER) * 22 + 26
    sub = frame[0:panel_h, 0:panel_w].copy()
    frame[0:panel_h, 0:panel_w] = cv2.addWeighted(sub, 0.3, np.zeros_like(sub), 0.7, 0)
    cv2.putText(frame, 'person: %s' % person, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, 'total %d' % total, (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1)
    for i, lab in enumerate(ORDER):
        y0 = 58 + i * 22
        n = counts.get(lab, 0)
        done = n >= TARGET_PER_LABEL
        c = (0, 235, 0) if done else ((0, 200, 255) if n else (150, 150, 150))
        key = [k for k, v in KEYMAP.items() if v == lab][0]
        cv2.putText(frame, '[%s] %-10s %3d' % (key, lab, n), (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
        bar = int(70 * min(1.0, n / float(TARGET_PER_LABEL)))
        cv2.rectangle(frame, (156, y0 - 8), (156 + bar, y0 + 1), c, -1)

    # 底部状态条(bottom status bar)
    cv2.rectangle(frame, (0, h - 46), (w, h), (0, 0, 0), -1)
    if state == 'calibrating':
        msg = 'CALIBRATING %d%% - relaxed face, hold still' % int(result.calibration_progress * 100)
        col = (0, 215, 255)
    elif state == 'prepare':
        msg = 'GET READY: %s  in %.1fs' % ((last_label or '').upper(), state_left)
        col = (0, 215, 255)
    elif state == 'recording':
        msg = 'RECORDING %s ... %.1fs' % ((last_label or '').upper(), state_left)
        col = (0, 235, 0)
    elif state == 'noface':
        msg = 'NO FACE - face the camera'
        col = (80, 80, 255)
    else:
        msg = 'press n/h/s/a/u to record   c recalibrate   DEL undo   q quit'
        col = (210, 210, 210)
    cv2.putText(frame, msg, (10, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    if live_pred:
        cv2.putText(frame, 'current model thinks: %s' % live_pred, (10, h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)
    return frame


# ---------------- 主流程(main loop) ----------------

def collect(args):
    person = args.person
    samples = load_samples(args.data)
    counts = collections.Counter(s['label'] for s in samples if s['person'] == person)

    cap = open_camera(args.camera_index)
    if cap is None:
        print('无法打开摄像头。若是权限问题，请在 系统设置 > 隐私与安全性 > 摄像头 中授权。')
        print('(cannot open the camera; grant access in System Settings > Privacy & Security > Camera)')
        return 1

    # 采集时必须关掉基线漂移，否则基线会慢慢把正在做的表情吸收进去，标签就不准了
    # (baseline drift must be off while collecting, otherwise the baseline slowly absorbs the expression
    #  being held and the labels stop matching the data)
    analyzer = FX.ExpressionAnalyzer(calibration_frames=args.calibration_frames,
                                     drift_rate=0.0)

    print()
    print('采集者(person): %s' % person)
    print('先保持自然表情不要动，等基线标定完成；标定不理想按 c 重来。')
    print('然后按情绪键录制：n中性 h高兴 s悲伤 a生气 u惊讶')
    print('注意：悲伤和生气在训练时会合并成"不开心"，但仍分别采集，方便将来改回五类。')
    print('每按一次会给 %.1f 秒准备时间，然后录制 %.1f 秒。' % (args.prepare, args.seconds))
    print('-' * 74)
    sys.stdout.flush()

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    state = 'idle'
    state_until = 0.0
    pending_label = None
    burst = []
    burst_ids = []
    new_records = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            if args.mirror:
                frame = cv2.flip(frame, 1)

            result = analyzer.process(frame)
            now = time.time()

            live_pred = ''
            if result.found and not result.calibrating and result.emotion:
                live_pred = '%s %.2f' % (result.emotion, result.emotion_score)

            # 状态机(state machine)
            if not result.found:
                shown_state = 'noface'
                if state in ('prepare', 'recording'):
                    print('  中途丢失人脸，本段作废(face lost, burst discarded)')
                    state, pending_label, burst = 'idle', None, []
            elif result.calibrating:
                shown_state = 'calibrating'
            elif state == 'prepare':
                shown_state = 'prepare'
                if now >= state_until:
                    state = 'recording'
                    state_until = now + args.seconds
                    burst = []
            elif state == 'recording':
                shown_state = 'recording'
                if result.deviations is not None:
                    burst.append((dict(result.features), dict(result.deviations)))
                if now >= state_until:
                    if burst:
                        ts = time.time()
                        base = analyzer.baseline
                        recs = []
                        for feats, devs in burst:
                            recs.append({
                                'person': person,
                                'label': pending_label,
                                'time': ts,
                                'features': {k: round(feats[k], 6) for k in FX.FEATURE_KEYS},
                                'baseline': {k: round(base[k], 6) for k in FX.FEATURE_KEYS},
                                'deviations': {k: round(devs[k], 6) for k in FX.FEATURE_KEYS},
                            })
                        append_samples(args.data, recs)
                        new_records.extend(recs)
                        burst_ids.append(len(recs))
                        counts[pending_label] += len(recs)
                        print('  已保存 %s(%s) %d 帧   该情绪累计 %d'
                              % (LABEL_ZH[pending_label], pending_label,
                                 len(recs), counts[pending_label]))
                        sys.stdout.flush()
                    else:
                        print('  本段没有有效帧(no valid frames in this burst)')
                    state, pending_label, burst = 'idle', None, []
            else:
                shown_state = 'idle'

            total = sum(counts.values())
            frame = draw(frame, result, counts, person, shown_state,
                         max(0.0, state_until - now), pending_label, live_pred, total)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord('c'):
                analyzer.recalibrate()
                state, pending_label, burst = 'idle', None, []
                print('重新标定中性基线(recalibrating the neutral baseline)')
            elif key in (8, 127):          # BACKSPACE / DELETE
                if burst_ids:
                    n = burst_ids.pop()
                    all_recs = load_samples(args.data)
                    removed = all_recs[-n:]
                    rewrite_samples(args.data, all_recs[:-n])
                    for r in removed:
                        counts[r['label']] -= 1
                    print('已删除上一段 %d 帧(%s)' % (n, removed[0]['label'] if removed else '?'))
                else:
                    print('本次运行还没有可删除的录制(nothing recorded yet in this run)')
            elif state == 'idle' and not result.calibrating and result.found:
                ch = chr(key) if 32 <= key < 127 else ''
                if ch in KEYMAP:
                    pending_label = KEYMAP[ch]
                    state = 'prepare'
                    state_until = now + args.prepare
                    print('准备录制 %s(%s) ...' % (LABEL_ZH[pending_label], pending_label))
                    sys.stdout.flush()

    except KeyboardInterrupt:
        print('\n收到 Ctrl+C(received Ctrl+C)')
    finally:
        cap.release()
        analyzer.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    print('-' * 74)
    print('本次新增 %d 帧样本，累计文件: %s' % (len(new_records), args.data))
    print()
    print_stats(args.data)
    print()
    print('下一步：用采集到的数据重新训练')
    print('  cd emotion_train')
    print('  PYTHONPATH=../TonyPi/Functions ../.venv-emotion/bin/python train_model.py --local')
    return 0


def main():
    args = build_parser().parse_args()
    if args.stats:
        print_stats(args.data)
        return 0
    if not args.person:
        print('请用 --person 指定采集者标识，例如 --person zhangsan')
        print('(specify the subject with --person, e.g. --person zhangsan)')
        return 1
    return collect(args)


if __name__ == '__main__':
    sys.exit(main())
