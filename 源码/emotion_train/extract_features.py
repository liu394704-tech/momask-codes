#!/usr/bin/env python3
# coding=utf8
"""
在多人公开数据集上提取 FaceExpression 的 19 个几何特征。

数据集: muxspace/facial_expressions (Apache-2.0)
         https://github.com/muxspace/facial_expressions
文件名为 LFW 风格，编码了人物身份，因此可以按人分组算出真实的个人中性基线。
"""
import os
import re
import csv
import sys
import time
import collections

import cv2
import numpy as np
import mediapipe as mp
import FaceExpression as FX

ROOT = '/tmp/fe/facial_expressions-master'
IMAGES = os.path.join(ROOT, 'images')
LEGEND = '/tmp/legend.csv'
OUT = '/tmp/features.npz'

LABEL_MAP = {
    'happiness': 'happy', 'sadness': 'sad', 'anger': 'angry',
    'surprise': 'surprised', 'disgust': 'disgust', 'fear': 'fearful',
    'neutral': 'neutral',
}

MAX_NEUTRAL_PER_PERSON = 3
MAX_HAPPY_PER_PERSON = 2


def person_of(filename):
    return re.sub(r'[_-]?\d+.*$', '', filename)


rows = list(csv.DictReader(open(LEGEND)))
by_person = collections.defaultdict(list)
for r in rows:
    lab = LABEL_MAP.get(r['emotion'].strip().lower())
    if lab is None:
        continue
    by_person[person_of(r['image'])].append((r['image'], lab))

# 只保留同时有中性和表情的人，这样才能算个人基线
selected = []
for pid, items in by_person.items():
    neutral = [f for f, l in items if l == 'neutral']
    others = [(f, l) for f, l in items if l != 'neutral']
    if not neutral or not others:
        continue
    picked = [(f, 'neutral') for f in neutral[:MAX_NEUTRAL_PER_PERSON]]
    happy_n = 0
    for f, l in others:
        if l == 'happy':
            if happy_n >= MAX_HAPPY_PER_PERSON:
                continue
            happy_n += 1
        picked.append((f, l))
    selected.append((pid, picked))

total = sum(len(p) for _, p in selected)
print('可用人数 %d，待处理图片 %d 张' % (len(selected), total))
sys.stdout.flush()

mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1,
    refine_landmarks=True, min_detection_confidence=0.5)

X, labels, persons = [], [], []
done = fail = 0
t0 = time.time()

for pid, items in selected:
    for fname, lab in items:
        path = os.path.join(IMAGES, fname)
        img = cv2.imread(path)
        if img is None:
            fail += 1
            continue
        # 数据集图片偏小，放大有助于 FaceMesh 定位
        if max(img.shape[:2]) < 400:
            scale = 400.0 / max(img.shape[:2])
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        out = mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not out.multi_face_landmarks:
            fail += 1
            continue
        h, w = img.shape[:2]
        pts = np.array([[lm.x * w, lm.y * h] for lm in out.multi_face_landmarks[0].landmark],
                       dtype=np.float32)
        feats = FX.compute_features(pts)
        if feats is None:
            fail += 1
            continue

        X.append([feats[k] for k in FX.FEATURE_KEYS])
        labels.append(lab)
        persons.append(pid)
        done += 1
        if done % 500 == 0:
            print('  已处理 %d/%d  失败 %d  用时 %.0fs' % (done, total, fail, time.time() - t0))
            sys.stdout.flush()

mesh.close()
X = np.array(X, dtype=np.float64)
np.savez(OUT, X=X, labels=np.array(labels), persons=np.array(persons),
         feature_keys=np.array(FX.FEATURE_KEYS))
print('完成: 成功 %d，检测失败 %d，用时 %.0fs' % (done, fail, time.time() - t0))
print('各标签样本数:', dict(collections.Counter(labels)))
print('保存到', OUT)
