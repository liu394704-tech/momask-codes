#!/usr/bin/env python3
# coding=utf8
"""
训练几何情绪分类器（多分类逻辑回归，纯 numpy）。

数据来源可以是两者之一或组合:
  --public  公开数据集提取出的特征(extract_features.py 产出的 /tmp/features.npz)
  --local   emotion_collect.py 采集的自标注数据(emotion_train/data/samples.jsonl)

关键设计:
  1. 输入是"相对个人中性基线的偏移量"。基线必须来自本人，不能用别人的数据替代；
     模型学的是"多大的偏移、什么组合算哪种情绪"，这部分才需要多人数据。
  2. 训练/测试按"人"划分。人数少时自动改用留一人交叉验证(LOPO)，
     这样每个人都当过一次测试集，指标才反映对陌生人的泛化能力。
  3. 按类别频率加权，避免样本多的类别压倒稀有情绪。
  4. 用 F1 选操作点，因为召回可以靠"什么都报"刷高。

用法(usage):
    python3 train_model.py --public                  # 只用公开数据集
    python3 train_model.py --local                   # 只用自己采集的数据
    python3 train_model.py --public --local          # 两者合并（推荐）
    python3 train_model.py --local --local-weight 3  # 提高自采数据的权重
"""
import os
import sys
import json
import argparse
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FUNCTIONS_DIR = os.path.join(HERE, '..', 'TonyPi', 'Functions')
if FUNCTIONS_DIR not in sys.path:
    sys.path.insert(0, FUNCTIONS_DIR)

import FaceExpression as FX

PUBLIC_NPZ = '/tmp/features.npz'
LOCAL_JSONL = os.path.join(HERE, 'data', 'samples.jsonl')
DEFAULT_OUT = os.path.join(HERE, '..', 'TonyPi', 'Functions', 'model',
                           'geometry_emotion_model.json')

# 只训练这 5 类。数据里其它标签(disgust / fearful)会被忽略但不会删除，
# 想恢复只需改这一行。
# (only these five classes are trained. Other labels such as disgust and fearful are ignored but not
#  deleted from the data, so restoring them means changing only this line)
KEEP_CLASSES = tuple(FX.EMOTIONS)

# 细分标签 -> 模型类别。sad 和 angry 合并为 unhappy，因为实测两者在几何上无法可靠区分。
# 采集的数据文件里仍然是细分标签，合并只发生在这里，改这一行就能改回五类。
# (fine-grained label to model class. sad and angry merge into unhappy because measurements show the two
#  cannot be separated reliably from geometry. The collected data files keep the fine-grained labels and
#  the merge happens only here, so changing this one mapping restores the five-class setup)
LABEL_MERGE = dict(FX.LABEL_MERGE)


def merge_label(label):
    return LABEL_MERGE.get(label)

# 公开数据集里不可靠的类别。
# 实测：公开数据的 sadness 多为名人抓拍照，表情极微弱、几何上接近中性，
# 跨人 F1 只有 0.09；把它混进训练会教会模型"sad 约等于 neutral"，
# 结果自采数据上的 sad 召回从 1.00 掉到 0.08。
# 因此当自采数据已经提供足够的该类样本时，就不再使用公开数据的这一类。
# neutral / happy / surprised 则相反，公开数据提供了宝贵的跨人多样性，保留。
# (classes that are unreliable in the public dataset. Measurements show its sadness images are faint
#  candid celebrity photos that sit close to neutral in geometry, with a cross-person F1 of only 0.09.
#  Mixing them in teaches the model that sad is roughly neutral, which dropped sad recall on the locally
#  collected data from 1.00 to 0.08. So once the local data supplies enough samples of such a class, the
#  public version of it is skipped. neutral, happy and surprised are the opposite: the public data adds
#  valuable cross-person diversity and is kept)
# 公开数据里不可靠的原始类别（注意是采集标签，不是合并后的类别）。
# 实测公开数据的 sadness 多为名人抓拍照，表情极微弱、几何上接近中性，
# 混进训练会把模型往 neutral 拉。
# (unreliable raw labels in the public data, note these are collection labels rather than merged classes.
#  Measurements show its sadness images are faint candid celebrity photos close to neutral in geometry,
#  and mixing them in pulls the model toward neutral)
PUBLIC_UNRELIABLE = ('sad',)
LOCAL_ENOUGH = 100

# ---------------- 眉毛遮挡增强(brow occlusion augmentation) ----------------
#
# 实测问题：眉毛被头发或手遮挡时，FaceMesh 会把发际边缘当成眉毛，检测到的位置偏低、
# 形状被压平。而 angry 的权重里 brow_raise 是 -1.13（第二大），
# 于是"眉毛看起来变低"会强烈推向生气 —— 即使嘴在笑。
# 用真实 happy 样本模拟：眉毛下移 0.06 时，43% 的笑脸被判成生气；下移 0.09 时 62%。
# (measured problem: when the brows are covered by hair or a hand, FaceMesh takes the hairline edge for
#  the brow, so the detected position sits lower and flatter. Since brow_raise carries a weight of -1.13
#  for angry, the second largest, a brow that merely looks lower pushes strongly toward anger even while
#  the mouth is smiling. Simulating this on real happy samples flipped 43% of them to angry at a drop of
#  0.06 and 62% at 0.09)
#
# 解决办法：训练时按这个方式人为破坏眉毛特征，标签保持不变。
# 模型因此学会"眉毛不可靠时要更多依赖嘴和眼睛"。
# (the fix is to corrupt the brow features this way during training while keeping the label, so the model
#  learns to lean more on the mouth and eyes when the brows are unreliable)
BROW_FEATURES = ('brow_raise', 'brow_inner_h', 'brow_outer_h', 'brow_slope', 'brow_inner_dist')

# ---------------- 模型实际使用哪些特征(which features the model actually uses) ----------------
#
# 默认不使用眉毛特征。原因是实测下来眉毛是净负担：
#   方案                     平均   unhappy  surprised  遮挡误判
#   21特征 无增强            0.49    0.24     0.46       74%
#   21特征 + 遮挡增强        0.50    0.31     0.41       11%
#   16特征(去掉眉毛)         0.50    0.34     0.32        0%
# 去掉眉毛后总体准确率不变、unhappy 反而更好，而"眉毛被遮挡误判成不开心"这个问题
# 从"统计上压低到 11%"变成"结构上不可能"。代价只有 surprised 掉 9 个点。
# (brow features are unused by default because measurements show they are a net liability. Dropping them
#  leaves overall accuracy unchanged, improves unhappy recall, and turns the occluded-brow failure from
#  something merely reduced to 11% into something structurally impossible. The only cost is 9 points of
#  surprised recall)
#
# 想恢复眉毛特征就用 --features all，遮挡增强会自动生效。
# (pass --features all to restore them; the occlusion augmentation then applies automatically)
FEATURE_SETS = {
    'no-brow': tuple(k for k in FX.FEATURE_KEYS if k not in BROW_FEATURES),
    'all': tuple(FX.FEATURE_KEYS),
}
DEFAULT_FEATURE_SET = 'no-brow'
# 强度按实测扫描选定。代价是 angry 召回从 0.66 降到 0.61、sad 从 0.75 降到 0.69，
# 换来的是重度遮挡下的误判率从 94% 降到 2%，这个交换很值。
# (the strength was chosen by measurement. It costs angry recall 0.66 -> 0.61 and sad 0.75 -> 0.69, in
#  exchange for the error rate under heavy occlusion falling from 94% to 2%, which is well worth it)
# 下移范围要覆盖到 0.18：只训练到 0.12 时，下移 0.14 的极端遮挡仍有 22% 误判，
# 因为超出了训练覆盖的范围。
# (the shift range must reach 0.18. Training only up to 0.12 still left a 22% error rate at an extreme
#  shift of 0.14, because that lies outside the trained range)
AUGMENT_RATIO = 1.0          # 额外生成多少比例的遮挡样本(share of extra occluded samples)

# 偏移幅度以"归一化后的标准差"为单位，不用原始数值。
# 因为按人做了尺度归一化后，同一个原始偏移量对不同人对应的归一化幅度差别很大
# （原始 0.02~0.18 换算过来是 0.33~5.0），覆盖不一致就会留下漏洞：
# 实测用原始单位时，极端遮挡下的误判率从 2% 反弹到 19%。
# (the shift is expressed in units of the normalised standard deviation rather than raw values. After the
#  per-person scale normalisation the same raw shift maps to very different normalised magnitudes across
#  people, 0.02 to 0.18 becoming 0.33 to 5.0, and that uneven coverage leaves gaps: with raw units the
#  error rate under extreme occlusion bounced back from 2% to 19%)
# 上限取 6.0 而不是 4.5：自身表情幅度小于群体平均的人，同样的原始遮挡量换算成
# 归一化幅度会更大，4.5 覆盖不到，实测极端遮挡下仍有 9% 误判。
# (the upper bound is 6.0 rather than 4.5 because for someone whose own expression magnitude is below the
#  population average the same raw occlusion maps to a larger normalised shift, which 4.5 fails to cover
#  and which measured a 9% error rate under extreme occlusion)
AUGMENT_DROP_Z = (0.4, 6.0)


def augment_brow_occlusion_z(D, Y, persons, Wt, bursts, keys,
                             ratio=AUGMENT_RATIO, seed=0):
    """
    在归一化空间生成眉毛被遮挡的增强样本。
    (generate brow-occlusion samples in the normalised space)

    必须在按人尺度归一化之后调用。
    (must be called after the per-person scale normalisation)
    """
    # 特征集里没有眉毛特征时无需增强：遮挡根本影响不到模型
    # (no augmentation is needed when the feature set has no brow features, since occlusion cannot reach
    #  the model at all)
    if ratio <= 0 or not any(k in keys for k in BROW_FEATURES):
        return None
    idx = {k: i for i, k in enumerate(keys)}
    rng = np.random.default_rng(seed)
    pick = rng.random(len(D)) <= ratio
    n = int(pick.sum())
    if n == 0:
        return None

    Da = D[pick].copy()
    drop = rng.uniform(AUGMENT_DROP_Z[0], AUGMENT_DROP_Z[1], size=n)
    for k in ('brow_raise', 'brow_inner_h', 'brow_outer_h'):
        Da[:, idx[k]] -= drop
    Da[:, idx['brow_slope']] *= rng.uniform(0.0, 0.4, size=n)
    Da[:, idx['brow_inner_dist']] += rng.normal(0, 0.25, size=n)
    return Da, Y[pick], persons[pick], Wt[pick], bursts[pick]

# 样本太少无法训练的类别会被自动剔除
MIN_SAMPLES_PER_CLASS = 30

# 尺度归一化的收缩强度。实测 m=100~200 效果相当，取 200 更保守，
# 因为实时启动时估计不准，靠群体先验更安全。
# (shrinkage strength for the scale normalisation. Measurements showed m of 100 to 200 perform about the
#  same, and 200 is the more conservative pick since the runtime estimate is poor at start-up and leaning
#  on the population prior is safer)
SCALE_PRIOR_M = 200.0


def build_parser():
    p = argparse.ArgumentParser(description='train the geometry emotion classifier')
    p.add_argument('--public', action='store_true', help='使用公开数据集特征(use the public dataset)')
    p.add_argument('--local', action='store_true', help='使用自采数据(use the locally collected data)')
    p.add_argument('--public-npz', default=PUBLIC_NPZ)
    p.add_argument('--local-jsonl', default=LOCAL_JSONL)
    # 实测：3 人 2917 帧自采 vs 2736 帧公开数据，两边体量本来就接近，
    # 权重 1 时留一人平均 0.573，加到 3 降到 0.563，加到 5 降到 0.537——
    # 上调权重等于过拟合这 3 个人。人数变多后可以再试。
    # (measured: 2917 local frames from three people against 2736 public ones are already balanced. A
    #  weight of 1 gives a leave-one-person-out mean of 0.573, 3 drops it to 0.563 and 5 to 0.537, so
    #  raising the weight amounts to overfitting these three people. Worth revisiting with more subjects)
    p.add_argument('--local-weight', type=float, default=1.0,
                   help='自采样本的额外权重(extra weight for locally collected samples)')
    p.add_argument('--out', default=DEFAULT_OUT)
    p.add_argument('--neutral-bias', type=float, default=0.5,
                   help='导出模型的默认中性偏置(default neutral bias of the exported model)')
    p.add_argument('--features', choices=sorted(FEATURE_SETS), default=DEFAULT_FEATURE_SET,
                   help='模型使用的特征集，no-brow 表示不用眉毛特征'
                        '(feature set used by the model; no-brow excludes the brow features)')
    p.add_argument('--public-drop', default=','.join(PUBLIC_UNRELIABLE),
                   help='弃用公开数据里的这些原始类别，逗号分隔，留空表示都用'
                        '(drop these raw labels from the public data, comma separated)')
    p.add_argument('--augment', type=float, default=AUGMENT_RATIO,
                   help='眉毛遮挡增强的比例，0 表示关闭'
                        '(share of brow-occlusion augmented samples, 0 disables it)')
    p.add_argument('--dry-run', action='store_true', help='只评估，不写出模型(evaluate only)')
    return p


# ---------------- 载入数据(loading) ----------------

def load_public(path):
    """公开数据集：按人分组算中性基线，再转成偏移量"""
    if not os.path.exists(path):
        print('找不到公开数据集特征 %s，跳过。可先运行 extract_features.py' % path)
        return []
    d = np.load(path, allow_pickle=True)
    X, labels, persons = d['X'], d['labels'], d['persons']
    keys = [str(k) for k in d['feature_keys']]
    if keys != list(FX.FEATURE_KEYS):
        print('警告：公开数据集特征与当前代码不一致，请重新运行 extract_features.py')
        return []

    by_person = collections.defaultdict(list)
    for i, p in enumerate(persons):
        by_person[str(p)].append(i)

    out = []
    for p, idxs in by_person.items():
        nidx = [i for i in idxs if labels[i] == 'neutral']
        if not nidx:
            continue
        base_all = np.median(X[nidx], axis=0)
        for i in idxs:
            lab = merge_label(str(labels[i]))
            if lab is None or lab not in KEEP_CLASSES:
                continue
            if lab == 'neutral':
                # 留一法，否则中性样本的偏移会被构造成 0，无法估计误报率
                others = [j for j in nidx if j != i]
                if not others:
                    continue
                base = np.median(X[others], axis=0)
            else:
                base = base_all
            out.append(('public:' + p, lab, X[i] - base, 1.0, 'public:' + p, False,
                        str(labels[i])))
    return out


def load_local(path, weight):
    """自采数据：采集时已经算好偏移量，直接用"""
    if not os.path.exists(path):
        print('找不到自采数据 %s，跳过。可先运行 emotion_collect.py' % path)
        return []
    out = []
    bad = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                lab = merge_label(str(r['label']))
                if lab is None or lab not in KEEP_CLASSES:
                    continue
                dev = r['deviations']
                vec = np.array([dev[k] for k in FX.FEATURE_KEYS], dtype=np.float64)
                # 同一段录制共享一个时间戳，用它做分组，避免相邻帧泄漏到测试集
                # (frames of one burst share a timestamp; grouping by it keeps neighbouring frames from
                #  leaking into the test set)
                burst = 'local:%s:%s' % (r['person'], r.get('time', 0))
            except (ValueError, KeyError):
                bad += 1
                continue
            out.append(('local:' + str(r['person']), lab, vec, weight, burst, False,
                        str(r['label'])))
    if bad:
        print('自采数据中有 %d 行无法解析或特征不匹配，已跳过（可能是改动特征前采的）' % bad)
    return out


# ---------------- 训练(training) ----------------

def softmax(S):
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)


def fit(Z, y, w, K, l2=1.0, iters=4000, lr=0.5):
    n, dim = Z.shape
    W = np.zeros((dim, K))
    Yoh = np.zeros((n, K))
    Yoh[np.arange(n), y] = 1.0
    wcol = w.reshape(-1, 1)
    tot = w.sum()
    for _ in range(iters):
        G = Z.T @ ((softmax(Z @ W) - Yoh) * wcol) / tot
        G[:-1] += l2 / n * W[:-1]
        W -= lr * G
    return W


def standardize(Xtr):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def prep(A, mu, sd):
    return np.hstack([(A - mu) / sd, np.ones((len(A), 1))])


def class_weights(y, K, sample_w):
    """按类别频率反比加权，并叠加样本自身权重"""
    counts = np.array([sample_w[y == k].sum() for k in range(K)], dtype=np.float64)
    cw = counts.sum() / (K * np.maximum(counts, 1e-9))
    return cw[y] * sample_w


def fmt(value):
    """某个类别在测试集里没有样本时召回为 None(recall is None when a class has no test samples)"""
    return 'n/a ' if value is None else '%.2f' % value


def metrics(pred, y, classes, ni):
    conf = collections.defaultdict(collections.Counter)
    for t, p in zip(y, pred):
        conf[classes[t]][classes[p]] += 1
    rec, prec, f1 = {}, {}, {}
    for c in classes:
        k = classes.index(c)
        tot = int((y == k).sum())
        tp = int(((pred == k) & (y == k)).sum())
        pp = int((pred == k).sum())
        rec[c] = tp / tot if tot else None
        prec[c] = tp / pp if pp else None
        if rec[c] is not None and prec[c] is not None and (rec[c] + prec[c]) > 0:
            f1[c] = 2 * rec[c] * prec[c] / (rec[c] + prec[c])
        else:
            f1[c] = 0.0
    return rec, prec, f1, conf


def main():
    args = build_parser().parse_args()
    if not args.public and not args.local:
        args.public = args.local = True
        print('未指定数据来源，默认两者都用(no source specified, using both)')

    loc = load_local(args.local_jsonl, args.local_weight) if args.local else []
    pub = load_public(args.public_npz) if args.public else []

    # 按"原始标签"弃用公开数据里不可靠的类别。
    # 用原始标签而不是合并后的类别，因为公开数据的 sadness 很微弱但 anger 尚可用，
    # 合并成 unhappy 后就分不开了。
    # (unreliable public classes are dropped by raw label rather than by merged class, because the public
    #  sadness images are very faint while its anger is still usable, and after merging into unhappy the
    #  two can no longer be told apart)
    drop_raw = [s.strip() for s in args.public_drop.split(',') if s.strip()]
    if pub and drop_raw:
        before = len(pub)
        pub = [d for d in pub if d[6] not in drop_raw]
        print('弃用公开数据的原始类别 %s（%d -> %d 样本）' % (drop_raw, before, len(pub)))

    data = []
    if pub:
        print('公开数据集样本 %d' % len(pub))
        data += pub
    if loc:
        print('自采样本 %d（权重 %.1f）' % (len(loc), args.local_weight))
        data += loc

    if not data:
        print('没有可用数据，先运行 extract_features.py 或 emotion_collect.py')
        return 1

    # 眉毛遮挡增强放在按人归一化之后，见下方
    # (the brow-occlusion augmentation runs after the per-person normalisation, see below)

    # 剔除样本过少的类别
    counts = collections.Counter(d[1] for d in data)
    drop = {c for c, n in counts.items() if n < MIN_SAMPLES_PER_CLASS}
    if drop:
        print('以下类别样本不足 %d 个，本次剔除: %s'
              % (MIN_SAMPLES_PER_CLASS, {c: counts[c] for c in drop}))
        data = [d for d in data if d[1] not in drop]

    # 只保留选定特征集里的列。模型文件会记录 feature_keys，
    # 推理时 GeometryEmotionModel 按这个列表从完整偏移量里取值，所以运行时代码不用改。
    # (keep only the columns of the chosen feature set. The model file records feature_keys and at
    #  inference GeometryEmotionModel picks those entries out of the full deviation dict, so no runtime
    #  code changes are needed)
    feat_keys = list(FEATURE_SETS[args.features])
    col = [list(FX.FEATURE_KEYS).index(k) for k in feat_keys]
    print('特征集 %s：使用 %d / %d 个特征%s'
          % (args.features, len(feat_keys), len(FX.FEATURE_KEYS),
             '（不含眉毛，遮挡时结构上无法误判）' if args.features == 'no-brow' else ''))

    persons = np.array([d[0] for d in data])
    Y = np.array([d[1] for d in data])
    D = np.array([d[2] for d in data])[:, col]
    Wt = np.array([d[3] for d in data], dtype=np.float64)
    bursts = np.array([d[4] for d in data])
    is_aug = np.array([d[5] for d in data])
    del data

    # ---------- 按人做尺度归一化 ----------
    #
    # 实测：不同人做同一表情的距离是 0.13~0.28，而同一人不同表情的距离只有 0.13~0.22。
    # 也就是个体差异盖过了类别差异，单一全局模型学不到通用规律。
    # 按人把偏移量除以他自己的表情幅度，各人的标签才可比。
    # 留一人交叉验证：不做归一化平均 0.40，做了 0.47。
    # (measured: the same expression differs by 0.13 to 0.28 between people while different expressions
    #  within one person differ by only 0.13 to 0.22, so individual variation dominates class variation
    #  and a single global model cannot learn anything general. Dividing each person's deviations by their
    #  own expression magnitude makes labels comparable across people. Leave-one-person-out went from 0.40
    #  to 0.47)
    #
    # 尺度向群体先验收缩，因为实时启动时只看到中性帧，直接用样本标准差会严重低估，
    # 导致偏移被放大、疯狂误报。训练必须用和推理一样的公式。
    # (the scale is shrunk toward the population prior because at runtime only neutral frames are seen at
    #  first, so a raw sample standard deviation would be badly underestimated, amplifying deviations and
    #  causing a flood of false positives. Training must use the same formula as inference)
    pop_scale = D.std(axis=0)
    pop_scale[pop_scale < 1e-9] = 1.0
    idx_by_p = collections.defaultdict(list)
    for i, p in enumerate(persons):
        idx_by_p[p].append(i)
    for p, idxs in idx_by_p.items():
        n = len(idxs)
        base = D[idxs].std(axis=0) if n >= 2 else pop_scale
        sd = (n * base + SCALE_PRIOR_M * pop_scale) / (n + SCALE_PRIOR_M)
        sd[sd < 1e-9] = 1.0
        D[idxs] = D[idxs] / sd
    print('已按人做尺度归一化（收缩强度 m=%.0f）' % SCALE_PRIOR_M)

    classes = sorted(set(Y))
    if 'neutral' not in classes:
        print('数据里没有 neutral 样本，无法训练。请务必采集中性表情。')
        return 1
    CI = {c: i for i, c in enumerate(classes)}
    ni = CI['neutral']
    K = len(classes)
    y = np.array([CI[c] for c in Y])

    print()
    # ---------- 眉毛遮挡增强（在归一化空间进行）----------
    # 增强样本沿用原样本的人和录制段，因此不会跨越训练/测试划分，
    # 而且被标记为 is_aug，评估时一律排除。
    # (the augmented samples inherit the person and burst of their source so they never cross the
    #  train/test split, and being flagged as is_aug they are always excluded from evaluation)
    if args.augment > 0:
        aug = augment_brow_occlusion_z(D, Y, persons, Wt, bursts,
                                       feat_keys, ratio=args.augment)
        if aug is not None:
            Da, Ya, Pa, Wa, Ba = aug
            print('眉毛遮挡增强样本 %d（应对头发或手遮住眉毛的情况）' % len(Da))
            D = np.vstack([D, Da])
            Y = np.concatenate([Y, Ya])
            persons = np.concatenate([persons, Pa])
            Wt = np.concatenate([Wt, Wa])
            bursts = np.concatenate([bursts, Ba])
            is_aug = np.concatenate([is_aug, np.ones(len(Da), dtype=bool)])
            y = np.array([CI[c] for c in Y])

    print('总样本 %d，人数 %d，类别 %s' % (len(D), len(set(persons)), classes))
    print('标签分布:', dict(collections.Counter(Y)))

    # ---------- 评估划分：对自采的人做留一人交叉验证 ----------
    #
    # 这是最有意义的指标：每次留一个采集者完全不参与训练，用他的干净样本做测试。
    # 它同时满足两点——测试的是刻意做出的清晰表情（和实际使用一致），
    # 而且测试者从未被模型见过（真正的跨人泛化）。
    # (this is the most meaningful metric: one collected subject is held out from training entirely and
    #  their clean samples form the test set. It satisfies both requirements at once, testing deliberate
    #  clear expressions that match real use, on a person the model has never seen)
    #
    # 增强样本只能进训练集，测试一律用干净帧，否则指标会被遮挡样本稀释。
    # (augmented samples may only enter the training set; testing always uses clean frames, otherwise the
    #  metric gets diluted by the occluded copies)
    local_people = sorted(set(p for p in persons if p.startswith('local:')))
    folds = []
    fold_names = []

    if len(local_people) >= 2:
        print()
        print('对 %d 位采集者做留一人交叉验证(每次留一人完全不参与训练)' % len(local_people))
        for p in local_people:
            te = (persons == p) & (~is_aug)      # 测试：该人的干净帧
            tr = persons != p                    # 训练：其他人 + 公开数据（含增强）
            folds.append((tr, te))
            fold_names.append(p.replace('local:', ''))
        eval_scheme = 'leave-one-collected-person-out'
    else:
        # 自采不足 2 人时退回按公开数据的人划分，并说明指标含义受限
        # (with fewer than two collected subjects, fall back to a split over the public people)
        pub_people = sorted(set(p for p in persons if p.startswith('public:')))
        if len(pub_people) < 2:
            print('数据不足，无法做跨人评估')
            return 1
        rng = np.random.default_rng(0)
        shuffled = list(pub_people)
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * 0.7)
        train_p = set(shuffled[:cut])
        tr = np.array([(not p.startswith('public:')) or p in train_p for p in persons])
        folds = [(tr, (~tr) & (~is_aug))]
        fold_names = ['public']
        eval_scheme = 'grouped split over public people'
        print()
        print('自采数据不足 2 人，退回公开数据划分：训练 %d 人 / 测试 %d 人'
              % (cut, len(shuffled) - cut))
        print('注意：公开数据是名人抓拍照，表情微弱，指标会明显偏低')

    # 每个 l2 只训练一次，把测试折的 logits 缓存下来。
    # 之后搜索偏置只是在缓存上加常数，不再重复训练——否则偏置搜索会慢上百倍。
    # (each l2 is trained only once and the test-fold logits are cached. Searching the biases then only
    #  adds constants to the cache instead of retraining, which is orders of magnitude faster)
    _cache = {}

    def logits_for(l2):
        if l2 not in _cache:
            parts, trues = [], []
            for tr, te in folds:
                mu, sd = standardize(D[tr])
                Ztr, Zte = prep(D[tr], mu, sd), prep(D[te], mu, sd)
                w = class_weights(y[tr], K, Wt[tr])
                W = fit(Ztr, y[tr], w, K, l2=l2)
                parts.append(Zte @ W)
                trues.append(y[te])
            _cache[l2] = (np.vstack(parts), np.concatenate(trues))
        return _cache[l2]

    def run_folds(l2, neutral_bias, class_bias=None):
        S, true = logits_for(l2)
        S = S.copy()
        if class_bias is not None:
            S = S + class_bias
        S[:, ni] += neutral_bias
        return S.argmax(axis=1), true

    print()
    print('--- 正则强度扫描(跨人评估) ---')
    best = None
    for l2 in (0.1, 1.0, 5.0, 20.0):
        pred, true = run_folds(l2, 0.0)
        rec, prec, f1, _ = metrics(pred, true, classes, ni)
        expr = [f1[c] for c in classes if c != 'neutral']
        score = sum(expr) / len(expr)
        print('l2=%-6.1f 中性正确率 %.2f  表情F1均值 %.2f  %s'
              % (l2, rec['neutral'], score,
                 ' '.join('%s=%s' % (c[:4], fmt(rec[c])) for c in classes if c != 'neutral')))
        if best is None or score > best[0]:
            best = (score, l2)
    l2 = best[1]
    print('选定 l2=%.1f' % l2)

    # ---------- 搜索各类别偏置 ----------
    #
    # 有 2 位以上采集者后，留一人交叉验证同时满足"跨人"和"刻意表情"两点，
    # 因此偏置直接在它上面搜索，不再需要以前那套"同人评估 + 跨人下限约束"的双轨。
    # 早期只有 1 个人时不得不用同人评估，那会把偏置推向那一个人的表情风格。
    # (once there are two or more collected subjects, leave-one-person-out satisfies both the
    #  cross-person and the deliberate-expression requirements, so the bias is searched directly on it and
    #  the earlier two-track scheme of a within-person score plus a cross-person floor is no longer
    #  needed. With only one subject a within-person score was unavoidable, and it pulled the bias toward
    #  that single person's style)
    # ---------- 遮挡稳健性探针 ----------
    # 偏置搜索只看 F1 时，会把 unhappy 偏置推到 +0.75，结果眉毛被遮挡的笑脸有 19%
    # 被判成不开心——正是之前专门修过的问题。所以把这件事直接写进搜索约束：
    # 用"眉毛被破坏的 happy 样本"当探针，要求误判率不超过上限。
    # (searching the bias on F1 alone pushed the unhappy bias to +0.75, which left 19% of smiles with
    #  occluded brows labelled unhappy, exactly the failure that had been fixed before. So the requirement
    #  is written into the search: happy samples with corrupted brows act as a probe and their error rate
    #  must stay under a ceiling)
    # 上限的取舍：
    #   0.05 -> 遮挡误判 6%，但 unhappy 跨人召回只有 0.25，真实的不开心表情基本触发不了
    #   0.12 -> 遮挡误判约 10%，unhappy 召回回到 0.35 以上
    # 选择偏向召回，因为机器人是按"8 帧里 6 帧一致"触发动作的，
    # 每帧 10% 的瞬时误判几乎不可能形成稳定误触发，而召回不足则会让整个功能失效。
    # (the trade-off: a ceiling of 0.05 gives a 6% occlusion error but leaves cross-person unhappy recall at
    #  only 0.25, so a genuine unhappy expression barely ever triggers, while 0.12 gives about 10% error and
    #  brings recall back above 0.35. Recall is favoured because the robot acts on a vote of six frames out
    #  of eight, so a 10% per-frame error can hardly ever form a stable false trigger while insufficient
    #  recall disables the feature outright)
    OCCLUSION_CEILING = 0.12
    NEGATIVE = tuple(c for c in classes if c in ('unhappy', 'sad', 'angry'))

    probe = None
    if NEGATIVE and any(k in feat_keys for k in BROW_FEATURES):
        rng3 = np.random.default_rng(2)
        hp = (Y == 'happy') & (~is_aug)
        if hp.sum() >= 30:
            Dp = D[hp].copy()
            n = len(Dp)
            # 探针专门测最坏情况，幅度取得比训练增强更狠。
            # 用和增强一样的范围时约束绑不到极端遮挡，实测极端下仍有 14% 误判。
            # (the probe deliberately targets the worst case with a harsher magnitude than the training
            #  augmentation. Using the same range left the constraint slack at the extreme end, where the
            #  measured error was still 14%)
            drop = rng3.uniform(2.0, 9.0, size=n)
            for k in ('brow_raise', 'brow_inner_h', 'brow_outer_h'):
                Dp[:, feat_keys.index(k)] -= drop
            Dp[:, feat_keys.index('brow_slope')] *= 0.2
            mu_p, sd_p = standardize(D)
            Wp = fit(prep(D, mu_p, sd_p), y, class_weights(y, K, Wt), K, l2=l2)
            probe = prep(Dp, mu_p, sd_p) @ Wp

    def occlusion_error(class_bias, neutral_bias):
        if probe is None:
            return 0.0
        S = probe + class_bias
        S[:, ni] += neutral_bias
        pred = S.argmax(axis=1)
        bad = sum(int((pred == CI[c]).sum()) for c in NEGATIVE)
        return bad / float(len(pred))

    # 中性正确率下限。放松的脸被误判成情绪，比漏掉一个表情更让人烦，
    # 所以把它做成硬约束而不是混进目标函数——混进去的话它样本多又容易，会掩盖真正关心的项。
    # (a floor on neutral accuracy. A relaxed face wrongly labelled as an emotion is more annoying than a
    #  missed expression, so it is a hard constraint rather than a term in the objective, where its large
    #  and easy sample set would mask what actually matters)
    NEUTRAL_FLOOR = 0.75

    def score_bias(class_bias, neutral_bias):
        if occlusion_error(class_bias, neutral_bias) > OCCLUSION_CEILING:
            return -1.0        # 遮挡时误判太多，否决(too many occlusion errors, rejected)
        pred, true = run_folds(l2, neutral_bias, class_bias)
        rec, _, f1t, _ = metrics(pred, true, classes, ni)
        if rec['neutral'] is not None and rec['neutral'] < NEUTRAL_FLOOR:
            return -1.0        # 中性误报太多，否决(too many false positives on neutral, rejected)
        # 只看表情类别的 F1，中性已经由上面的下限保证
        # (only the expression classes count; neutral is guaranteed by the floor above)
        expr = [f1t[c] for c in classes if c != 'neutral']
        return sum(expr) / len(expr)

    print()
    print('--- 按类别搜索决策偏置(在留一人交叉验证上搜索，目标 F1 均值) ---')
    class_bias = np.zeros(K)
    best_f1 = score_bias(class_bias, args.neutral_bias)
    for _pass in range(2):
        for k in range(K):
            if k == ni:
                continue
            # 范围收窄到 ±0.75，避免偏置过度补偿某一类
            # (the range is capped at ±0.75 so the bias cannot over-compensate for one class)
            for cand in np.arange(-0.75, 0.76, 0.25):
                trial = class_bias.copy()
                trial[k] = cand
                s = score_bias(trial, args.neutral_bias)
                if s > best_f1 + 1e-6:
                    best_f1, class_bias = s, trial
    print('偏置:', {classes[k]: round(float(class_bias[k]), 2) for k in range(K)})
    print('F1 均值 %.3f' % best_f1)

    # ---------- 权衡曲线 ----------
    print()
    print('=== 中性偏置权衡曲线(跨人评估) ===')
    print('  偏置越大越保守：中性更多、误报更少，但表情更难触发')
    print('%-9s %-10s %-10s %s' % ('中性偏置', '中性正确率', '表情F1均值', '各表情召回'))
    curve = []
    for nb in (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
        pred, true = run_folds(l2, nb, class_bias)
        rec, prec, f1, _ = metrics(pred, true, classes, ni)
        expr_f1 = sum(f1[c] for c in classes if c != 'neutral') / (K - 1)
        curve.append({'neutral_bias': nb,
                      'neutral_accuracy': round(rec['neutral'], 3),
                      'expression_f1_mean': round(expr_f1, 3),
                      'per_class_recall': {c: (round(rec[c], 3) if rec[c] is not None else None)
                                           for c in classes if c != 'neutral'}})
        print('%-9.2f %-10s %-10.2f %s' % (nb, fmt(rec['neutral']), expr_f1,
              ' '.join('%s=%s' % (c[:4], fmt(rec[c])) for c in classes if c != 'neutral')))

    # ---------- 选定操作点的详细结果 ----------
    pred, true = run_folds(l2, args.neutral_bias, class_bias)
    rec, prec, f1, conf = metrics(pred, true, classes, ni)
    print()
    print('=== 默认操作点(中性偏置 %.2f)的详细结果 ===' % args.neutral_bias)
    for c in classes:
        tot = sum(conf[c].values())
        if not tot:
            continue
        top = ', '.join('%s×%d' % (k, v) for k, v in conf[c].most_common(3))
        print('  %-10s %5d 样本  召回 %s  准确率 %s  F1 %.2f   最常判成: %s'
              % (c, tot, fmt(rec[c]), fmt(prec[c]), f1[c], top))

    # ---------- 按人分别汇报 ----------
    # 每个人的数字是"模型没见过这个人"时的表现。
    # 人与人之间差异大，说明表情风格差异明显，需要更多人的数据来抹平。
    # (each person's numbers are the performance when the model has never seen them. A large spread
    #  between people means their expression styles differ a lot and more subjects are needed to even out)
    per_person = {}
    if len(local_people) >= 2:
        print()
        print('=== 按人分别汇报（每个人都是"模型没见过他"时的结果）===')
        print('%-14s %s  %s' % ('人', '  '.join('%-9s' % c[:9] for c in classes), '整体'))
        for (tr, te), name in zip(folds, fold_names):
            S = logits_for(l2)[0]
            # 重新按折取出该人的预测
            offset = 0
            for (tr2, te2), nm in zip(folds, fold_names):
                if nm == name:
                    break
                offset += int(te2.sum())
            n = int(te.sum())
            Sp = S[offset:offset + n] + class_bias
            Sp[:, ni] += args.neutral_bias
            pp, tt = Sp.argmax(axis=1), y[te]
            rp, _, _, _ = metrics(pp, tt, classes, ni)
            acc = float((pp == tt).sum()) / max(1, n)
            per_person[name] = {'accuracy': round(acc, 3), 'frames': n,
                                'recall': {c: (round(rp[c], 3) if rp[c] is not None else None)
                                           for c in classes}}
            print('%-14s %s  %.2f  (%d 帧)'
                  % (name, '  '.join('%-9s' % fmt(rp[c]) for c in classes), acc, n))

        spread = max(v['accuracy'] for v in per_person.values()) - \
            min(v['accuracy'] for v in per_person.values())
        print()
        print('人与人之间的整体正确率差距: %.2f' % spread)
        if spread > 0.20:
            print('  差距偏大，说明各人表情风格差异明显。补更多人的数据是最有效的改进。')
        else:
            print('  差距不大，说明模型对不同人的表情风格已经比较稳定。')

    if args.dry_run:
        print()
        print('--dry-run 已指定，不写出模型')
        return 0

    # ---------- 用全部数据重训并导出 ----------
    mu, sd = standardize(D)
    Z = prep(D, mu, sd)
    w = class_weights(y, K, Wt)
    W = fit(Z, y, w, K, l2=l2)

    # ---------- 校准最终模型的中性偏置 ----------
    #
    # 上面的偏置是在留一模型上调的，每个留一模型只见过 2 个人；
    # 最终模型见过全部 3 个人，因此更自信，同一个偏置下会明显更偏向表情类别。
    # 实测：留一时 neutral 0.84，换成最终模型后同样偏置下只有 0.53。
    # (the biases above were tuned on the leave-one-out models, each of which saw only two subjects, while
    #  the final model sees all three and is therefore more confident, leaning further toward the
    #  expression classes at the same bias. Measured: neutral was 0.84 under leave-one-out but only 0.53
    #  with the final model at the same bias)
    #
    # 所以按"让最终模型的中性正确率贴近留一时的水平"来校准中性偏置。
    # (so the neutral bias is calibrated to bring the final model's neutral accuracy close to the
    #  leave-one-out level)
    target_neutral = rec['neutral'] if rec['neutral'] is not None else 0.85
    clean_local = (~is_aug) & np.array([p.startswith('local:') for p in persons])
    final_nb = args.neutral_bias
    if clean_local.sum() >= 50:
        S_all = Z[clean_local] @ W + class_bias
        y_loc = y[clean_local]
        n_mask = y_loc == ni
        best = None
        for nb in np.arange(0.0, 3.01, 0.05):
            S = S_all.copy()
            S[:, ni] += nb
            pred = S.argmax(axis=1)
            acc = float((pred[n_mask] == ni).sum()) / max(1, int(n_mask.sum()))
            gap = abs(acc - target_neutral)
            if best is None or gap < best[0]:
                best = (gap, nb, acc)
        _, final_nb, got = best
        print()
        print('校准最终模型的中性偏置: %.2f -> %.2f（中性正确率 %.2f，对齐留一时的 %.2f）'
              % (args.neutral_bias, final_nb, got, target_neutral))

    model = {
        'sources': ([] + (['public: muxspace/facial_expressions (Apache-2.0)'] if args.public else [])
                    + (['local: self-collected via emotion_collect.py'] if args.local else [])),
        'note': 'per-person neutral baseline is subtracted before applying this model',
        'feature_keys': feat_keys,
        'feature_set': args.features,
        'classes': classes,
        'mean': [float(v) for v in mu],
        'std': [float(v) for v in sd],
        'weights': [[float(v) for v in row] for row in W[:-1]],
        'bias': [float(v) for v in W[-1]],
        # neutral 一项必须清零，中性偏置由 neutral_bias 单独控制，否则会重复叠加
        'class_bias': [(0.0 if k == ni else float(class_bias[k])) for k in range(K)],
        'population_scale': [float(v) for v in pop_scale],
        'scale_prior_m': SCALE_PRIOR_M,
        'default_neutral_bias': float(final_nb),
        'tuning_neutral_bias': args.neutral_bias,
        'l2': l2,
        'evaluation': eval_scheme,
        'collected_people': len(local_people),
        'per_person': per_person,
        'people': len(set(persons)),
        'samples': int(len(D)),
        'tradeoff_curve': curve,
        'eval_neutral_accuracy': round(float(rec['neutral']), 4),
        'eval_recall': {c: (round(float(rec[c]), 4) if rec[c] is not None else None)
                        for c in classes if c != 'neutral'},
    }
    out = os.path.abspath(args.out)
    json.dump(model, open(out, 'w'), indent=2)
    print()
    print('模型已写出 %s' % out)
    print('直接运行 emotion_mac_test.py 即可生效')
    return 0


if __name__ == '__main__':
    sys.exit(main())
