# 情绪识别：5 类模型的训练流程

识别 **中性 / 高兴 / 悲伤 / 生气 / 惊讶** 五种情绪。

模型文件：`../TonyPi/Functions/model/geometry_emotion_model.json`

## 为什么是 5 类

原来做了 7 类，`disgust` 和 `fearful` 被删掉，依据是实测数据：

- 公开数据集里 `disgust` 只有 167 个样本、`fearful` 只有 10 个
- 跨人评估中 `disgust` 召回**始终为 0.00**
- 二者在几何上与 `angry` / `surprised` 高度重叠，细分只会互相污染

想恢复只需改 `train_model.py` 里的 `KEEP_CLASSES` 一行，采集到的
disgust / fearful 样本仍保留在数据文件里，没有删除。

## 识别原理

```
摄像头 -> FaceMesh(468 点) -> 21 个几何特征 -> 减去个人中性基线
       -> 多人数据训练的线性模型 -> 5 类情绪
```

**个人基线必须来自本人**，它测的是「这张脸放松时长什么样」，用别人的数据替代只会更差。
模型学的是「多大的偏移、什么组合算哪种情绪」，这部分才需要多人数据。
这两件事经常被混为一谈。

## 完整流程

### 1. 采集（推荐，效果最好）

```bash
.venv-emotion/bin/python emotion_collect.py --person zhangsan
.venv-emotion/bin/python emotion_collect.py --person lisi
.venv-emotion/bin/python emotion_collect.py --stats     # 查看统计
```

窗口内按键：`n` 中性 · `h` 高兴 · `s` 悲伤 · `a` 生气 · `u` 惊讶
`c` 重新标定 · `DELETE` 删除上一段 · `q` 退出

按一下 → 1 秒准备 → 录制 1 秒（约 30 帧）→ 自动保存。

**采集要点：**

1. **标定时必须是真正放松的脸，不要笑。** 多人测试出现过「人人都是 sad」，
   根因就是标定那一秒对着镜头不自觉微笑，基线被记成了微笑。按 `c` 可重来。
2. 采集期间基线漂移是关闭的，否则基线会把正在做的表情吸收进去。
3. 每种情绪 60 帧以上，表情做**明确**一些。
4. **至少 3~5 个人**。只有 1 个人时无法验证跨人泛化。

### 2. 训练

```bash
cd emotion_train
PYTHONPATH=../TonyPi/Functions ../.venv-emotion/bin/python train_model.py --public --local --local-weight 3
```

可选：`--local` 只用自采 · `--public` 只用公开数据 · `--dry-run` 只看指标不覆盖模型

### 3. 验证

```bash
.venv-emotion/bin/python emotion_mac_test.py
```

## 两种评估指标，含义不同

训练脚本会同时给出两个数字，**不要混着看**：

| 指标 | 测试集 | 回答的问题 |
|---|---|---|
| 跨人评估 | 公开数据集的名人抓拍照 | 陌生人来用准不准 |
| 同人内部评估 | 自采数据，按录制段划分 | 采集过的人自己用准不准 |

同人评估按**录制段**划分，保证同一段的相邻帧不会同时出现在训练和测试集，
否则相邻帧几乎相同，指标会虚高。

### 当前实测结果（1 个人的自采数据 + 公开数据集）

```
同人内部评估（52 段中留 15 段做测试，372 帧）
  angry      召回 0.80   准确率 1.00
  happy      召回 0.95   准确率 1.00
  neutral    召回 0.92   准确率 0.49
  sad        召回 1.00   准确率 0.92
  surprised  召回 1.00   准确率 1.00
  整体正确率 0.92

跨人评估（260 个未参与训练的人，全是名人抓拍照）
  中性正确率 0.67   happy 0.67   surprised 0.51   angry 0.37
```

跨人指标明显更低，主要因为公开数据集是名人抓拍照，表情微弱、标签噪声大，
和刻意做出的清晰表情分布差别很大。**要提升跨人表现，唯一有效的办法是让更多人采集。**

## 一个重要发现：公开数据集的 sad 会毒害模型

实测对比：

```
                sad    angry   happy   neutral  surprised   整体
只用自采        1.00    0.62    0.84     0.88      0.77     0.78
公开+自采       0.08    0.52    0.94     0.96      1.00     0.71
```

公开数据的 `sadness` 多为名人抓拍照，表情极微弱、几何上接近中性，
跨人 F1 只有 0.09。混进训练等于在教模型「sad ≈ neutral」，
结果自采数据上的 sad 召回从 1.00 掉到 0.08。

所以 `train_model.py` 里有 `PUBLIC_UNRELIABLE = ('sad',)`：
当自采数据已提供足够的 sad 样本时，就不再使用公开数据的这一类。
`neutral` / `happy` / `surprised` 则相反，公开数据提供了宝贵的跨人多样性，保留。

## 另一个坑：偏置要在部署相关的评估上调

早期版本在跨人测试集上搜索类别偏置，那里 sad 学不出来，
于是搜索把 sad 偏置压到 **-1.0**，这个惩罚又反过来杀死了自采数据上的 sad。

现在改为：有自采数据时，在**同人评估**上搜索偏置，因为那才是实际使用场景。
搜索范围也故意收窄到 ±0.75 —— 放宽到 ±1.5 时，只有 1 个人的数据会把偏置推到极端，
同人正确率只从 0.92 涨到 0.96，但陌生人的中性正确率会从 0.67 掉到 0.52。

## 第三个坑：眉毛被遮挡会误判成生气

用户反馈「眉毛被遮挡时，明明在笑却显示生气」。用真实的 happy 样本模拟后确认了：

```text
眉毛下移幅度   笑脸被判成生气的比例
0.02              4%
0.06             43%
0.09             62%
0.14             94%
```

**原因**：眉毛被头发或手遮住时，FaceMesh 会把发际边缘当成眉毛，检测位置偏低。
而 `brow_raise` 对 `angry` 的权重是 **-1.13**（第二大），
所以「眉毛看起来变低」会强烈推向生气，即使嘴在笑。

**解决办法**：训练时生成眉毛被遮挡的增强样本（`augment_brow_occlusion`），
标签保持不变，让模型学会眉毛不可靠时更多依赖嘴和眼睛。

```text
增强设置            angry  sad    遮挡误判(下移 0.06/0.09/0.14)
不增强              0.66   0.75      11%   40%   94%
ratio=1.0          0.61   0.69       0%    0%    2%
```

代价是 angry 召回掉约 5 点、sad 掉约 7 点，换来重度遮挡下 94% → 2%。这个交换很值。

增强的下移范围必须覆盖到 0.18：只训练到 0.12 时，下移 0.14 的极端遮挡仍有 22% 误判，
因为超出了训练覆盖范围。

修复后在各种遮挡形态下重新验证（真实笑脸被判成生气的比例）：

```text
眉毛下移 0.06 / 0.09 / 0.14   ->  0%  0%  2%
眉毛上移 0.06 / 0.09          ->  0%  0%
眉毛形状被抹平                ->  0%
眉毛信息完全丢失              ->  0%
```

**试过但没采用的方案**：运行时「有笑意就压制生气」。
真实的生气样本本身就带有类似微笑的线索，压制会让 angry 召回从 0.89 掉到 0.69、
sad 从 0.95 掉到 0.72，而增强已经把问题解决了，所以删掉了这条规则。

## 第四个坑：偏置搜索会系统性放大生气误报

偏置搜索只看同人评估时，会一路把 `angry` 推到 +0.75
（同人数据里生气样本多且刻意），代价是陌生人的中性正确率从 0.67 掉到 0.59。

现在加了约束 `CROSS_NEUTRAL_FLOOR = 0.62`：任何候选偏置都不能让跨人中性正确率跌破这个下限。

## 灵敏度调节

```bash
.venv-emotion/bin/python emotion_mac_test.py --neutral-bias 0.0   # 更灵敏
.venv-emotion/bin/python emotion_mac_test.py --neutral-bias 1.0   # 更保守
.venv-emotion/bin/python emotion_mac_test.py --boost sad=0.5      # 只提高悲伤
```

## 公开数据集的复现步骤

数据集：[muxspace/facial_expressions](https://github.com/muxspace/facial_expressions)，Apache-2.0。
文件名是 LFW 风格（`Aaron_Eckhart_0001.jpg`），编码了人物身份，所以能按人分组算基线。

```bash
# 下载（约 245MB）。注意 raw.githubusercontent 常被拦，用 codeload
curl -sSL -o /tmp/fe.tar.gz \
  "https://codeload.github.com/muxspace/facial_expressions/tar.gz/refs/heads/master"
mkdir -p /tmp/fe && tar xzf /tmp/fe.tar.gz -C /tmp/fe

# 标签文件走 API（legend.csv 不是 LFS 文件）
curl -sS -H "Accept: application/vnd.github.raw" \
  "https://api.github.com/repos/muxspace/facial_expressions/contents/data/legend.csv" \
  -o /tmp/legend.csv

# 提取特征（约 45 秒）
cd emotion_train
PYTHONPATH=../TonyPi/Functions ../.venv-emotion/bin/python extract_features.py
```
