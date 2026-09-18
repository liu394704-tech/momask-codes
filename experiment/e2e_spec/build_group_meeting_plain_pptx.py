#!/usr/bin/env python3
"""组会 PPT：大白话、四段结构、简洁版。"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path("/Users/emmaliu/Desktop/HKUST/RBM-project/Model/momask-codes")
OUT = ROOT / "experiment/e2e_spec/组会汇报_大白话_202608.pptx"
FIG_LAT = ROOT / "experiment/pi_bench_joints/fig_latency.png"
FIG_MOT = ROOT / "experiment/pi_bench_joints/fig_motion_quality.png"

INK = RGBColor(0x1B, 0x1B, 0x1B)
MUTED = RGBColor(0x5C, 0x5C, 0x5C)
NAVY = RGBColor(0x1B, 0x3A, 0x4B)
TEAL = RGBColor(0x2A, 0x6F, 0x6F)
CORAL = RGBColor(0xC4, 0x5C, 0x41)
CREAM = RGBColor(0xF7, 0xF4, 0xEE)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LINE = RGBColor(0xE6, 0xE0, 0xD6)
FONT = "PingFang SC"
W, H = Inches(13.333), Inches(7.5)


def font(run, text, size, bold=False, color=INK):
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = FONT
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = rPr.makeelement(qn("a:ea"), {})
        rPr.append(ea)
    ea.set("typeface", FONT)


def rect(slide, l, t, w, h, fill):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    return sh


def txt(slide, l, t, w, h, text, size=16, bold=False, color=INK, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    font(r, text, size, bold, color)
    return box


def blank(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, W, H, CREAM)
    return s


def bar(slide):
    rect(slide, 0, 0, Inches(0.12), H, NAVY)


def foot(slide, page, total=16):
    txt(slide, Inches(0.55), Inches(7.12), Inches(10), Inches(0.28),
        "组会  ·  感知到动作  ·  2026.08", 11, False, MUTED)
    txt(slide, Inches(11.6), Inches(7.12), Inches(1.2), Inches(0.28),
        "%d / %d" % (page, total), 11, False, MUTED, PP_ALIGN.RIGHT)


def title_block(slide, kicker, title, sub=""):
    bar(slide)
    txt(slide, Inches(0.55), Inches(0.32), Inches(12), Inches(0.32), kicker, 13, False, TEAL)
    txt(slide, Inches(0.55), Inches(0.62), Inches(12.2), Inches(0.55), title, 28, True, NAVY)
    if sub:
        txt(slide, Inches(0.55), Inches(1.18), Inches(12.2), Inches(0.4), sub, 15, False, MUTED)
    rect(slide, Inches(0.55), Inches(1.62), Inches(1.15), Inches(0.06), CORAL)


def card(slide, l, t, w, h, fill=WHITE):
    sh = rect(slide, l, t, w, h, fill)
    sh.line.fill.solid()
    sh.line.color.rgb = LINE
    sh.line.width = Pt(1)
    return sh


def build():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    P = 16

    # 1 cover
    s = blank(prs)
    rect(s, 0, 0, Inches(0.18), H, NAVY)
    rect(s, 0, Inches(6.55), W, Inches(0.95), NAVY)
    txt(s, Inches(0.7), Inches(1.55), Inches(11), Inches(0.4), "组会汇报", 16, False, TEAL)
    txt(s, Inches(0.7), Inches(2.05), Inches(12), Inches(1.3),
        "机器人怎么看、怎么听、怎么动", 36, True, NAVY)
    txt(s, Inches(0.7), Inches(3.45), Inches(11.5), Inches(0.9),
        "端侧动作生成已经能跑。云侧联调看清了慢在哪。\n接下来用双轨办法先落地，再把整套搬回树莓派。",
        18, False, MUTED)
    txt(s, Inches(0.7), Inches(6.75), Inches(12), Inches(0.45),
        "HKUST RBM  ·  2026.08", 14, False, WHITE)

    # 2 今天四件事
    s = blank(prs)
    title_block(s, "提纲", "今天只讲四件事", "都用大白话。数字只留最有用的几个。")
    items = [
        ("01", "端侧 MoMask", "树莓派上，一句话生成一套动作，大概 2 秒。"),
        ("02", "云侧联调", "电脑上把看、听、想接起来。慢，主要慢在开机拍照和反复加载模型。"),
        ("03", "A + B 双轨", "先立刻做一个简单动作，细动作在后台慢慢生成。"),
        ("04", "下一步", "先把视听做快做准，再整套上树莓派，最后才做记忆。"),
    ]
    for i, (n, t, d) in enumerate(items):
        y = Inches(1.95 + i * 1.15)
        card(s, Inches(0.55), y, Inches(12.2), Inches(1.02))
        txt(s, Inches(0.8), y + Inches(0.28), Inches(0.8), Inches(0.45), n, 22, True, CORAL)
        txt(s, Inches(1.7), y + Inches(0.12), Inches(10.6), Inches(0.4), t, 20, True, NAVY)
        txt(s, Inches(1.7), y + Inches(0.52), Inches(10.6), Inches(0.4), d, 15, False, MUTED)
    foot(s, 2, P)

    # 3 part1 divider
    s = blank(prs)
    rect(s, 0, 0, W, H, NAVY)
    txt(s, Inches(0.7), Inches(2.4), Inches(12), Inches(0.4), "第一部分", 16, False, CORAL)
    txt(s, Inches(0.7), Inches(2.9), Inches(12), Inches(1.0), "端侧已经能出动作", 36, True, WHITE)
    txt(s, Inches(0.7), Inches(4.1), Inches(11.5), Inches(0.8),
        "给树莓派一句英文动作描述，它吐出全身关节坐标。\n不渲染视频，只关心：稳不稳、快不快、像不像那种情绪。",
        18, False, RGBColor(0xC9, 0xD4, 0xD8))

    # 4 kpi
    s = blank(prs)
    title_block(s, "01  端侧 MoMask", "一句话结果", "树莓派 5，4GB 内存，CPU，100 条情绪均衡指令。前 2 条当热身，不计入平均。")
    kpis = [
        ("2.3 秒", "平均一条", "最短 1.9，最长 2.7"),
        ("2.6 秒", "最慢的那 5%", "大部分都挤在 2.2～2.4"),
        ("100%", "数字全合法", "没有算崩、没有乱码关节"),
        ("10 秒", "模型只加载一次", "这笔账不算进「一条 2.3 秒」"),
    ]
    for i, (n, a, b) in enumerate(kpis):
        x = Inches(0.55 + i * 3.15)
        card(s, x, Inches(2.0), Inches(2.95), Inches(2.55))
        txt(s, x + Inches(0.15), Inches(2.2), Inches(2.65), Inches(0.85), n, 28, True, NAVY, PP_ALIGN.CENTER)
        txt(s, x + Inches(0.15), Inches(3.05), Inches(2.65), Inches(0.45), a, 16, True, TEAL, PP_ALIGN.CENTER)
        txt(s, x + Inches(0.15), Inches(3.5), Inches(2.65), Inches(0.7), b, 13, False, MUTED, PP_ALIGN.CENTER)
    txt(s, Inches(0.55), Inches(4.85), Inches(12.2), Inches(1.8),
        "人话：模型开一次机要十秒。开好之后，再说「挥手」「鞠躬」，大约两秒就能出一套动作。\n"
        "这个 2.3 秒，才是以后端侧应该报的「生成速度」。",
        17, False, INK)
    foot(s, 4, P)

    # 5 latency viz
    s = blank(prs)
    title_block(s, "01  端侧 MoMask", "时间稳不稳？看这张图", "左：98 条的耗时分布。右：不同情绪平均耗时，大家都在 2 秒出头，没有谁特别拖后腿。")
    if FIG_LAT.exists():
        s.shapes.add_picture(str(FIG_LAT), Inches(0.5), Inches(1.85), Inches(12.3), Inches(3.55))
    txt(s, Inches(0.55), Inches(5.55), Inches(12.2), Inches(1.3),
        "读图：柱子堆在 2.3 秒附近，没有拖到 4 秒、5 秒的尾巴。\n"
        "高兴、悲伤、惊讶，生成时间差不多——慢不慢主要看机器，不太看这句话是开心还是难过。",
        16, False, INK)
    foot(s, 5, P)

    # 6 motion viz
    s = blank(prs)
    title_block(s, "01  端侧 MoMask", "生成的动作有没有情绪差别？", "没有真人标注，只能看关节自己动得猛不猛。速度大＝动作更开；抖动大＝变化更急。")
    if FIG_MOT.exists():
        s.shapes.add_picture(str(FIG_MOT), Inches(0.5), Inches(1.85), Inches(12.3), Inches(3.55))
    txt(s, Inches(0.55), Inches(5.55), Inches(12.2), Inches(1.3),
        "读图：高兴动得最开、变化最急；悲伤最收着。这和常识一致——说明文本真的在调动作，不是每句都生成同一套站桩。\n"
        "抖动高不等于难看，高兴本来就该更「蹦」。",
        16, False, INK)
    foot(s, 6, P)

    # 7 part2
    s = blank(prs)
    rect(s, 0, 0, W, H, NAVY)
    txt(s, Inches(0.7), Inches(2.4), Inches(12), Inches(0.4), "第二部分", 16, False, CORAL)
    txt(s, Inches(0.7), Inches(2.9), Inches(12), Inches(1.0), "云侧联调：先在电脑上把链路跑通", 32, True, WHITE)
    txt(s, Inches(0.7), Inches(4.15), Inches(11.5), Inches(1.0),
        "看和听在电脑本地做。想一想用云端小模型。\n整套还没有完整搬到树莓派——4GB 装不下同时编译和推理。",
        18, False, RGBColor(0xC9, 0xD4, 0xD8))

    # 8 how cloud works
    s = blank(prs)
    title_block(s, "02  云侧联调", "看、听、想，各干各的", "没有把视频丢给大模型看图聊天。云端只负责「听成文字」和「写成动作句子」。")
    cols = [
        ("看", "电脑本地",
         "摄像头拍脸\n468 个关键点\n算嘴眼是否上扬\n判 4 种情绪"),
        ("听", "先本地，再上云",
         "麦克风录 3～4 秒\n送到云端转成文字\n不做语音情绪\n文字拿去当「用户想干什么」"),
        ("想", "云端小模型",
         "模型：gpt-4o-mini\n只看结构化结果\n写出一句英文动作\n再交给 MoMask"),
    ]
    for i, (t, sub, body) in enumerate(cols):
        x = Inches(0.5 + i * 4.2)
        card(s, x, Inches(1.95), Inches(3.95), Inches(4.55))
        rect(s, x, Inches(1.95), Inches(3.95), Inches(0.7), NAVY if i != 1 else TEAL)
        txt(s, x, Inches(2.05), Inches(3.95), Inches(0.5), t, 22, True, WHITE, PP_ALIGN.CENTER)
        txt(s, x + Inches(0.2), Inches(2.8), Inches(3.55), Inches(0.4), sub, 14, True, CORAL)
        txt(s, x + Inches(0.2), Inches(3.25), Inches(3.55), Inches(2.9), body, 16, False, INK)
    foot(s, 8, P)

    # 9 latency
    s = blank(prs)
    title_block(s, "02  云侧联调", "为啥一轮要十几秒？", "电脑实测：看脸约 10 秒，听约 3 秒，想约 2.5 秒。真正的云开销只有后两笔。")
    rows = [
        ("看脸 10 秒", "每轮重新打开摄像头，还要对着镜头「放轻松」标定大约 8 秒。\n真正用来判情绪的只有大约 12 帧。10 秒不是识别慢，是每次都在重新开机。"),
        ("听 + 想 5～7 秒", "录音可以和看脸同时进行，所以 3.5 秒录音被 10 秒盖住了。\n转文字、写动作句才是额外等的云时间。"),
        ("MoMask 8.9 秒是冤枉账", "树莓派上模型已经在内存里，一条只要 2.3 秒。\n电脑这次每轮重新启动 MoMask，把「开机 10 秒」也算进去了，所以变成 8.9 秒。模型没有变慢。"),
    ]
    for i, (t, d) in enumerate(rows):
        y = Inches(1.9 + i * 1.55)
        card(s, Inches(0.55), y, Inches(12.2), Inches(1.4))
        txt(s, Inches(0.8), y + Inches(0.15), Inches(11.7), Inches(0.4), t, 18, True, NAVY)
        txt(s, Inches(0.8), y + Inches(0.58), Inches(11.7), Inches(0.7), d, 15, False, MUTED)
    foot(s, 9, P)

    # 10 4GB limit
    s = blank(prs)
    title_block(s, "02  还没全部上端侧", "不是不想上树莓派，是 4GB 太挤", "测试机只有 4GB 运行内存。装东西和跑东西，抢的是同一块内存。")
    points = [
        ("已经上去的", "MoMask 能跑；A+B 的规则决策能空跑通。代码已经拷到板上。"),
        ("还在电脑上的", "摄像头情绪、云端听写、云端写动作句。这些还没在树上完整串起来。"),
        ("卡住的原因", "端侧要用的小模型 Qwen 没有现成安装包，只能现场编译。编译器一开多核，内存会被吃光，机器直接死机。"),
        ("人话", "4GB 够「跑一次生成」，不够「一边装大工具、一边跑生成」。16GB 的真机就不会这么脆。"),
    ]
    for i, (t, d) in enumerate(points):
        y = Inches(1.9 + i * 1.18)
        card(s, Inches(0.55), y, Inches(12.2), Inches(1.05))
        txt(s, Inches(0.8), y + Inches(0.12), Inches(11.7), Inches(0.35), t, 17, True, CORAL if i == 2 else NAVY)
        txt(s, Inches(0.8), y + Inches(0.5), Inches(11.7), Inches(0.45), d, 15, False, MUTED)
    foot(s, 10, P)

    # 11 part3
    s = blank(prs)
    rect(s, 0, 0, W, H, NAVY)
    txt(s, Inches(0.7), Inches(2.4), Inches(12), Inches(0.4), "第三部分", 16, False, CORAL)
    txt(s, Inches(0.7), Inches(2.9), Inches(12), Inches(1.0), "备选：A + B，先动起来", 36, True, WHITE)
    txt(s, Inches(0.7), Inches(4.15), Inches(11.5), Inches(0.9),
        "人等 15 秒会觉得机器人死机。\n所以正式落地不走「全部算完再动」，而走双轨。",
        18, False, RGBColor(0xC9, 0xD4, 0xD8))

    # 12 AB
    s = blank(prs)
    title_block(s, "03  双轨架构", "一边马上动，一边慢慢做漂亮动作", "A 是预置动作，保证互动。B 是 MoMask，保证动作细。默认两条一起跑。")
    card(s, Inches(0.55), Inches(1.95), Inches(6.0), Inches(4.55))
    rect(s, Inches(0.55), Inches(1.95), Inches(6.0), Inches(0.65), TEAL)
    txt(s, Inches(0.55), Inches(2.05), Inches(6.0), Inches(0.45), "A 轨 · 保底", 20, True, WHITE, PP_ALIGN.CENTER)
    txt(s, Inches(0.85), Inches(2.85), Inches(5.4), Inches(3.3),
        "开心就招手，难过就鞠躬。\n动作是现成的，几乎马上能播。\n"
        "不求好看，求「机器人还活着」。\n\n"
        "拿不准、没听清、后面算失败：\n这一轮只走 A，前面已经做的动作也不撤回。",
        16, False, INK)

    card(s, Inches(6.8), Inches(1.95), Inches(6.0), Inches(4.55))
    rect(s, Inches(6.8), Inches(1.95), Inches(6.0), Inches(0.65), NAVY)
    txt(s, Inches(6.8), Inches(2.05), Inches(6.0), Inches(0.45), "B 轨 · 正式", 20, True, WHITE, PP_ALIGN.CENTER)
    txt(s, Inches(7.1), Inches(2.85), Inches(5.4), Inches(3.3),
        "根据表情和说话内容，写一句细的英文。\nMoMask 再生成连续关节。\n"
        "热启动大约 2 秒，可以接受。\n\n"
        "B 可以比 A 晚到。A 先把场撑住，\nB 到了再换成更像样的动作。",
        16, False, INK)
    foot(s, 12, P)

    # 13 why AB reduces wait
    s = blank(prs)
    title_block(s, "03  双轨架构", "这样为什么能少等？", "用户感知的是「机器人什么时候开始动」，不是「后台全部算完」。")
    steps = [
        ("现在如果单轨死等", "看脸 10 秒 + 云 5 秒 + 生成 2 秒\n人要盯着十几秒的木头人"),
        ("改成 A + B", "表情一稳，A 立刻招手/鞠躬\nB 在后台写句子、出关节"),
        ("工程上还要改两刀", "摄像头不要每轮重开\nMoMask 不要每次重新加载"),
        ("改完之后的体感", "眼前马上有动作\n细动作大约两秒后补上"),
    ]
    for i, (t, d) in enumerate(steps):
        x = Inches(0.5 + (i % 2) * 6.4)
        y = Inches(1.95 + (i // 2) * 2.25)
        card(s, x, y, Inches(6.1), Inches(2.05))
        txt(s, x + Inches(0.25), y + Inches(0.25), Inches(5.6), Inches(0.45), t, 18, True, NAVY)
        txt(s, x + Inches(0.25), y + Inches(0.8), Inches(5.6), Inches(1.0), d, 16, False, MUTED)
    foot(s, 13, P)

    # 14 next
    s = blank(prs)
    title_block(s, "04  下一步", "三步，不并行抢", "先把看和听做好，再完整上树莓派，最后才碰记忆。")
    steps = [
        ("1", "视听又快又准",
         "摄像头常开，标定只做一次。\n少帧、稳表情。\n听改成板上的小模型，不再每次走云。"),
        ("2", "整套部署到端侧并实测",
         "Qwen 小模型装完再测写句子。\n和 MoMask 错开跑，避免 4GB 挤爆。\n有 16GB 真机后，再测同机完整链路。"),
        ("3", "这些结束再做记忆",
         "记住刚才聊过什么、情绪有没有变。\n用来改下一轮动作，而不是再堆一个大模型。\n现在接入会干扰主链路验收。"),
    ]
    for i, (n, t, d) in enumerate(steps):
        x = Inches(0.5 + i * 4.2)
        card(s, x, Inches(1.95), Inches(3.95), Inches(4.55))
        rect(s, x, Inches(1.95), Inches(3.95), Inches(1.15), NAVY)
        txt(s, x, Inches(2.05), Inches(3.95), Inches(0.5), n, 22, True, CORAL, PP_ALIGN.CENTER)
        txt(s, x, Inches(2.5), Inches(3.95), Inches(0.5), t, 18, True, WHITE, PP_ALIGN.CENTER)
        txt(s, x + Inches(0.25), Inches(3.35), Inches(3.45), Inches(2.8), d, 16, False, INK)
    foot(s, 14, P)

    # 15 one page
    s = blank(prs)
    title_block(s, "带走三句话", "今天如果只记住这些")
    lines = [
        "树上：一句话出动作，大约两秒，高兴动得开、悲伤收着。",
        "云上：慢主要是每次重开摄像头，加上每次重开 MoMask。不是模型突然变差。",
        "落地：A 先动、B 后补。整套上树莓派要等内存和安装收尾；记忆放最后。",
    ]
    for i, t in enumerate(lines):
        y = Inches(2.05 + i * 1.45)
        card(s, Inches(0.55), y, Inches(12.2), Inches(1.25))
        rect(s, Inches(0.55), y, Inches(0.14), Inches(1.25), CORAL)
        txt(s, Inches(1.0), y + Inches(0.35), Inches(11.4), Inches(0.6), t, 20, False, INK)
    foot(s, 15, P)

    # 16 thanks
    s = blank(prs)
    rect(s, 0, 0, W, H, NAVY)
    txt(s, Inches(0.7), Inches(2.7), Inches(12), Inches(1.0), "谢谢。", 40, True, WHITE)
    txt(s, Inches(0.7), Inches(3.85), Inches(12), Inches(0.9),
        "先对齐口径，再谈优化。", 20, False, RGBColor(0xC9, 0xD4, 0xD8))
    txt(s, Inches(0.7), Inches(6.6), Inches(12), Inches(0.4),
        "图来自 experiment/pi_bench_joints　　数字来自 100 条端侧实测与 Mac 联调 CSV",
        13, False, RGBColor(0x8A, 0xA0, 0xA8))

    prs.save(str(OUT))
    print("wrote", OUT)


if __name__ == "__main__":
    build()
