#!/usr/bin/env python3
"""两页实验方案。写入仓库，并覆盖云端桌面上的同名 PPT。"""
from __future__ import annotations

import shutil
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

REPO = Path(__file__).resolve().parent / "陪伴机器人记忆遗忘_综述与实验方案.pptx"
DESKTOP = Path("/home/ubuntu/Desktop/陪伴机器人记忆遗忘_综述与实验方案.pptx")

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


def paras(slide, l, t, w, h, lines, size=14, color=INK, spacing=6):
    box = slide.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(spacing)
        r = p.add_run()
        font(r, line, size, False, color)
    return box


def blank(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, W, H, CREAM)
    return s


def card(slide, l, t, w, h, fill=WHITE):
    sh = rect(slide, l, t, w, h, fill)
    sh.line.fill.solid()
    sh.line.color.rgb = LINE
    sh.line.width = Pt(1)
    return sh


def foot(slide, page):
    txt(slide, Inches(0.42), Inches(7.15), Inches(10), Inches(0.26),
        "ForgeBench-E  ·  个人实验方案  ·  2026.10.29", 11, False, MUTED)
    txt(slide, Inches(11.5), Inches(7.15), Inches(1.4), Inches(0.26),
        "%d / 2" % page, 11, False, MUTED, PP_ALIGN.RIGHT)


def build():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    prs.core_properties.title = "陪伴机器人记忆遗忘：个人实验方案"
    prs.core_properties.subject = "ForgeBench-E 两页方案"

    # ----- 1 方案建立 -----
    s = blank(prs)
    rect(s, 0, 0, Inches(0.12), H, NAVY)
    txt(s, Inches(0.42), Inches(0.22), Inches(12), Inches(0.28), "ForgeBench-E", 13, False, TEAL)
    txt(s, Inches(0.42), Inches(0.48), Inches(12.4), Inches(0.46), "个人实验方案怎么建立", 28, True, NAVY)
    txt(s, Inches(0.42), Inches(1.0), Inches(12.4), Inches(0.34),
        "评的是记忆本会不会真的忘掉。大模型只做决策骨干；同一骨干换记忆本，差出来的才算记忆分。",
        14, False, MUTED)
    rect(s, Inches(0.42), Inches(1.4), Inches(1.05), Inches(0.055), CORAL)

    blocks = [
        ("评什么",
         "链路是：感知 → 记忆本 → 决策 → 动作。分数记在记忆本上。舵机或 MoMask 没做成，另记执行分，不进遗忘分。\n"
         "主实验只评事实：口味、约定、住址、血压、关系。社交上该点头还是拥抱，以后单独做，不进主表。"),
        ("一条数据",
         "先做 L2：写好的文字直接送进记忆本。每条有遗忘版和对照版，两版只差删除那一句。对照版没记住，遗忘不计分。\n"
         "机器人没看见的记「未见」。删甜度要留下咖啡温度；删血压要留下搀扶。删完之后还要能学会同类新事实。"),
        ("怎么记分",
         "五处一起查：语言、规划、记忆本、画像、动作。问句里不放答案。\n"
         "总评 FER：五处都干净，红队追问也钓不出，该留的还在，并且对照版当时记得。只看嘴上干不干净，不算忘掉。"),
        ("下一步",
         "现有 MemoryStore 还没有删除接口，衰减强度也不参与检索。先补上删除、让衰减进入检索、删后重建画像、清掉动作缓存。\n"
         "通过线：对照版存入 ≥ 0.8；至少一处比语言更脏；保持率不低于 0.8。然后再扩到 80 条成对数据。"),
    ]
    for i, (title, body) in enumerate(blocks):
        col, row = i % 2, i // 2
        x = Inches(0.42 + col * 6.45)
        y = Inches(1.62 + row * 2.65)
        card(s, x, y, Inches(6.25), Inches(2.5))
        txt(s, x + Inches(0.2), y + Inches(0.12), Inches(5.85), Inches(0.38), title, 18, True, NAVY)
        txt(s, x + Inches(0.2), y + Inches(0.55), Inches(5.85), Inches(1.8), body, 14, False, INK)
    foot(s, 1)

    # ----- 2 已跑的小测试 -----
    s = blank(prs)
    rect(s, 0, 0, Inches(0.12), H, NAVY)
    txt(s, Inches(0.42), Inches(0.22), Inches(12), Inches(0.28), "已有结果", 13, False, TEAL)
    txt(s, Inches(0.42), Inches(0.48), Inches(12.4), Inches(0.46), "已经跑过的小测试", 28, True, NAVY)
    txt(s, Inches(0.42), Inches(1.0), Inches(12.4), Inches(0.34),
        "用来确认链路和尺子。下面的数字不是某个大模型的遗忘分数。",
        14, False, MUTED)
    rect(s, Inches(0.42), Inches(1.4), Inches(1.05), Inches(0.055), CORAL)

    rows = [
        (CREAM, "记忆本冒烟",
         "tests/test_memory_hierarchy.py 写了 7 个检查：情绪画像、按情绪检索、丢掉低置信度、纯音频也能入库、能回链到离线 JSON、关键词检索、提示词少于 800 token。",
         "和遗忘直接相关、并且这次复跑通过的一点：现有 MemoryStore 没有 forget()。strength 只是占位，检索不读它。7 个夹具检查依赖的会话文件不在当前检出里，这次没有重跑出新数字。"),
        (WHITE, "决策冒烟",
         "experiment/e2e_spec/decision_smoke_summary.json：mock 规则 20 条全部通过，pass_rate = 1.0。",
         "检查的是决策输出合不合法。它不调用大模型，也不测遗忘。"),
        (WHITE, "遗忘尺子",
         "experiment/forget_bench：20 个家庭故事、144 条探针、7 种记账规则。单测 9 项，这次复跑全部通过。",
         "情节层：语言遗忘率 1.000，行动泄漏 0.933，FER = 0。林阿姨的「不加糖」删掉后，嘴上说没有这项记录，手上仍倒黑咖啡。主题摘要 FER 0.853，误伤 0.625，删血压时搀扶也停了。什么都不记的系统 FER = 0；五处都擦掉的 Oracle，FER-E = 1。"),
    ]
    y = 1.62
    heights = [1.7, 1.35, 2.15]
    for (fill, title, a, b), h in zip(rows, heights):
        card(s, Inches(0.42), Inches(y), Inches(12.48), Inches(h), fill if fill != CREAM else RGBColor(0xE7, 0xF0, 0xF0))
        txt(s, Inches(0.62), Inches(y + 0.1), Inches(12.0), Inches(0.32), title, 16, True, NAVY)
        txt(s, Inches(0.62), Inches(y + 0.44), Inches(12.05), Inches(h - 0.52), a + "\n" + b, 13, False, INK)
        y += h + 0.1
    foot(s, 2)

    if len(prs.slides) != 2:
        raise SystemExit("expected 2 slides")
    REPO.parent.mkdir(parents=True, exist_ok=True)
    DESKTOP.parent.mkdir(parents=True, exist_ok=True)
    prs.save(REPO)
    shutil.copyfile(REPO, DESKTOP)
    return REPO, DESKTOP


if __name__ == "__main__":
    repo, desktop = build()
    print(repo)
    print(desktop)
