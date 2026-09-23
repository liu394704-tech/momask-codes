#!/usr/bin/env python3
"""Group-meeting slides for the TonyPi multimodal phrase loop."""
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

FONT = "WenQuanYi Micro Hei"
INK = RGBColor(0x1C, 0x24, 0x30)
NAVY = RGBColor(0x1B, 0x3A, 0x4B)
ACCENT = RGBColor(0xC4, 0x5C, 0x26)
MUTED = RGBColor(0x5C, 0x6B, 0x73)
PAPER = RGBColor(0xF6, 0xF3, 0xEE)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
CARD = RGBColor(0xFF, 0xFF, 0xFF)
LINE = RGBColor(0xE2, 0xDB, 0xD2)
OK = RGBColor(0x2F, 0x6F, 0x4E)
BAD = RGBColor(0xA3, 0x3B, 0x2B)
WARN = RGBColor(0x8A, 0x5A, 0x12)

W, H = Inches(13.333), Inches(7.5)


def _set_run(run, text, size, color, bold=False):
    run.text = text
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    rpr = run._r.get_or_add_rPr()
    for tag in ("latin", "ea", "cs"):
        node = rpr.find(qn("a:%s" % tag))
        if node is None:
            node = rpr.makeelement(qn("a:%s" % tag), {})
            rpr.append(node)
        node.set("typeface", FONT)


def add_text(slide, text, x, y, w, h, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    p = tf.paragraphs[0]
    p.alignment = align
    _set_run(p.add_run(), text, size, color, bold)
    return box


def add_lines(slide, lines, x, y, w, h, size=16, color=INK, spacing=8):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(spacing)
        bold = line.startswith("**")
        text = line[2:] if bold else line
        _set_run(p.add_run(), text, size, color, bold)
    return box


def rect(slide, x, y, w, h, fill, line=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
    shape.shadow.inherit = False
    return shape


def card(slide, x, y, w, h):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = CARD
    shape.line.color.rgb = LINE
    shape.shadow.inherit = False
    return shape


def footer(slide, page, total):
    add_text(slide, "TonyPi 多模态情绪动作  ·  组会汇报", Inches(0.55), Inches(7.12), Inches(8), Inches(0.28), 11, MUTED)
    add_text(slide, "%d / %d" % (page, total), Inches(11.4), Inches(7.12), Inches(1.4), Inches(0.28), 11, MUTED, align=PP_ALIGN.RIGHT)


def bg(slide):
    rect(slide, 0, 0, W, H, PAPER)
    rect(slide, 0, 0, Inches(0.12), H, NAVY)


def title_bar(slide, title, sub=None):
    add_text(slide, title, Inches(0.55), Inches(0.32), Inches(12), Inches(0.5), 28, NAVY, True)
    if sub:
        add_text(slide, sub, Inches(0.55), Inches(0.84), Inches(12), Inches(0.34), 14, MUTED)


def build():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    prs.slides._sldIdLst
    blank = prs.slide_layouts[6]
    total = 8

    # 1 cover
    s = prs.slides.add_slide(blank)
    rect(s, 0, 0, W, H, NAVY)
    rect(s, 0, 0, Inches(0.18), H, ACCENT)
    add_text(s, "组会阶段汇报", Inches(0.7), Inches(1.55), Inches(10), Inches(0.4), 16, RGBColor(0xE7, 0xD3, 0xC4))
    add_text(s, "TonyPi 多模态情绪动作", Inches(0.7), Inches(2.05), Inches(11), Inches(0.7), 40, WHITE, True)
    add_text(s, "人脸识别  ·  778 条动作短语  ·  WonderPi 入口  ·  真机延迟记录", Inches(0.7), Inches(2.85), Inches(11), Inches(0.4), 18, RGBColor(0xD5, 0xDE, 0xE4))
    add_text(s, "本阶段：先让机器人按表情做一组不重复的社交动作，并记下每一步耗时。\n语音唤醒和端侧大模型尚未在真机上跑通。", Inches(0.7), Inches(3.7), Inches(10), Inches(0.9), 16, RGBColor(0xF6, 0xF3, 0xEE))
    add_text(s, "2026年9月23日", Inches(0.7), Inches(6.4), Inches(6), Inches(0.35), 14, RGBColor(0xC5, 0xD0, 0xD6))

    # 2 goal + done
    s = prs.slides.add_slide(blank)
    bg(s)
    title_bar(s, "本阶段目标与已完成", "目标是真机上能看见表情对应的动作，而不是先接 MoMask 关节。")
    items = [
        ("目标", "摄像头看脸，语音听唤醒。按情绪从 778 条短语里选一组 2–3 个动作。连续几组不要看起来一样。每次把耗时写进表。"),
        ("已接到机器人", "WonderPi 里原有的「人脸检测」就是入口，没有新按钮。点进去会先挥手，脸留在画面里就播一组动作。"),
        ("情绪", "使用板上的 FaceExpression。四类都出现过：高兴、中性、不高兴、惊讶。单帧大约 0.02 秒。"),
        ("动作", "778 条社交短语。踢、打类不播。身体仍播工厂动作文件。16 路舵机坐标是按机身尺寸算的对照表，不直接驱动舵机。"),
    ]
    for i, (head, body) in enumerate(items):
        y = Inches(1.4) + Inches(i * 1.3)
        card(s, Inches(0.5), y, Inches(12.3), Inches(1.18))
        rect(s, Inches(0.5), y, Inches(0.1), Inches(1.18), ACCENT if i == 0 else NAVY)
        add_text(s, head, Inches(0.85), y + Inches(0.14), Inches(2.2), Inches(0.36), 16, NAVY, True)
        add_text(s, body, Inches(3.1), y + Inches(0.18), Inches(9.3), Inches(0.84), 16, INK)
    footer(s, 2, total)

    # 3 path
    s = prs.slides.add_slide(blank)
    bg(s)
    title_bar(s, "现在的链路", "手机只负责打开「人脸检测」。识别和选动作都在机器人里。")
    steps = [
        ("1", "人脸", "FaceExpression\n约 0.02 秒"),
        ("2", "语音", "唤醒后录音\n本轮未进入"),
        ("3", "决策", "想用端侧 Qwen\n实际落到规则"),
        ("4", "短语", "778 选 1 组\n2–3 个动作"),
        ("5", "记录", "每轮一行\nwonderpi_latency.csv"),
    ]
    for i, (n, name, desc) in enumerate(steps):
        x = Inches(0.45) + Inches(i * 2.55)
        card(s, x, Inches(1.7), Inches(2.35), Inches(2.5))
        add_text(s, n, x + Inches(0.15), Inches(1.85), Inches(2), Inches(0.4), 20, ACCENT, True)
        add_text(s, name, x + Inches(0.15), Inches(2.3), Inches(2), Inches(0.4), 20, NAVY, True)
        add_text(s, desc, x + Inches(0.15), Inches(2.85), Inches(2.05), Inches(0.9), 14, MUTED)
        if i < 4:
            add_text(s, "→", x + Inches(2.2), Inches(2.55), Inches(0.4), Inches(0.4), 20, ACCENT, True)
    add_lines(s, [
        "真机 17 轮：1 次是点进按钮，16 次是看到脸之后自动播。",
        "决策请求的是 edge_auto。表里每一行的实际决策都是 edge_rule。",
        "从开始到决定播哪一组，几乎等于动作播放时间，大约 2 到 9 秒。识别本身不是这段时间。",
    ], Inches(0.55), Inches(4.55), Inches(12.2), Inches(2.2), 18, INK, 10)
    footer(s, 3, total)

    # 4 log table summary
    s = prs.slides.add_slide(blank)
    bg(s)
    title_bar(s, "真机日志：17 轮里实际发生了什么", "来源：wonderpi_latency.csv")
    headers = ["项目", "日志里的结果", "说明"]
    rows = [
        ("触发", "enter ×1，face ×16", "没有 wakeup"),
        ("录音 / 语音识别", "全部为 0，文本为空", "唤醒词没有进程序"),
        ("人脸情绪", "高兴、中性、不高兴、惊讶", "脸这条是通的"),
        ("人脸耗时", "约 0.018–0.022 秒", "不是瓶颈"),
        ("决策", "请求 edge_auto，实际 edge_rule", "No module named llama_cpp"),
        ("决策耗时", "约 0.001 秒", "规则，不是大模型"),
        ("动作播放", "约 2.3–8.8 秒", "身体在动，所以总时间长"),
    ]
    top = Inches(1.4)
    rect(s, Inches(0.5), top, Inches(12.3), Inches(0.46), NAVY)
    xs = [Inches(0.65), Inches(3.3), Inches(7.4)]
    widths = [Inches(2.5), Inches(3.9), Inches(5.1)]
    for x, w, htxt in zip(xs, widths, headers):
        add_text(s, htxt, x, top + Inches(0.06), w, Inches(0.34), 14, WHITE, True)
    for i, row in enumerate(rows):
        y = top + Inches(0.46) + Inches(i * 0.68)
        fill = WHITE if i % 2 == 0 else RGBColor(0xFB, 0xF8, 0xF4)
        rect(s, Inches(0.5), y, Inches(12.3), Inches(0.68), fill)
        for x, w, val in zip(xs, widths, row):
            add_text(s, val, x, y + Inches(0.14), w, Inches(0.4), 14, INK, i == 0 and False)
    footer(s, 4, total)

    # 5 problems
    s = prs.slides.add_slide(blank)
    bg(s)
    title_bar(s, "存在的问题", "都对应这张表，或对应表里能看到的动作名字。")
    problems = [
        (BAD, "语音完全没有唤醒", "17 轮的触发只有 enter 和 face。录音、识别、文本都是空的，连识别错误都没有。程序没收到「小幻小幻」，语音小板没有把词送进 /dev/ttyUSB0。"),
        (WARN, "端侧大模型没有跑", "权重文件在机器人上，运行库没有。每一行都退回规则，决策约 0.001 秒。这不是 Qwen 的耗时。"),
        (ACCENT, "动作看起来在重复", "规则只禁止完全同名的片段，而且只禁 3 轮。左转和右转算两个名字，所以隔几轮又是一次转身。"),
        (NAVY, "手臂和腿会碰到", "举手、挥手可以和鞠躬、下蹲排在同一组。这两组关节叠在一起，前臂会碰到腿。"),
    ]
    for i, (color, head, body) in enumerate(problems):
        col = i % 2
        row = i // 2
        x = Inches(0.45) + Inches(col * 6.4)
        y = Inches(1.45) + Inches(row * 2.6)
        card(s, x, y, Inches(6.15), Inches(2.4))
        rect(s, x, y, Inches(0.1), Inches(2.4), color)
        add_text(s, head, x + Inches(0.3), y + Inches(0.2), Inches(5.6), Inches(0.45), 18, NAVY, True)
        add_text(s, body, x + Inches(0.3), y + Inches(0.75), Inches(5.6), Inches(1.45), 15, INK)
    footer(s, 5, total)

    # 6 what was changed
    s = prs.slides.add_slide(blank)
    bg(s)
    title_bar(s, "针对重复和碰撞已改的规则", "代码已写好。要在机器人上再装一次，才会体现在下一次测试里。")
    rows = [
        ("原来", "现在"),
        ("只禁同名片段，禁 3 轮", "同名片段禁 6 轮"),
        ("左转、右转当成不同动作", "所有转身算一类，接下来 4 组不再转"),
        ("侧步、举手、下蹲也各自可马上再来", "侧步、前进后退、举手、鞠躬、下蹲各自成类，同样隔开"),
        ("举手可以接着鞠躬或下蹲", "这两组不再放进同一条短语"),
    ]
    rect(s, Inches(0.55), Inches(1.5), Inches(12.2), Inches(0.5), NAVY)
    add_text(s, "原来", Inches(0.75), Inches(1.58), Inches(5.5), Inches(0.35), 16, WHITE, True)
    add_text(s, "现在", Inches(6.8), Inches(1.58), Inches(5.5), Inches(0.35), 16, WHITE, True)
    for i, (old, new) in enumerate(rows[1:]):
        y = Inches(2.0) + Inches(i * 1.05)
        rect(s, Inches(0.55), y, Inches(6.0), Inches(0.95), WHITE)
        rect(s, Inches(6.65), y, Inches(6.1), Inches(0.95), RGBColor(0xF3, 0xF7, 0xF4))
        add_text(s, old, Inches(0.75), y + Inches(0.22), Inches(5.6), Inches(0.55), 16, MUTED)
        add_text(s, new, Inches(6.85), y + Inches(0.22), Inches(5.7), Inches(0.55), 16, OK, True)
    footer(s, 6, total)

    # 7 next
    s = prs.slides.add_slide(blank)
    bg(s)
    title_bar(s, "下阶段计划", "先把已经改过的规则装回机器人，再补语音。大模型排在语音之后。")
    plans = [
        ("1", "再上机验证动作", "重新安装后连测十几轮。看表里相邻几组是否还出现同一类转身或举手。同时看手臂和下蹲是否还叠在一起。"),
        ("2", "接通语音唤醒", "确认语音小板在 /dev/ttyUSB0。说「小幻小幻」后，表里要出现一行 wakeup，并且录音时间大于 0。"),
        ("3", "再测语音识别耗时", "唤醒通了之后，才安装本地识别。现在表里没有这一列的有效数字，不能报一个识别时间。"),
        ("4", "端侧大模型", "权重已经在机器人上。缺的是 aarch64 上的 llama-cpp。装上之前，决策继续用规则，不把 0.001 秒说成模型时间。"),
    ]
    for i, (n, head, body) in enumerate(plans):
        y = Inches(1.4) + Inches(i * 1.32)
        card(s, Inches(0.5), y, Inches(12.3), Inches(1.2))
        add_text(s, n, Inches(0.75), y + Inches(0.32), Inches(0.6), Inches(0.5), 22, ACCENT, True)
        add_text(s, head, Inches(1.5), y + Inches(0.14), Inches(4), Inches(0.4), 18, NAVY, True)
        add_text(s, body, Inches(1.5), y + Inches(0.55), Inches(10.8), Inches(0.55), 15, INK)
    footer(s, 7, total)

    # 8 close
    s = prs.slides.add_slide(blank)
    bg(s)
    title_bar(s, "这一阶段可以怎么说", "")
    points = [
        ("做成了", "WonderPi「人脸检测」能看脸，并按四类情绪播社交动作。耗时表能分开人脸、决策和动作播放。"),
        ("还没做成", "语音唤醒没有进日志。端侧 Qwen 没有运行。MoMask 关节不驱动舵机。"),
        ("日志说明", "人会感觉到的等待，主要是动作播放的 2 到 9 秒，不是识别。"),
        ("下一步", "先复测去重和碰撞，再让表里出现 wakeup，最后才测大模型和语音识别时间。"),
    ]
    for i, (head, body) in enumerate(points):
        y = Inches(1.35) + Inches(i * 1.25)
        rect(s, Inches(0.55), y, Inches(0.12), Inches(1.05), ACCENT if i == 0 else NAVY)
        add_text(s, head, Inches(0.9), y + Inches(0.05), Inches(2.4), Inches(0.4), 18, NAVY, True)
        add_text(s, body, Inches(3.4), y + Inches(0.08), Inches(9.2), Inches(0.85), 16, INK)
    footer(s, 8, total)

    out = "/workspace/docs/组会_TonyPi多模态阶段汇报.pptx"
    prs.save(out)
    print(out)


if __name__ == "__main__":
    build()
