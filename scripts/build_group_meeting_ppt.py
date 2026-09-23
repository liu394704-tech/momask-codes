#!/usr/bin/env python3
"""Group-meeting slides along multimodal input, on-device judgment, and MoMask."""
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
    add_text(slide, "标准路径：多模态输入 → 端侧判断 → MoMask 生成", Inches(0.55), Inches(7.12), Inches(9), Inches(0.28), 11, MUTED)
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
    blank = prs.slide_layouts[6]
    total = 8

    s = prs.slides.add_slide(blank)
    rect(s, 0, 0, W, H, NAVY)
    rect(s, 0, 0, Inches(0.18), H, ACCENT)
    add_text(s, "组会阶段汇报", Inches(0.7), Inches(1.35), Inches(11), Inches(0.35), 16, RGBColor(0xE7, 0xD3, 0xC4))
    add_text(s, "多模态情绪输入 → 端侧判断 → MoMask 生成", Inches(0.7), Inches(1.85), Inches(12), Inches(0.85), 32, WHITE, True)
    add_text(s, "这是要做成的路径。当前工作只汇报已经落在这条路径上的部分。", Inches(0.7), Inches(2.85), Inches(11), Inches(0.4), 18, RGBColor(0xD5, 0xDE, 0xE4))
    add_text(s, "778 条预设短语是判断结果的临时动作出口，用来在 MoMask 还不能驱动舵机时看到身体反应。\n它不是这条路径的终点。", Inches(0.7), Inches(3.55), Inches(11), Inches(0.9), 16, RGBColor(0xF6, 0xF3, 0xEE))
    add_text(s, "2026年9月23日", Inches(0.7), Inches(6.4), Inches(6), Inches(0.35), 14, RGBColor(0xC5, 0xD0, 0xD6))

    s = prs.slides.add_slide(blank)
    bg(s)
    title_bar(s, "标准路径上，三步现在各在哪里", "只把真机已经发生的事放进「已通」。")
    cols = [
        (OK, "1  多模态输入", "脸：已通", "板上 FaceExpression。17 轮里出现高兴、中性、不高兴、惊讶。约 0.02 秒。", "语音：未通", "唤醒、录音、识别全是 0。没有 wakeup 这一行。"),
        (WARN, "2  端侧判断", "位置：已留在机器人上", "不走云端。Qwen 权重已在机内。决策接口按端侧来接。", "模型：这一轮没跑", "实际是规则，约 0.001 秒。原因是没有 llama_cpp。"),
        (ACCENT, "3  MoMask 生成", "接口：能出关节，不驱动身体", "同一条判断可以写出动作描述，再交给 MoMask 出关节文件。", "真机身体：仍是临时出口", "舵机播的是预设短语。关节还没有换成 16 个舵机。"),
    ]
    for i, (color, title, a, b, c, d) in enumerate(cols):
        x = Inches(0.4) + Inches(i * 4.28)
        card(s, x, Inches(1.45), Inches(4.1), Inches(5.35))
        rect(s, x, Inches(1.45), Inches(4.1), Inches(0.12), color)
        add_text(s, title, x + Inches(0.22), Inches(1.7), Inches(3.7), Inches(0.7), 18, NAVY, True)
        add_text(s, a, x + Inches(0.22), Inches(2.5), Inches(3.7), Inches(0.4), 15, OK, True)
        add_text(s, b, x + Inches(0.22), Inches(2.95), Inches(3.7), Inches(1.15), 14, INK)
        add_text(s, c, x + Inches(0.22), Inches(4.2), Inches(3.7), Inches(0.45), 15, BAD, True)
        add_text(s, d, x + Inches(0.22), Inches(4.7), Inches(3.7), Inches(1.3), 14, INK)
    footer(s, 2, total)

    s = prs.slides.add_slide(blank)
    bg(s)
    title_bar(s, "这一阶段适合汇报的内容", "每条都回答标准路径上的一个问题，不把临时出口说成目标。")
    items = [
        ("输入已经能进机器人", "WonderPi 的「人脸检测」打开后，摄像头画面和表情类别能进到决策。这是多模态里的视觉一路。"),
        ("判断被放在端侧", "真机不调用云端中转。权重用的是 Qwen2.5-1.5B、4-bit。这一轮没装上运行库，所以表里看到的是规则退回。"),
        ("判断结果已经能带动身体", "情绪出来之后会选一组动作并让舵机播出来。用来证明「判断 → 动作」这条线是通的。"),
        ("耗时被拆开了", "人脸约 0.02 秒，规则决策约 0.001 秒，动作播放约 2 到 9 秒。以后模型推理和 MoMask 生成可以记在同一张表里。"),
    ]
    for i, (head, body) in enumerate(items):
        y = Inches(1.4) + Inches(i * 1.32)
        card(s, Inches(0.5), y, Inches(12.3), Inches(1.2))
        rect(s, Inches(0.5), y, Inches(0.1), Inches(1.2), NAVY if i % 2 == 0 else ACCENT)
        add_text(s, head, Inches(0.85), y + Inches(0.12), Inches(11.6), Inches(0.36), 18, NAVY, True)
        add_text(s, body, Inches(0.85), y + Inches(0.52), Inches(11.6), Inches(0.55), 15, INK)
    footer(s, 3, total)

    s = prs.slides.add_slide(blank)
    bg(s)
    title_bar(s, "预设短语在标准路径里的位置", "它接在判断之后、MoMask 上身之前。")
    card(s, Inches(0.5), Inches(1.45), Inches(12.3), Inches(1.35))
    add_text(s, "现在：  脸（语音空着）  →  规则判断  →  778 条短语里的一组  →  舵机", Inches(0.75), Inches(1.65), Inches(11.8), Inches(0.4), 18, NAVY, True)
    add_text(s, "目标：  脸 + 语音  →  端侧模型判断  →  MoMask 生成关节  →  重定向到 16 个舵机", Inches(0.75), Inches(2.15), Inches(11.8), Inches(0.4), 18, ACCENT, True)
    add_lines(s, [
        "可以讲：短语让我们在生成模型还不能驱动舵机时，先看到情绪对应的身体反应，并暴露动作质量问题。",
        "不要讲成：系统已经用 778 条短语完成了动作生成。",
        "短语上已经看到两个问题，留在临时出口里修，不改变终点：同类转身隔几轮重复；举手再下蹲时手臂碰到腿。规则已改，尚未再次上机。",
    ], Inches(0.6), Inches(3.1), Inches(12.1), Inches(3.2), 18, INK, 14)
    footer(s, 4, total)

    s = prs.slides.add_slide(blank)
    bg(s)
    title_bar(s, "真机 17 轮，对照标准路径", "来源：wonderpi_latency.csv。这张表还没有 MoMask 的生成时间。")
    headers = ["标准路径上的一步", "这 17 轮", "能不能当成完成"]
    data = [
        ("视觉输入", "16 轮由脸触发，四类情绪都有", "能"),
        ("语音输入", "录音和识别都是 0，没有唤醒行", "不能"),
        ("端侧模型判断", "请求了端侧模型，实际是规则", "不能"),
        ("判断耗时", "约 0.001 秒", "只代表规则"),
        ("身体动作", "播放约 2.3–8.8 秒", "是短语，不是 MoMask"),
        ("MoMask 生成", "这张表没有这一列", "还没测"),
    ]
    top = Inches(1.4)
    rect(s, Inches(0.5), top, Inches(12.3), Inches(0.48), NAVY)
    xs = [Inches(0.7), Inches(4.3), Inches(9.5)]
    for x, htxt in zip(xs, headers):
        add_text(s, htxt, x, top + Inches(0.08), Inches(3.5), Inches(0.32), 14, WHITE, True)
    for i, row in enumerate(data):
        y = top + Inches(0.48) + Inches(i * 0.78)
        rect(s, Inches(0.5), y, Inches(12.3), Inches(0.78), WHITE if i % 2 == 0 else RGBColor(0xFB, 0xF8, 0xF4))
        color = OK if row[2] == "能" else BAD
        add_text(s, row[0], xs[0], y + Inches(0.18), Inches(3.4), Inches(0.42), 16, INK, True)
        add_text(s, row[1], xs[1], y + Inches(0.18), Inches(5.0), Inches(0.42), 16, INK)
        add_text(s, row[2], xs[2], y + Inches(0.18), Inches(2.8), Inches(0.42), 16, color, True)
    footer(s, 5, total)

    s = prs.slides.add_slide(blank)
    bg(s)
    title_bar(s, "相对标准路径，还差什么", "")
    gaps = [
        ("输入差一路", "视觉有了，听觉没有。多模态现在实际是单模态。语音小板要先把「小幻小幻」送进程序，表里出现 wakeup，并且录音时间大于 0。"),
        ("判断还不是模型", "权重在，运行库不在。树莓派是 aarch64，现成的 llama-cpp 轮子对不上。装上之前，不能汇报端侧模型的推理时间。"),
        ("生成还没接到身体", "MoMask 可以按文字出关节，但没有从人体关节到 TonyPi 16 个舵机的实时对应。所以身体仍播工厂动作，不播生成结果。"),
        ("临时出口自己的问题", "短语会重复同类转身，也会把举手和下蹲排在一起。这是出口质量，不是生成模型的结果。"),
    ]
    for i, (head, body) in enumerate(gaps):
        y = Inches(1.35) + Inches(i * 1.35)
        card(s, Inches(0.5), y, Inches(12.3), Inches(1.22))
        add_text(s, head, Inches(0.75), y + Inches(0.12), Inches(11.8), Inches(0.36), 18, NAVY, True)
        add_text(s, body, Inches(0.75), y + Inches(0.52), Inches(11.8), Inches(0.58), 15, INK)
    footer(s, 6, total)

    s = prs.slides.add_slide(blank)
    bg(s)
    title_bar(s, "下阶段按这条路径往前", "顺序不要反过来：先补输入，再让判断真的是模型，然后才让 MoMask 上身。")
    plans = [
        ("1", "补上语音输入", "接通唤醒。日志里要有 wakeup，录音大于 0。脸和语音一起进同一次判断。"),
        ("2", "让端侧模型真正判断", "在 aarch64 上装上 llama-cpp，用已有的 Qwen 权重。表里的实际决策不再是规则，并记下推理时间。"),
        ("3", "用同一次判断跑 MoMask", "模型写出的动作描述交给 MoMask，只生成关节文件，先不驱动舵机。把生成时间写进同一张表。"),
        ("4", "再考虑替换临时出口", "有了关节到 16 舵机的对应之后，身体才改播生成结果。在那之前，短语继续负责能看的动作，并把重复和碰撞收干净。"),
    ]
    for i, (n, head, body) in enumerate(plans):
        y = Inches(1.4) + Inches(i * 1.32)
        card(s, Inches(0.5), y, Inches(12.3), Inches(1.2))
        add_text(s, n, Inches(0.75), y + Inches(0.32), Inches(0.55), Inches(0.5), 22, ACCENT, True)
        add_text(s, head, Inches(1.5), y + Inches(0.12), Inches(10.8), Inches(0.38), 18, NAVY, True)
        add_text(s, body, Inches(1.5), y + Inches(0.54), Inches(10.8), Inches(0.52), 15, INK)
    footer(s, 7, total)

    s = prs.slides.add_slide(blank)
    bg(s)
    title_bar(s, "汇报时可以怎么说", "")
    points = [
        ("路径", "要做的是：多模态情绪进来，端侧模型做判断，MoMask 生成动作。"),
        ("这一阶段", "视觉输入和「判断带动身体」已经在真机上看到。判断目前是规则替身，动作目前是预设短语。"),
        ("不要说成完成", "不要说语音已经多模态，不要说端侧模型已经在判断，不要说机器人在播 MoMask。"),
        ("下一步", "先让语音进表，再让表里的决策变成模型，然后记一次 MoMask 只出关节的时间。"),
    ]
    for i, (head, body) in enumerate(points):
        y = Inches(1.4) + Inches(i * 1.3)
        rect(s, Inches(0.55), y, Inches(0.12), Inches(1.1), ACCENT if i == 0 else NAVY)
        add_text(s, head, Inches(0.9), y + Inches(0.28), Inches(2.2), Inches(0.45), 18, NAVY, True)
        add_text(s, body, Inches(3.2), y + Inches(0.22), Inches(9.4), Inches(0.7), 16, INK)
    footer(s, 8, total)

    out = "/workspace/docs/组会_TonyPi多模态阶段汇报.pptx"
    prs.save(out)
    print(out)


if __name__ == "__main__":
    build()

