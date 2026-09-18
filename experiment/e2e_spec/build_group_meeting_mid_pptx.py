#!/usr/bin/env python3
"""组会 PPT：正式但不刻板，四段结构不变。"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path("/Users/emmaliu/Desktop/HKUST/RBM-project/Model/momask-codes")
OUT = ROOT / "experiment/e2e_spec/组会汇报_20260820.pptx"
FIG_LAT = ROOT / "experiment/pi_bench_joints/fig_latency.png"
FIG_MOT = ROOT / "experiment/pi_bench_joints/fig_motion_quality.png"

NAVY = RGBColor(0x1A, 0x36, 0x4A)
SLATE = RGBColor(0x3D, 0x4F, 0x5F)
TEAL = RGBColor(0x2C, 0x6E, 0x6A)
INK = RGBColor(0x1F, 0x24, 0x28)
MUTED = RGBColor(0x5A, 0x64, 0x6E)
LINE = RGBColor(0xD9, 0xE0, 0xE6)
BG = RGBColor(0xF6, 0xF8, 0xFA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
SOFT = RGBColor(0xEE, 0xF3, 0xF6)
FONT = "PingFang SC"
W, H = Inches(13.333), Inches(7.5)
N_PAGES = 15


def fnt(run, text, size, bold=False, color=INK):
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
    fnt(r, text, size, bold, color)
    return box


def slide_bg(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, W, H, BG)
    return s


def header(slide, section, title, sub=""):
    rect(slide, 0, 0, W, Inches(1.18), NAVY)
    rect(slide, 0, Inches(1.18), W, Inches(0.05), TEAL)
    txt(slide, Inches(0.55), Inches(0.16), Inches(12), Inches(0.28), section, 12, False, RGBColor(0xA8, 0xC4, 0xC2))
    txt(slide, Inches(0.55), Inches(0.42), Inches(12.2), Inches(0.42), title, 24, True, WHITE)
    if sub:
        txt(slide, Inches(0.55), Inches(0.84), Inches(12.2), Inches(0.28), sub, 12, False, RGBColor(0xC5, 0xD2, 0xD8))
    rect(slide, 0, Inches(7.28), W, Inches(0.22), NAVY)
    txt(slide, Inches(0.55), Inches(7.28), Inches(10), Inches(0.22),
        "社交机器人  ·  感知—决策—动作生成  ·  组会 2026.08.20", 10, False, WHITE)


def card(slide, l, t, w, h):
    sh = rect(slide, l, t, w, h, WHITE)
    sh.line.fill.solid()
    sh.line.color.rgb = LINE
    sh.line.width = Pt(0.75)
    return sh


def build():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H

    # cover
    s = slide_bg(prs)
    rect(s, 0, 0, W, H, NAVY)
    rect(s, 0, 0, Inches(0.14), H, TEAL)
    txt(s, Inches(0.7), Inches(1.7), Inches(12), Inches(0.35), "组会进展汇报", 14, False, TEAL)
    txt(s, Inches(0.7), Inches(2.15), Inches(12), Inches(1.2),
        "感知、决策与端侧动作生成", 34, True, WHITE)
    txt(s, Inches(0.7), Inches(3.5), Inches(11.5), Inches(1.1),
        "端侧 MoMask 基准已完成；云侧打通视听决策链路并完成时延拆解。\n"
        "提出 A∥B 双轨作为工程落地路径，并明确后续部署与优化顺序。",
        16, False, RGBColor(0xC5, 0xD2, 0xD8))
    txt(s, Inches(0.7), Inches(6.55), Inches(12), Inches(0.4),
        "HKUST  ·  RBM Project  ·  2026.08.20", 14, False, WHITE)

    # outline
    s = slide_bg(prs)
    header(s, "提纲", "汇报结构")
    items = [
        ("01", "端侧 MoMask 测试", "树莓派文本→关节的时延与动作质量"),
        ("02", "云侧联调", "视听觉处理、所用模型、时延成因，以及尚未完整上板的原因"),
        ("03", "A∥B 双轨方案", "用预置动作承接交互，用 MoMask 提供细粒度表达"),
        ("04", "下一步", "视听优化 → 端侧完整部署与实测 → 记忆相关工作"),
    ]
    for i, (n, t, d) in enumerate(items):
        y = Inches(1.55 + i * 1.28)
        card(s, Inches(0.55), y, Inches(12.2), Inches(1.12))
        rect(s, Inches(0.55), y, Inches(1.2), Inches(1.12), TEAL if i % 2 == 0 else NAVY)
        txt(s, Inches(0.55), y + Inches(0.32), Inches(1.2), Inches(0.45), n, 20, True, WHITE, PP_ALIGN.CENTER)
        txt(s, Inches(2.0), y + Inches(0.18), Inches(10.4), Inches(0.4), t, 20, True, NAVY)
        txt(s, Inches(2.0), y + Inches(0.6), Inches(10.4), Inches(0.38), d, 14, False, MUTED)

    # 1.1 kpi
    s = slide_bg(prs)
    header(s, "01  端侧 MoMask", "测试结论",
           "设备：Raspberry Pi 5（4GB，CPU，4 线程）　·　100 条情绪均衡指令　·　计入 98 条")
    kpis = [
        ("2.30 s", "平均生成时延"),
        ("2.33 s", "中位数"),
        ("2.64 s", "P95"),
        ("100%", "关节数值有效率"),
    ]
    for i, (n, lab) in enumerate(kpis):
        x = Inches(0.55 + i * 3.15)
        card(s, x, Inches(1.55), Inches(2.95), Inches(2.05))
        txt(s, x, Inches(1.75), Inches(2.95), Inches(0.85), n, 28, True, NAVY, PP_ALIGN.CENTER)
        txt(s, x, Inches(2.65), Inches(2.95), Inches(0.55), lab, 14, False, MUTED, PP_ALIGN.CENTER)
    card(s, Inches(0.55), Inches(3.85), Inches(12.2), Inches(2.95))
    txt(s, Inches(0.85), Inches(4.1), Inches(11.6), Inches(0.4), "说明", 16, True, TEAL)
    txt(s, Inches(0.85), Inches(4.55), Inches(11.6), Inches(1.95),
        "模型加载一次约 10.1 秒，不计入单条平均。加载完成后，单条文本生成关节坐标约 2.3 秒。\n"
        "输出为 (T, 22, 3) 关节序列，本次测试关闭视频渲染，聚焦推理时延与数值稳定性。\n"
        "该 2.3 秒应作为端侧动作生成的基准指标。",
        16, False, INK)

    # 1.2 latency fig
    s = slide_bg(prs)
    header(s, "01  端侧 MoMask", "时延分布",
           "左图：98 条端到端耗时直方图；右图：各情绪平均时延")
    if FIG_LAT.exists():
        s.shapes.add_picture(str(FIG_LAT), Inches(0.55), Inches(1.5), Inches(12.2), Inches(3.7))
    card(s, Inches(0.55), Inches(5.35), Inches(12.2), Inches(1.65))
    txt(s, Inches(0.85), Inches(5.55), Inches(11.6), Inches(1.25),
        "样本集中在 2.2–2.4 秒，未出现明显长尾。各情绪平均时延均在 2.2–2.5 秒，差异不大。\n"
        "生成快慢主要由设备与推理配置决定，对情绪类别不敏感。",
        15, False, INK)

    # 1.3 motion fig
    s = slide_bg(prs)
    header(s, "01  端侧 MoMask", "动作质量（代理指标）",
           "无真值标注，用关节平均速度与加加速度（jerk）观察不同情绪下的运动幅度")
    if FIG_MOT.exists():
        s.shapes.add_picture(str(FIG_MOT), Inches(0.55), Inches(1.5), Inches(12.2), Inches(3.7))
    card(s, Inches(0.55), Inches(5.35), Inches(12.2), Inches(1.65))
    txt(s, Inches(0.85), Inches(5.55), Inches(11.6), Inches(1.25),
        "高兴类别速度与 jerk 最高，悲伤最低，与情绪对应的动作强度一致。\n"
        "说明文本条件确实在调节运动风格，而不是对所有提示生成同一套动作。jerk 偏高并不等于质量差。",
        15, False, INK)

    # 2.1 cloud pipeline
    s = slide_bg(prs)
    header(s, "02  云侧联调", "链路如何组织",
           "在 Mac 上完成摄像头、麦克风、决策与 MoMask 的贯通；视频帧不上云")
    cols = [
        ("视觉", "本地",
         "MediaPipe FaceMesh（468 点）\n几何特征 + 逻辑回归\n四类：中性 / 高兴 / 不快 / 惊讶"),
        ("听觉", "本地采集 + 云端转写",
         "麦克风录音 3.5–4 秒\n云端 ASR：gpt-4o-transcribe\n当前不做语音情绪分类"),
        ("决策", "云端 LLM",
         "gpt-4o-mini（OpenAI 兼容中转）\n输入为结构化感知 JSON\n输出英文 action_prompt"),
    ]
    for i, (t, sub, body) in enumerate(cols):
        x = Inches(0.5 + i * 4.2)
        card(s, x, Inches(1.55), Inches(3.95), Inches(5.25))
        rect(s, x, Inches(1.55), Inches(3.95), Inches(0.62), NAVY)
        txt(s, x, Inches(1.65), Inches(3.95), Inches(0.42), t, 18, True, WHITE, PP_ALIGN.CENTER)
        txt(s, x + Inches(0.2), Inches(2.35), Inches(3.55), Inches(0.4), sub, 13, True, TEAL)
        txt(s, x + Inches(0.2), Inches(2.85), Inches(3.55), Inches(3.5), body, 15, False, INK)

    # 2.2 latency
    s = slide_bg(prs)
    header(s, "02  云侧联调", "时延拆解",
           "Mac 实机一轮约 15–17 秒；其中视觉约占 10 秒。云端只贡献 ASR 与 Decide。")
    rows = [
        ["环节", "实测", "处理方式", "说明"],
        ["视觉", "9.9–10.6 s", "本机 FaceMesh", "每轮重新开摄像头；标定等待上限 8 秒；正式采样 12 帧"],
        ["录音", "3.5–6 s", "与视觉并行", "被视觉墙钟覆盖，不额外增加等待"],
        ["ASR", "2.2–4.8 s", "云端转写", "whisper-1 不可用，改用 gpt-4o-transcribe"],
        ["Decide", "2.2–2.8 s", "gpt-4o-mini", "根据情绪与转写生成动作描述"],
        ["MoMask（实机）", "未执行", "Arbiter 降级", "弱表情 + 低置信度，本轮只走预置动作"],
        ["MoMask（mock 感知）", "8.88 s", "Mac CPU 子进程", "含模型冷启动，不可与端侧 2.3 s 直接比较"],
    ]
    tbl = s.shapes.add_table(len(rows), 4, Inches(0.45), Inches(1.5), Inches(12.4), Inches(4.55)).table
    widths = [Inches(2.5), Inches(2.0), Inches(2.6), Inches(5.3)]
    for i, w in enumerate(widths):
        tbl.columns[i].width = int(w)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT
            run = p.add_run()
            fnt(run, str(val), 12, r == 0, WHITE if r == 0 else INK)
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if r == 0 else (SOFT if r % 2 else WHITE)
    txt(s, Inches(0.55), Inches(6.2), Inches(12.2), Inches(0.85),
        "视觉耗时的主要来源是标定，不是 12 帧分类本身。合计帧量约为 warmup 15 + 标定 20 + 采样 12。",
        13, False, MUTED)

    # 2.3 8.9 vs 2.3 and 4GB
    s = slide_bg(prs)
    header(s, "02  云侧联调", "两个容易混淆的数字，以及为何尚未完整上板")
    card(s, Inches(0.5), Inches(1.5), Inches(6.05), Inches(5.3))
    txt(s, Inches(0.75), Inches(1.7), Inches(5.55), Inches(0.4), "2.3 s 与 8.9 s", 18, True, NAVY)
    txt(s, Inches(0.75), Inches(2.25), Inches(5.55), Inches(4.2),
        "端侧 2.3 秒：模型常驻内存后的单条生成时间，是正式基准。\n\n"
        "Mac 8.9 秒：每次以子进程重新启动 gen_t2m.py，把加载 CLIP 与四份权重算进同一笔时间。\n\n"
        "树莓派上仅加载就需要约 10 秒。两者测量口径不同，不能解释为模型变慢。",
        15, False, INK)
    card(s, Inches(6.8), Inches(1.5), Inches(6.05), Inches(5.3))
    txt(s, Inches(7.05), Inches(1.7), Inches(5.55), Inches(0.4), "完整链路仍留在电脑侧", 18, True, NAVY)
    txt(s, Inches(7.05), Inches(2.25), Inches(5.55), Inches(4.2),
        "测试机为 4GB 树莓派。MoMask 推理可以运行，但端侧决策模型（Qwen2.5-1.5B）需要本机编译，多核编译峰值会占满内存并导致死机。\n\n"
        "因此：视听采集、云端决策仍在 Mac 验证；板上目前打通的是 MoMask 与规则版 A∥B 空跑。\n\n"
        "16GB 真机上将明显缓解该限制。",
        15, False, INK)

    # 3 AB
    s = slide_bg(prs)
    header(s, "03  工程方案", "A∥B 双轨：先保证可交互，再补细动作",
           "默认模式 A_parallel_B。失败时本轮仅保留 A，不撤销已执行动作。")
    card(s, Inches(0.5), Inches(1.5), Inches(6.05), Inches(5.3))
    rect(s, Inches(0.5), Inches(1.5), Inches(6.05), Inches(0.62), TEAL)
    txt(s, Inches(0.5), Inches(1.6), Inches(6.05), Inches(0.42), "A  ·  预置动作", 18, True, WHITE, PP_ALIGN.CENTER)
    txt(s, Inches(0.8), Inches(2.4), Inches(5.45), Inches(4.0),
        "根据粗情绪与强度选择 ActionGroup（招手、鞠躬等）。\n\n"
        "启动快，表达有限，用于兜底和即时反馈。\n\n"
        "置信度过低、无有效语音、或后续模块失败时，本轮只走 A。",
        15, False, INK)
    card(s, Inches(6.8), Inches(1.5), Inches(6.05), Inches(5.3))
    rect(s, Inches(6.8), Inches(1.5), Inches(6.05), Inches(0.62), NAVY)
    txt(s, Inches(6.8), Inches(1.6), Inches(6.05), Inches(0.42), "B  ·  MoMask", 18, True, WHITE, PP_ALIGN.CENTER)
    txt(s, Inches(7.1), Inches(2.4), Inches(5.45), Inches(4.0),
        "由决策模块写出英文动作描述，再生成连续关节。\n\n"
        "表达更细，时延以热启动约 2.3 秒为基准。\n\n"
        "B 允许滞后到达：A 先维持交互，B 完成后替换为更完整的动作。",
        15, False, INK)

    # 3.2 why it helps
    s = slide_bg(prs)
    header(s, "03  工程方案", "对时延的实际含义")
    items = [
        ("单轨串行", "标定、云端推理与动作生成依次等待，用户面对的是十余秒无响应。"),
        ("A∥B 并行", "表情稳定后即可执行预置动作；MoMask 在后台生成，不阻塞第一反应。"),
        ("配套改造", "摄像头保持开启、标定只做一次；MoMask 改为常驻进程，避免每次冷启动。"),
        ("预期体感", "交互立即有动作；细粒度关节约在两秒量级补齐。云往返只影响 B，不影响 A。"),
    ]
    for i, (t, d) in enumerate(items):
        y = Inches(1.5 + i * 1.28)
        card(s, Inches(0.55), y, Inches(12.2), Inches(1.12))
        txt(s, Inches(0.85), y + Inches(0.16), Inches(11.6), Inches(0.35), t, 17, True, TEAL)
        txt(s, Inches(0.85), y + Inches(0.55), Inches(11.6), Inches(0.42), d, 15, False, INK)

    # 4 next
    s = slide_bg(prs)
    header(s, "04  下一步", "按顺序推进，不并行铺开")
    steps = [
        ("1", "视听觉",
         "提高速度与准确度",
         "摄像头常开，缩短标定。\n稳定多帧投票。\n端侧 ASR 替换云端转写。"),
        ("2", "端侧部署",
         "完整上板并实测",
         "完成 Qwen 1.5B 安装与决策冒烟。\n4GB 上与 MoMask 错峰运行。\n16GB 真机上验证同机链路。"),
        ("3", "记忆",
         "主链路稳定后再做",
         "用于连续交互中的情绪与意图延续。\n现阶段接入会干扰时延与正确性验收。"),
    ]
    for i, (n, k, t, d) in enumerate(steps):
        x = Inches(0.5 + i * 4.2)
        card(s, x, Inches(1.5), Inches(3.95), Inches(5.25))
        rect(s, x, Inches(1.5), Inches(3.95), Inches(1.35), NAVY)
        txt(s, x, Inches(1.58), Inches(3.95), Inches(0.4), n, 16, True, TEAL, PP_ALIGN.CENTER)
        txt(s, x, Inches(1.95), Inches(3.95), Inches(0.35), k, 14, False, RGBColor(0xA8, 0xC4, 0xC2), PP_ALIGN.CENTER)
        txt(s, x, Inches(2.28), Inches(3.95), Inches(0.42), t, 16, True, WHITE, PP_ALIGN.CENTER)
        txt(s, x + Inches(0.25), Inches(3.15), Inches(3.45), Inches(3.2), d, 15, False, INK)

    # takeaway
    s = slide_bg(prs)
    header(s, "小结", "三点共识")
    lines = [
        "端侧 MoMask 基准为热启动约 2.3 秒/条，数值稳定，且不同情绪对应不同运动幅度。",
        "云侧十余秒主要来自每轮视觉标定与 MoMask 冷启动；ASR 与决策合计约 5–7 秒。完整链路受 4GB 内存限制，尚未全部部署到板上。",
        "落地采用 A∥B：预置动作保证响应，MoMask 补细动作。下一步先优化视听，再完整上板实测，最后考虑记忆。",
    ]
    for i, t in enumerate(lines):
        y = Inches(1.55 + i * 1.65)
        card(s, Inches(0.55), y, Inches(12.2), Inches(1.45))
        rect(s, Inches(0.55), y, Inches(0.12), Inches(1.45), TEAL)
        txt(s, Inches(0.95), y + Inches(0.4), Inches(11.5), Inches(0.7), t, 16, False, INK)

    # thanks
    s = slide_bg(prs)
    rect(s, 0, 0, W, H, NAVY)
    rect(s, 0, 0, Inches(0.14), H, TEAL)
    txt(s, Inches(0.7), Inches(2.7), Inches(12), Inches(0.9), "谢谢。欢迎讨论。", 32, True, WHITE)
    txt(s, Inches(0.7), Inches(3.8), Inches(12), Inches(0.8),
        "数据来源：树莓派 100 条基准、Mac 联调 timing.csv。",
        15, False, RGBColor(0xC5, 0xD2, 0xD8))

    prs.save(str(OUT))
    print("wrote", OUT)


if __name__ == "__main__":
    build()
