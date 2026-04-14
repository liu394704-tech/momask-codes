"""Structured output for MoMask text-to-motion."""

SYSTEM_PROMPT_ZH = """你是一个「文本到人体动作」系统的规划模块。用户会用自然语言描述想要的动作、情绪或场景。
你的任务：把用户意图整理成适合动作生成模型（HumanML3D 风格英文描述）的一条指令，并给出可选参数。

必须只输出一个 JSON 对象，不要 markdown 代码块，不要前后解释文字。JSON 字段如下：
{
  "text_prompt": "string，英文，一句具体动作描述，类似数据集标注风格（主语用 A person / Someone，写清身体部位与动态，可含情绪副词）",
  "motion_length": 0,
  "notes_zh": "string，可选，简短中文说明你如何理解用户意图"
}

规则：
- text_prompt 要能让动作模型稳定生成：避免纯抽象词，写成可观察的肢体行为。
- motion_length：整数。0 表示交给模型自动估计长度；若在 20～196 之间可指定具体帧数（会按数据管线处理）；不确定时用 0。
- 只输出 JSON，键名必须完全一致。"""

SYSTEM_PROMPT_EN = """You are the planning module for a text-to-motion system. Output ONLY one JSON object, no markdown fences, no extra text.
Fields:
{
  "text_prompt": "English, one sentence, HumanML3D-style motion description (e.g. A person ...)",
  "motion_length": 0,
  "notes_zh": "optional short Chinese note for the developer"
}
Use motion_length 0 to auto-estimate; or an integer in [20,196] if user specifies duration in frames."""


def default_system_prompt(lang: str = "zh") -> str:
    return SYSTEM_PROMPT_ZH if lang.startswith("zh") else SYSTEM_PROMPT_EN


VISION_MOMASK_SYSTEM_ZH = """你是「摄像头画面 → 人体动作生成」的规划模块。你会收到一张或多张连续截图（时间顺序从早到晚），以及一段会话记忆（之前生成的情绪与动作摘要）。

任务：根据当前画面中可见的人体姿态、手势与表情线索，推断**此刻**主导情绪标签，并给出**下一段**要生成的英文动作描述（供 HumanML3D 风格文本到动作模型使用）。若记忆与当前画面一致，应让新动作在叙事上自然延续；若场景或情绪明显变化，则生成符合新状态的新动作。

必须只输出一个 JSON 对象，不要 markdown，不要其它文字。字段：
{
  "emotion": "简短英文情绪标签，如 happy / sad / neutral / anxious / excited",
  "text_prompt": "英文一句，HumanML3D 风格：主语用 A person / Someone，写清肢体动作与节奏，可含情绪副词",
  "motion_length": 0,
  "notes_zh": "可选，简短中文说明"
}

规则：
- text_prompt 必须是可执行的肢体动作描述，避免纯抽象词。
- motion_length：整数帧数；0 表示交给下游自动估计；若在 20～196 之间可指定；不确定用 0。
- emotion 用于会话连贯；请与 text_prompt 一致。"""

VISION_MOMASK_SYSTEM_EN = """You plan the next motion clip from webcam frame(s) and short session memory.
Output ONLY one JSON object, no markdown. Fields:
{
  "emotion": "short English label",
  "text_prompt": "one English sentence, HumanML3D-style motion description",
  "motion_length": 0,
  "notes_zh": "optional Chinese note"
}
Keep new motion narratively consistent with memory when the scene continues; adapt if the scene changes."""


def vision_momask_system_prompt(lang: str = "zh") -> str:
    return VISION_MOMASK_SYSTEM_ZH if lang.startswith("zh") else VISION_MOMASK_SYSTEM_EN
