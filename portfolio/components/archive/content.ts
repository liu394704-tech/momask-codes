export type Reason = {
  id: string;
  title: string;
  shortTitle: string;
  body: string;
  note: string;
  tag: string;
  author?: string;
  source?: string;
};

// 履历以《AI产品-刘淑雯》简历为准，不补充 MoMask 仓库中的实现细节。
export const templateProfile = {
  brand: 'SHUWEN',
  name: '刘淑雯',
  role: 'AI 产品',
  discipline: 'AI PRODUCT',
  disciplines: 'AI 产品 · 智能硬件交互',
  pageTitle: '刘淑雯 · 作品集',
  website: 'https://liu394704-tech.github.io/shuwen.github.io/',
  location: '中国',
  introduction: '把用户洞察做成可以上线的 AI 产品。',
  biography: '英语可作为工作语言。曾在蚂蚁集团参与支付宝 AI 版资产页，在韶音定义可穿戴健康功能，在面壁智能做智能座舱的端侧大模型产品。正在研究怎样用多模态大模型生成有情感、也受物理约束的机器人动作。',
};

export const contactEmail = 'liu394704@gmail.com';
export const contactPhone = '+86 198-397-12205';
export const contactGithub = 'https://github.com/liu394704-tech';
export const otherPortfolioUrl = templateProfile.website;

// 开场保留模板的屏幕循环；个人内容写在档案与简历里。
export const introMedia: { src: string | null; poster: string | null } = {
  src: `${import.meta.env.BASE_URL}assets/template/intro-loop.mp4`,
  poster: `${import.meta.env.BASE_URL}assets/template/intro-loop-poster.jpg`,
};

export const reasons: Reason[] = [
  {
    id: '01',
    title: '语义和动作之间还有一道缝',
    shortTitle: '语义–动作',
    body: '服务机器人常常听得懂，身体却很僵。我在研究怎样用多模态大模型生成有情感表现力、同时受物理约束的非语言动作。',
    note: '2025.10 至今。',
    tag: '研究',
  },
  {
    id: '02',
    title: '心率什么时候不该开口',
    shortTitle: '场景边界',
    body: '在韶音围绕心率做了 20+ 次访谈和佩戴测试，把运动、通勤、静息里的 AI 健康与情绪功能分开。有些探索性需求，后来收成了可实现的方案。',
    note: '先划边界。',
    tag: '韶音',
  },
  {
    id: '03',
    title: '座舱里的端侧模型',
    shortTitle: '端侧',
    body: '在面壁智能写过登车问候和个性化记忆的需求。情绪识别、动作感知和多模态对话，都要先评估轻量化大模型能不能在车上成立，再写 PRD。',
    note: '先看边界，再写文档。',
    tag: '座舱',
  },
  {
    id: '04',
    title: '人管钱，也要管得住隐私',
    shortTitle: '阿宝',
    body: '在蚂蚁参与支付宝 AI 版「阿宝」资产页。围绕「人管钱」，我负责隐私保护、AI 支付接入和财富黑卡皮肤等方案，从 PRD、Demo 到验收。',
    note: '2026.6 发布。',
    tag: '蚂蚁',
  },
  {
    id: '05',
    title: '看板和二十次访谈',
    shortTitle: '用研',
    body: '资产页用看板看 DAU、渗透率和转化漏斗，也看多轮对话里用户真正在问什么。虚拟访谈加真人访谈做了 20+ 人次，用研报告用来排优先级。',
    note: '数字要对着人。',
    tag: '数据',
  },
  {
    id: '06',
    title: '如果你也在做能上线的 AI',
    shortTitle: '可以说你好',
    body: '邮件和电话都在简历上。我习惯从需求梳理走到 Demo，也愿意把用研、数据和跨团队推进放在同一条链路里。',
    note: '从一封信开始。',
    tag: '联系',
  },
  {
    id: '07',
    title: '动作要先过物理这一关',
    shortTitle: '安全层',
    body: '研究方案里，动作生成之后还要过一层安全过滤：用逆运动学挡住穿模和关节超限，让结果落在机器人自己的物理约束里。这一步计划在仿真环境中验证。',
    note: '先仿真，再谈上身。',
    tag: '约束',
  },
  {
    id: '08',
    title: '评测集要比演示多一轮',
    shortTitle: '评测',
    body: '座舱项目里，我参与搭过多轮对话和复杂场景的评测基准，也记录轻量化模型上车时的效能和性能。WAIC 和上海车展的展示准备，是这条链路对外的一次检验。',
    note: '演示之前先有评测。',
    tag: '评测',
  },
  {
    id: '09',
    title: '洞察、数据和一个能交出去的方案',
    shortTitle: '怎么做',
    body: '我习惯把用户洞察、数据发现和产品目标放在一起，收成 PRD 或 Demo。AI coding 用来搭原型、看数据和做自动化，需求到上线可以自己走完一圈。',
    note: '自己能闭环。',
    tag: '方法',
  },
];

export type PortfolioWork = {
  id: string;
  category: 'research' | 'product';
  title: string;
  caption: string;
  image: string;
  thumbnail: string;
  alt: string;
  width: number;
  height: number;
};

export const workCategories: Array<{
  id: 'all' | PortfolioWork['category'];
  label: string;
}> = [
  { id: 'all', label: '全部' },
  { id: 'research', label: '研究' },
  { id: 'product', label: '产品' },
];

const workAsset = (filename: string) => `${import.meta.env.BASE_URL}assets/template/${filename}`;

export const portfolioWorks: PortfolioWork[] = [
  {
    id: 'affective-motion',
    category: 'research',
    title: '情感动作',
    caption: '多模态大模型 · 2025.10 至今',
    image: workAsset('work-research.svg'),
    thumbnail: workAsset('work-research.svg'),
    alt: '黑白海报：基于多模态大模型的机器人情感动作生成研究，包含指令、动作生成和物理安全层。',
    width: 1200,
    height: 900,
  },
  {
    id: 'abao',
    category: 'product',
    title: '阿宝资产页',
    caption: '支付宝 AI 版 · 2026.6',
    image: workAsset('work-abao.svg'),
    thumbnail: workAsset('work-abao.svg'),
    alt: '黑白海报：支付宝 AI 版阿宝的资产页，隐私保护、AI 支付与财富黑卡皮肤。',
    width: 1200,
    height: 900,
  },
];

export type ResumeEntry = {
  title: string;
  subtitle?: string;
  period: string;
  summary?: string;
  bullets?: string[];
};

export const templateResume: {
  experience: ResumeEntry[];
  projects: ResumeEntry[];
  education: ResumeEntry[];
  awards: string[];
  practice: string;
  tools: string[];
  languages: string[];
} = {
  experience: [
    {
      title: '蚂蚁科技集团股份有限公司（杭州）',
      subtitle: 'AI 产品工程师',
      period: '2026.05 — 2026.09',
      bullets: [
        '资产功能模块建设：参与支付宝 AI 版「阿宝」（2026.6 发布）资产页的产品设计，围绕「人管钱」，负责隐私保护、AI 支付接入、财富黑卡皮肤接入等核心功能的方案与推进，覆盖 PRD、产品 Demo、跨团队协调和功能验收。',
        '数据分析与看板搭建：借助 AI coding 工具搭建资产模块核心数据看板，持续追踪各功能 DAU、渗透率及用户转化漏斗；针对多轮对话 query 分析用户关注的优化方向，并结合用研思考产品方案。',
        '用户研究与边界探索：探索 AI 访谈的落地方式，执行虚拟与真人用户访谈 20+ 人次，输出用研报告，驱动产品优先级决策。',
        '成果：产出多份产品 Demo 和 PRD，推动资产模块核心功能上线，支撑 AI 版资产页 V1.0 按时交付。资产页功能日均覆盖用户达 300 万；AI 记账、AI 支付等 AI 原生功能累计触发对话 2 万轮。',
      ],
    },
    {
      title: '韶音科技有限公司（深圳）',
      subtitle: '产品创新部 · AI 产品实习生',
      period: '2025.11 — 2026.03',
      bullets: [
        '主导 AI 健康功能前期定义：围绕心率等生理信号开展用户访谈与佩戴测试（20+ 样本），梳理运动、通勤、静息等场景，明确 AI 健康与情绪类功能的适用边界。',
        '系统拆解竞品方案：分析 10+ 款具备健康或情绪监测能力的智能耳机，从传感器配置、算法能力与交互路径输出结构化竞品报告，识别可落地能力与实际体验差距。',
        '参与构建体验评估框架：结合主观反馈（问卷、访谈）与客观生理数据（心率区间等），在多轮测试中对比不同功能方案。',
        '支持功能迭代决策：基于测试结果与用户反馈，推动部分功能定义从探索性需求收敛为可实现方案，提升需求评审效率。',
      ],
    },
    {
      title: '面壁智能科技有限公司（北京）',
      subtitle: '智能座舱 · AI 产品实习生',
      period: '2025.02 — 2025.07',
      bullets: [
        '端侧大模型需求与场景落地：剖析登车问候、个性化记忆等座舱场景的 AI 软硬协同需求。针对情绪识别、动作感知与多模态对话，评估端侧轻量化大模型的技术边界与可行性，并完成 PRD。',
        '多模态感知与 Prompt 工程：设计并优化多模态感知交互链路，主导座舱场景的 Prompt 模板工程，按用户意图与座舱环境做结构化整理与调优。',
        '构建数据飞轮与模型评测闭环：负责大模型上车的效能验证、轻量化功能测试及性能指标记录；与算法团队协作支持语义泛化训练，搭建多轮对话与复杂场景的评测基准。',
        '成果：为轻量化大模型上车提供数据与优化建议；协助完成面向 WAIC、第二十一届上海国际汽车工业展览会等活动的展示与技术准备。',
      ],
    },
  ],
  projects: [
    {
      title: '基于多模态大模型（MLLM）的机器人情感动作生成系统',
      subtitle: '研究方向',
      period: '2025.10 — 至今',
      summary: '面向具身情感智能。针对服务机器人「语义理解强，肢体表达僵硬」的语义–动作鸿沟，探索用多模态大模型驱动机器人生成符合物理约束、具有情感表现力的非语言交互动作。',
      bullets: [
        '生成式模型架构：拟采用 Latent Diffusion Model（MLD）作为动作生成基座，结合 CLIP 文本编码器，根据自然语言指令生成多样化的人体骨骼动作。',
        '物理安全性：针对穿模与关节超限，计划在 Isaac Gym 仿真中引入基于逆运动学（IK）的安全过滤层，使结果符合机器人 URDF 的物理约束。',
        '系统集成：规划用 ROS2 接口打通生成模型与底层控制，目标是在仿真环境中形成从指令输入到动作输出的端到端闭环。',
      ],
    },
  ],
  education: [
    {
      title: '香港科技大学（广州）',
      subtitle: '人工智能 · 硕士 · QS33',
      period: '2025.09 — 2027.07',
      summary: '研究方向：AI for Hardware、智能嵌入式系统、LLM 边端部署、智能硬件人机交互。',
      bullets: [
        '主修课程：嵌入式系统设计、传感系统与智能感知、物联网信息架构、边缘计算与应用、高性能计算、强化学习。',
      ],
    },
    {
      title: '南京理工大学',
      subtitle: '计算机科学与工程 · 本科 · 211 / 双一流',
      period: '2020.09 — 2024.06',
      summary: 'GPA 3.53/4.0，排名 31/290。',
      bullets: [
        '主修课程：计算机操作系统、高等数学（98）、数据库管理（95）、机器学习（93）、数据结构（90）、算法理论基础（90）、人机交互设计（90）。',
      ],
    },
  ],
  awards: [],
  practice: '英语可作为工作语言。编程上熟练使用 Python、C++ 和 MySQL，理解前后端交互，能独立完成轻量级 Demo 与功能验证。日常用 AI coding 工具搭原型、做数据分析和自动化处理，从需求梳理到 Demo 上线可以自己闭环。产品上习惯把用户洞察、数据发现和产品目标收成可落地方案，熟悉 PRD 与跨团队协作。',
  tools: [
    '产品：用户洞察、PRD、跨团队推进、从需求到上线',
    '研究与数据：访谈、看板、DAU / 渗透率 / 转化漏斗、评测基准',
    '工程：Python、C++、MySQL，可独立完成轻量 Demo',
    'AI 工具：用 AI coding 做原型、数据分析与自动化',
  ],
  languages: [
    '英语：可作为工作语言，雅思 7.0',
  ],
};
