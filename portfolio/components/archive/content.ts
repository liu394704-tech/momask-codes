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

// 资料来自现有个人主页与作品集，不写入 MoMask 仓库中的实现细节。
export const templateProfile = {
  brand: 'SHUWEN',
  name: '刘淑雯',
  role: '互联网产品经理 / AI 产品经理',
  discipline: 'AI PRODUCT',
  disciplines: 'AI 产品 · 智能硬件交互',
  pageTitle: '刘淑雯 · 作品集',
  website: 'https://liu394704-tech.github.io/shuwen.github.io/',
  location: '中国 · 广东',
  introduction: '把前沿大语言模型与智能硬件结合起来。',
  biography: '拥有计算机科学与人工智能双重学术背景，曾在字节跳动、面壁智能及韶音科技担任核心产品/数据实习生。致力于将前沿大语言模型与智能硬件相结合，创造极具商业价值与用户体验的 AI 产品。',
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
    title: '先让情绪有一个落点',
    shortTitle: '情绪要落地',
    body: '这是一款面向家庭的陪伴机器人，希望它能理解自然语言、识别情绪，再用动作、姿态和声音回应。我负责情绪与身体语言之间的关系。',
    note: '情绪要有身体。',
    tag: '机器人',
  },
  {
    id: '02',
    title: '从访谈里划出场景边界',
    shortTitle: '先问清楚',
    body: '在韶音做可穿戴时，我用访谈和佩戴测试去划 AI 健康功能的边界：心率这类信号什么时候有用，什么时候不该打扰人。',
    note: '功能从场景里长出来。',
    tag: '韶音',
  },
  {
    id: '03',
    title: '座舱里的下一句话',
    shortTitle: '下一句话',
    body: '在面壁智能参与登车问候和个性化记忆的方案与 PRD，并把座舱里的多模态链路写成可复用的 Prompt。评测集要能撑住多轮和复杂场景，而不是只看一句演示。',
    note: '方案要能被测。',
    tag: '座舱',
  },
  {
    id: '04',
    title: '语义要能变成动作',
    shortTitle: '语义到动作',
    body: '嵌入式大语言模型负责听懂上下文。我搭建的是从语义到动作和声音的映射，并和硬件、运动控制一起把语言、动作和传感器数据接上。',
    note: '语义要能被做出来。',
    tag: '语义',
  },
  {
    id: '05',
    title: '数据要能解释体验',
    shortTitle: '看数，也看人',
    body: '抖音电商期间写过千川、飞天的需求，并用看板跟直播效果。后来在可穿戴里把主观反馈和生理特征放在同一套度量里，让算法迭代有产品语言可对。',
    note: '数字要对着人。',
    tag: '度量',
  },
  {
    id: '06',
    title: '如果你也在做硬件里的智能',
    shortTitle: '可以说你好',
    body: '我在找 AI 产品与智能硬件交互相关的实习和合作。邮件、电话，或者现有个人主页，都可以。',
    note: '从一封信开始。',
    tag: '联系',
  },
  {
    id: '07',
    title: '社区里的一针',
    shortTitle: '城市针灸',
    body: '城中村社区服务平台把传感器、数据和可视化收成一套管理界面。我负责前期调研、原型和图像识别相关功能，并在五天开发后完成路演。',
    note: '小介入，先被用起来。',
    tag: '社区',
  },
  {
    id: '08',
    title: '一盏靠触感说话的灯',
    shortTitle: '触感',
    body: 'EI 情感交互灯用六种材质的触摸球传递情绪，灯罩可折叠，容器兼做小夜灯。我用 Figma 做界面和原型，并和同学一起完成建模与渲染。评估问卷里九成以上好评，NCDA 国家级三等奖。',
    note: '情绪也可以被摸到。',
    tag: '灯',
  },
  {
    id: '09',
    title: '还在长的判断',
    shortTitle: '仍在长',
    body: '我在意一个人在什么时刻需要机器出现，以及它该用动作、声音还是触感回应。作品可以更换，这个问题会留下来。',
    note: '保持好奇。',
    tag: '视角',
  },
];

export type PortfolioWork = {
  id: string;
  category: 'robot' | 'community' | 'hardware';
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
  { id: 'robot', label: '机器人' },
  { id: 'community', label: '社区' },
  { id: 'hardware', label: '硬件' },
];

const workAsset = (filename: string) => `${import.meta.env.BASE_URL}assets/template/${filename}`;

export const portfolioWorks: PortfolioWork[] = [
  {
    id: 'robot',
    category: 'robot',
    title: '陪伴机器人',
    caption: '多模态交互与嵌入式大模型',
    image: workAsset('work-robot.svg'),
    thumbnail: workAsset('work-robot.svg'),
    alt: '黑白海报：智能家居陪伴机器人情感交互系统，情绪、动作与声音。',
    width: 1200,
    height: 900,
  },
  {
    id: 'community',
    category: 'community',
    title: '社区针灸',
    caption: '城中村里的一处嵌入',
    image: workAsset('work-community.svg'),
    thumbnail: workAsset('work-community.svg'),
    alt: '黑白海报：基于城市针灸理念的社区嵌入式服务平台。',
    width: 900,
    height: 1200,
  },
  {
    id: 'lamp',
    category: 'hardware',
    title: 'EI 灯',
    caption: '用触感传递情绪',
    image: workAsset('work-lamp.svg'),
    thumbnail: workAsset('work-lamp.svg'),
    alt: '黑白海报：EI 情感交互灯，六种材质的触摸球与可折叠灯罩。',
    width: 1000,
    height: 1400,
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
      title: '韶音科技有限公司（深圳）',
      subtitle: '产品创新部 · AI 产品实习生',
      period: '2025.11 — 2026.03',
      bullets: [
        'AI 可穿戴设备功能定义：主导基于生理监测（如心率）的 AI 功能需求挖掘，设计用户访谈与佩戴测试，测定 AI 健康功能的场景边界。',
        '竞品反向工程与市场调研：深度调研智能耳机市场，输出结构化竞品研报，明确产品技术差异化定位。',
        '数据体验度量体系：构建多维生理数据分析框架，将主观反馈与客观生物特征结合，支持底层算法演进与产品决策。',
      ],
    },
    {
      title: '面壁智能科技有限公司（北京）',
      subtitle: '智能座舱 · AI 产品实习生（智能硬件方向）',
      period: '2025.02 — 2025.07',
      bullets: [
        '参与完成多项智能座舱核心方案（登车问候、个性化记忆）的落地及 PRD 撰写。',
        '设计并优化多模态感知交互链路，主导细化座舱场景的 Prompt 模板工程。',
        '搭建多轮对话与复杂场景的评测基准，为轻量化模型提供高质量评测集。',
      ],
    },
    {
      title: '字节跳动科技有限公司（上海）',
      subtitle: '抖音电商 · 数据产品实习生（流量工具方向）',
      period: '2024.01 — 2024.05',
      bullets: [
        '撰写产品需求文档，协助完成「千川」「飞天」两个流量产品的方案细化。',
        '通过看板跟踪产品效果，协助生服运营团队累计促成百万场直播 10+，月累计直播成交额连续上升。',
      ],
    },
  ],
  projects: [
    {
      title: '智能家居陪伴机器人情感交互系统',
      subtitle: '产品负责人',
      period: '',
      summary: '面向家庭场景的陪伴机器人，希望它具备自然的情感表达和多模态交互。系统把传感器输入、动作控制与大语言模型放在一起，使机器人能够理解自然语言、识别用户情绪，并以动作、姿态与声音回应。',
      bullets: [
        '负责机器人动作与情绪关系的建模设计，构建情绪驱动的姿态、手势与动态表达机制。',
        '负责嵌入式大语言模型的集成与语义理解，搭建从语义到行为的映射，使机器人能按上下文给出动作与声音反馈。',
        '与负责硬件架构和运动控制的组长协作，完成接口对接、模块调试和跨团队协调，让语言、动作与传感器数据流保持一致。',
      ],
    },
    {
      title: '基于城市针灸理念的社区嵌入式服务',
      subtitle: '项目负责人',
      period: '',
      summary: '针对一线城市城中村社区的服务平台，集成传感器、数据库、AI 和可视化界面。通过实时数据与 AI 辅助决策，为社区管理提供支持，并帮助预判潜在事件。',
      bullets: [
        '主导前期社会调研，设计问卷与访谈，分析数据并确定痛点与方案。',
        '负责平台原型和图像识别功能，协调团队完成系统搭建。',
        '主持项目路演并撰写说明书。在五天开发周期内完成设计，最终路演名列前茅。',
      ],
    },
    {
      title: 'EI 情感交互灯',
      subtitle: '产品负责人',
      period: '',
      summary: '一款借助触感传递情感的产品。触摸球有六种材质，在不同情景下代表不同情绪；安放球体的容器同时是小夜灯，灯罩可折叠，并配有对应的界面。',
      bullets: [
        '做前期市场调研，收集用户数据并确定痛点，保证设计贴合需求。',
        '使用 Figma 负责界面与原型，并配合完成 Rhino 建模与 Keyshot 渲染。',
        '撰写产品需求说明书，组织用户评估并持续迭代。问卷收获九成以上好评，获国家级设计比赛 NCDA 三等奖。',
      ],
    },
  ],
  education: [
    {
      title: '香港科技大学（广州）',
      subtitle: '人工智能 · 硕士',
      period: '2025.09 — 2027.06',
      summary: '研究方向：AI for Hardware、智能嵌入式系统、LLM 边端部署、智能硬件人机交互。',
    },
    {
      title: '南京理工大学',
      subtitle: '计算机科学与工程 · 本科 · GPA 3.53/4.0',
      period: '2020.09 — 2024.06',
      summary: '主修机器学习、数据结构、数据建模与分析、算法理论基础、人机交互设计。',
    },
  ],
  awards: [
    '第十三届 MathorCup 高校数学建模挑战赛 · 国家级一等奖 · 组长',
    '第十二届全国大学生数学建模竞赛 · 国家级一等奖 · 组长',
    '第十二届全国大学生高等数学竞赛 · 国家级一等奖',
  ],
  practice: '个人作品集目前对应三件事情：城中村社区的嵌入式服务、智能家居陪伴机器人的情感交互，以及 EI 情感交互灯。它们和实习里的可穿戴、智能座舱、流量工具放在一起看，都是先把人和场景问清楚，再决定产品如何回应。',
  tools: [
    '互联网产品策划 / PRD / 用户体验调研',
    '数据分析：SPSS / MySQL',
    'C++ / Python / Java',
    '大语言模型应用 / Prompt 工程 / 多模态交互',
    '边端部署与轻量化模型 / 场景评测与 Benchmark',
  ],
  languages: [
    '英语：雅思 7.0 / CET-6 600',
  ],
};
