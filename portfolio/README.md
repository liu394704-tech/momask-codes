# 刘淑雯 · 作品集

黑白纸面档案上的个人作品集和简历。滚动进入展板，点开右下角的简历纸条可以读完整履历。

视觉与交互动效来自 [KAKI 的 Open Archive](https://github.com/kakizzzzz/open-archive-portfolio)（MIT）。本目录替换的是个人资料、履历和两件选品；模板的版权声明保留在 [LICENSE](LICENSE)。

## 简历里有什么

内容以《AI产品-刘淑雯》这份简历为准：

- 教育：香港科技大学（广州）人工智能硕士（QS33），南京理工大学计算机本科（GPA 3.53/4.0，31/290）
- 实习：蚂蚁集团、韶音科技、面壁智能
- 研究方向：基于多模态大模型的机器人情感动作生成
- 个人总结：英语、Python / C++ / MySQL、AI coding 与产品推进

研究方向只保留简历里的方案表述，没有写入 MoMask 仓库中的实现细节。

另一份网页简历：<https://liu394704-tech.github.io/shuwen.github.io/>

## 本地运行

需要 Node.js 22 或更新版本。

```sh
cd portfolio
npm ci
npm run dev
```

发布前：

```sh
npm run build
npm run preview
```

`build` 含 TypeScript 检查，输出在 `portfolio/dist/`。

## 改内容

个人资料、联系方式、履历、纸条文案和作品列表都在 [`components/archive/content.ts`](components/archive/content.ts)。作品图在 [`public/assets/template/`](public/assets/template/)。
