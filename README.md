# Inferring-Intelligence

论文往往给出了方法与公式，却没有解释：它为什么会被设计成这样？

《推断智能》尝试从问题背景、数学结构、统计学直觉与研究经验出发，反推公式&算法背后的设计动机，重建从问题定义、核心假设、损失函数到算法的逻辑链条。

栏目围绕 AI 领域算法展开，不止回答“公式怎么推”，更追问“为什么偏偏是这个公式”。这里既有严格的数学推导，也有基于证据与经验的合理推断。

本仓库以整理后的手稿形式，开放课程笔记，并逐步补充录课提纲、课件和视频链接。

## 开始阅读

| 编号 | 手稿 | 主题 | 课程 |
| --- | --- | --- | --- |
| 01 | [DPO：从偏好数据到策略优化](manuscripts/01-dpo/README.md) | 偏好学习、KL 正则、最大似然 | [录课提纲](courses/01-dpo/README.md) · 尚未录制 |

## 仓库结构

```text
manuscripts/                 发布版手稿，按编号与主题归档
  01-dpo/README.md           第一篇：DPO
courses/                    与手稿对应的录课提纲、课件说明和视频链接
  01-dpo/README.md           DPO 录课提纲
templates/manuscript.md      新手稿模板
CONTRIBUTING.md              内容整理与纠错约定
ATTRIBUTION.md               转载要求与署名示例
LICENSE                     CC BY 4.0 许可全文
```

一篇手稿一个目录；插图放入该篇的 `assets/` 子目录，需要时再创建。课程沿用对应手稿的编号。视频发布后在课程页补充链接，原始录屏和大型媒体文件不直接提交到仓库。

## 如何阅读与参与

手稿尽量沿着“问题 → 假设 → 推导 → 算法 → 边界”的顺序展开，区分数学结论、建模假设与直觉解释。可以按文章独立阅读，不要求按编号顺序学习。

发现公式、论证或引用问题，欢迎提交 [Issue](https://github.com/anka-afk/Inferring-Intelligence/issues) 或 Pull Request；整理规则见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可与转载

Copyright © 2026 [anka](https://github.com/anka-afk)。除另有标注的第三方材料外，本仓库原创手稿与课程资料采用 [Creative Commons Attribution 4.0 International（CC BY 4.0）](https://creativecommons.org/licenses/by/4.0/) 许可。

**转载必须标明作者与出处，附原文和许可链接；如有修改，需说明。** 该许可允许分享、改编及商业使用。具体署名方式见 [ATTRIBUTION.md](ATTRIBUTION.md)，正式条款见 [LICENSE](LICENSE)。第三方论文、图片及引用内容的权利仍属于各自权利人。
