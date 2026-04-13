# Citation Audit Report for `thesis-medsam`

检查时间：2026-04-12  
检查范围：`thesis-medsam/ref/references.bib` + 全部 `*.tex` 中的 `\cite...{}`  
核查方式：本地 BibTeX 静态检查 + Crossref/公开论文页面在线核对

## 1. 总体结论

- `references.bib` 共 `71` 条文献。
- 正文实际引用了 `60` 条，未发现“正文引用 key 在 `.bib` 中不存在”的错误。
- 有 `11` 条参考文献目前未被正文引用。
- BibTeX 语法层面未发现明显缺失必填字段的问题。
- 但存在若干“文献信息真实性/版本选择/年份风格”层面的风险，主要集中在：
  - 预印本与正式发表版本混用
  - LNCS/会议年份 与 DOI 对应出版年份不一致
  - 少数条目可疑，可能是同一工作被重复或错误命名

## 2. 需要优先处理的条目

### 2.1 高风险：`zhang2024samed`

当前条目：

- key: `zhang2024samed`
- title: `SAMed: A General and Efficient Medical Image Segmentation Framework Based on Segment Anything Model`
- authors: `Zhang, Kaidong and Liu, Dong`
- venue: `MICCAI`
- year: `2024`

核查结论：

- 我没有检索到这个题名对应的稳定正式发表记录。
- 同一作者更明确、可检索到的版本是 arXiv 技术报告：
  - `Customized Segment Anything Model for Medical Image Segmentation`
  - arXiv: `2304.13785`
- 该 arXiv 论文正文中明确写到提出的方法名是 `SAMed`。

建议：

- 如果你想引用的是这篇工作本身，优先保留 `zhang2023customized`，并删除或重写 `zhang2024samed`。
- 如果你确实掌握该工作后续正式发表版本，需要补充可核验的 DOI / proceedings 页面 / OpenReview 页面，否则该条目属于高风险引用。

### 2.2 高风险：`wei2024imedsam`

当前条目：

- key: `wei2024imedsam`
- venue: `ECCV`
- year: `2024`
- note: `arXiv:2311.17081`
- 缺 DOI

在线核对结果：

- 公开可检索到的 DOI：`10.1007/978-3-031-72684-2_6`

建议：

- 若按正式 proceedings 版本引用，补上 DOI，并统一 `booktitle/pages/year`。
- 这里有一个风格选择问题：
  - 按会议届次写：`ECCV 2024`
  - 按 Springer 出版年写：可能显示为 `2025`
- 全文必须统一一种规则，不要同类条目混写。

### 2.3 高风险：`cao2022swinunet`

当前条目有 DOI：`10.1007/978-3-031-25066-8_9`

核查结论：

- DOI 对应的 Springer 记录出版年是 `2023`。
- 你当前写的是 `2022`，这更像“会议年份/工作年份”，不是 DOI 对应的出版年份。

建议：

- 二选一，但全文统一：
  - 保持 `ECCV Workshops 2022` 风格，则尽量不要混用 Springer 出版年逻辑。
  - 若按 DOI 对应正式出版版本，则该条年应改成 `2023`。

### 2.4 中高风险：`wong2024scribbleprompt`

当前条目：

- DOI 已存在：`10.1007/978-3-031-73661-2_12`
- year: `2024`

核查结论：

- DOI 对应的 Springer 记录出版年显示为 `2025`。
- 与 `wei2024imedsam`、`cao2022swinunet` 属于同类问题。

建议：

- 不一定是“错引”，但这是明显的年份风格不统一风险。
- 建议把所有 LNCS / Springer proceedings 条目统一到一个规则。

### 2.5 版本选择问题：`kervadec2019boundary`

当前条目：

- `Boundary Loss for Highly Unbalanced Segmentation`
- `MIDL 2019`

核查结论：

- 该工作后续存在 Journal 版本：
  - `Boundary loss for highly unbalanced segmentation`
  - Medical Image Analysis, `2021`
  - DOI: `10.1016/j.media.2020.101851`

建议：

- 如果正文讨论的是最初提出版本，当前 `MIDL 2019` 可以保留。
- 如果你希望引用更正式、更稳定的版本，建议改为 `Medical Image Analysis 2021` 版本。
- 关键是不要把 `MIDL 2019` 的条目和 `2021` 的 DOI 混在同一个条目里。

## 3. 可补充但不一定算错误的条目

### 3.1 可补 DOI 的条目

以下条目本地 BibTeX 中缺 DOI，但在线可找到稳定 proceedings DOI，可考虑补齐：

- `chen2022adaptformer`
  - 可能 DOI：`10.52202/068431-1212`
- `ji2022amos`
  - 可能 DOI：`10.52202/068431-2661`

说明：

- 这两条不是“必须修”，但如果你的参考文献习惯是“正式会议论文尽量带 DOI”，建议统一补齐。

### 3.2 合理的预印本条目

以下条目当前以 arXiv / preprint 形式保存，未检出明确错误，保留通常是可以接受的：

- `howard2017mobilenets`
- `chen2017deeplabv3`
- `cheng2023sammed2d`
- `li2021localvit`
- `zhu2024medsam2`
- `shaharabany2023autosam`
- `bommasani2021foundation`
- `paranjape2024adaptivesam`

说明：

- 这些条目的核心问题不是“错”，而是是否需要优先换成正式发表版本。
- 如果学校/学院要求优先引用正式出版版本，建议逐条替换；如果 thesis 允许引用 preprint，则当前做法基本可接受。

## 4. 未被正文使用的条目

以下 `11` 条目前在正文中没有 `\cite` 到：

- `bommasani2021foundation`
- `butoi2023universeg`
- `demsar2006statistical`
- `moor2023foundation`
- `paranjape2024adaptivesam`
- `roy2023mednext`
- `shaharabany2023autosam`
- `sudre2017generalised`
- `taghanaki2019combo`
- `wong2024scribbleprompt`
- `zhang2023customized`

建议：

- 如果这些文献不打算在终稿中出现，可以从 `.bib` 删除，减少答辩时被追问“为什么放进去但正文没引用”。
- 如果准备在后续章节或 related work 中使用，则保留即可。

## 5. 建议的修改优先级

### 第一优先级

- 处理 `zhang2024samed`
- 统一 `wei2024imedsam` / `cao2022swinunet` / `wong2024scribbleprompt` 的年份规则
- 决定 `kervadec2019boundary` 用 MIDL 2019 还是 MedIA 2021 版本

### 第二优先级

- 给 `chen2022adaptformer` 和 `ji2022amos` 补 DOI
- 统一所有 preprint 的写法：
  - 是否保留 `arXiv preprint arXiv:...`
  - 是否统一 `note = {[访问日期: ...]}` 的格式

### 第三优先级

- 清理未引用条目

## 6. 附件

- 机器核查明细：`thesis-medsam/ref/citation_check_results.json`

## 7. 本次核查使用的公开来源

- Crossref API: https://api.crossref.org/
- arXiv `Customized Segment Anything Model for Medical Image Segmentation`: https://arxiv.org/abs/2304.13785
- arXiv `AutoSAM: Adapting SAM to Medical Images by Overloading the Prompt Encoder`: https://arxiv.org/abs/2306.06370
- arXiv `SAM-Med2D`: https://arxiv.org/abs/2308.16184
- ResearchGate metadata page for `I-MedSAM`: https://www.researchgate.net/publication/385505474_I-MedSAM_Implicit_Medical_Image_Segmentation_with_Segment_Anything
