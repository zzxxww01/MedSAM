# 文档导航

本目录包含论文的全部核心知识。

---

## 📚 必读文档（按顺序）

### 0. **TODO.md** ← 待办清单（优先查看）
**内容**：8 幅图片制作清单、论文打磨清单、答辩准备清单、时间规划。

**适合**：了解还需要做什么、如何获取图片数据。

**阅读时间**：10 分钟

---

### 1. **THESIS_KNOWLEDGE_BASE.md** ← 从这里开始
**内容**：论文完整知识库，涵盖研究全景、三核方法、实验结果、答辩Q&A、待办图片清单。

**适合**：快速全面掌握整篇论文。

**阅读时间**：30-40 分钟

---

### 2. **EXPERIMENT_LOG.md** ← 数据查询
**内容**：全部 11 组实验的原始数据表（DSC/HD95/ASD）、实验卡片、差分证据表。

**适合**：论文写作时核对数值、查找 JSON 路径。

**阅读时间**：10 分钟

---

## 🎯 快速定位

| 你想了解... | 去哪里 |
|------------|--------|
| **还需要做什么** | **TODO.md** |
| 整体研究思路 | THESIS_KNOWLEDGE_BASE.md § 2 研究全景 |
| Balance Loss 原理 | THESIS_KNOWLEDGE_BASE.md § 4 |
| LoRA 为什么失败 | THESIS_KNOWLEDGE_BASE.md § 5 |
| LG-Adapter 设计 | THESIS_KNOWLEDGE_BASE.md § 6 |
| 全部实验数据 | EXPERIMENT_LOG.md |
| 答辩常见问题 | THESIS_KNOWLEDGE_BASE.md § 10 |
| 待制作图片清单 | THESIS_KNOWLEDGE_BASE.md § 12 |
| 论文章节映射 | THESIS_KNOWLEDGE_BASE.md § 9 |

---

## 📂 其他重要文件

| 路径 | 内容 |
|------|------|
| `../thesis-medsam/thesis.pdf` | 编译后的论文 PDF |
| `../thesis-medsam/pages/chapter*.tex` | 各章节 LaTeX 源文件 |
| `../thesis-medsam/ref/references.bib` | 参考文献（30篇） |
| `../models/medsam_fss.py` | LoRA + LG-Adapter 代码 |
| `../losses/balance_loss.py` | Balance Loss 代码 |
| `../work_dir/eval_metrics/*_summary.json` | 评估结果 JSON |

---

## ✅ 当前状态

- ✅ 全部 11 组实验完成
- ✅ 代码主体已定型（如需复现实验，优先以 `work_dir/eval_metrics/*_summary.json` 和论文正文当前版本为准）
- ✅ 论文主体已完成并完成一轮口径核对
- ✅ 参考文献扩充至 30 篇
- ⏳ **待制作 8 幅图片**（详见 TODO.md）
- ⏳ 论文细节打磨
- ⏳ 答辩 PPT 制作

---

**最后更新**：2026-03-03
