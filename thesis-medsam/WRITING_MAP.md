# 文档到 LaTeX 章节映射

> 目的：将 `docs/` 中的规划与实验事实快速落地到论文 `.tex` 文件。  
> 更新时间：2026-02-14

---

## 1. 主导航

1. 先看：`docs/THESIS_MASTER_GUIDE.md`
2. 再看：`docs/THESIS_TECHNICAL_REPORT.md`
3. 回填事实：`docs/EXPERIMENT_LOG.md`
4. 路线决策：`docs/FULL_EXPERIMENT_PLAN.md`

---

## 2. 章节映射

| LaTeX 文件 | 对应文档来源 | 说明 |
|---|---|---|
| `pages/chapter1.tex` | `docs/THESIS_PLAN.md`、`docs/THESIS_TECHNICAL_REPORT.md` | 绪论与创新点 |
| `pages/chapter2.tex` | `docs/THESIS_TECHNICAL_REPORT.md` | 理论基础与关键公式 |
| `pages/chapter3.tex` | `docs/EXPERIMENT_LOG.md`、`docs/THESIS_TECHNICAL_REPORT.md` | Balance 损失方法与 A3R3 结果 |
| `pages/chapter4.tex` | `docs/THESIS_TECHNICAL_REPORT.md`、`docs/FULL_EXPERIMENT_PLAN.md` | 架构创新设计 (LoRA 与 MSL-Adapter) |
| `pages/chapter5.tex` | `docs/FULL_EXPERIMENT_PLAN.md`、`docs/WEEKLY_TASKS.md` | 综合实验组织、消融分析与 SOTA 对比 |
| `pages/chapter6.tex` | `docs/THESIS_MASTER_GUIDE.md` | 总结与展望 |

---

## 3. 数据回填规则

1. 表格数值必须来自 `work_dir/eval_metrics/*_summary.json`。
2. 未完成实验统一写“待回填/进行中”，禁止写确定结论。
3. 每次更新实验结果后，同步修改：
   1. `docs/EXPERIMENT_LOG.md`
   2. `pages/chapter3.tex`
   3. `pages/chapter5.tex`

---

## 4. 编译入口

主文件：`thesis.tex`  
参考文献库：`ref/references.bib`
