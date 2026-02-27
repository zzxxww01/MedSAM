# MedSAM 论文与实验总控指南（Master Guide）

> 更新时间：2026-02-27  
> 目标：用一份文档串起“论文叙事、实验证据、执行命令、下一步决策”。

---

## 1. 一页全局快照

### 1.1 当前研究主线
- 研究主题：基于 Balance Loss 与 Attention 融合的医学图像分割改进（以 MedSAM 为基座）。
- 当前数据主线：`FLARE22 -> data/npy/CT_Abd`。
- 当前优先级：以 A3R3 作为 Loss 主干，进入 Attention 阶段（EXP-005）。

### 1.2 实验状态（同口径）

| 实验 | 状态 | DSC | HD95 | ASD | 结论 |
|---|---|---:|---:|---:|---|
| A0 Baseline | 已完成 | 0.940741 | 4.830503 | 0.537757 | 基线已建立 |
| A1 Inter-CBL | 已完成 | 0.940596 | 4.790533 | 0.531697 | 与 A0 接近 |
| A2 Intra-CBL | 已完成 | 0.952554 | 3.368403 | 0.374899 | 修正前最优 |
| A3 Balance(原始) | 已完成 | 0.903470 | 7.922879 | 0.886811 | 明显退化 |
| A3R1 修正 | 已完成 | 0.913660 | 6.638482 | 0.754714 | 部分恢复，未通过 |
| A3R2 对照 | 已完成 | 0.904431 | 6.617903 | 0.801991 | 不通过 |
| A3R3 对照 | 已完成 | 0.959554 | 2.251109 | 0.246323 | 当前全局最优 |
| B1 Attention-only | 已完成 | 0.943297 | 3.602789 | 0.437852 | 模块可运行，作为第4章基线 |
| C1 Attention + A3R3 | 已完成 | 0.942718 | 4.417467 | 0.501330 | 严重退化，暴露特征冲突 |

### 1.3 当前阶段结论
1. A3R2 失败，说明“仅延后切换时机”不是主导修复方向。
2. A3R3 全面优于 A2，说明主因是 Inter 权重强度，且已找到有效区间。
3. B1 已完成，证明 Attention 模块可运行并建立了独立基线。
4. C1 失败表明：原 Attention 融合路径不适用。**研究重心转向通过前沿的参数高效微调 (LoRA) 和多尺度特征适配器 (LG-Adapter) 打造更稳健的第二、第三创新点**。

---

## 2. 论文叙事主线（答辩可讲清版）

### 2.1 问题定义
- 类别间不平衡：前景远少于背景，易出现前景召回不足。
- 类别内不平衡：易样本过多，难样本（边界/低对比）学习不足。
- 组合挑战：直接叠加多种平衡策略可能出现优化冲突。

### 2.2 方法主张
1. Inter-CBL：平衡前景与困难背景贡献。
2. Intra-CBL：提升困难样本权重。
3. 两阶段 Balance：先稳定学习，再引入完整项。
4. Attention：在已稳定 Loss 主干上提升跨病例特征融合。

### 2.3 关键证据链
1. A2 > A0/A1：证明 Intra-CBL 有效。
2. A3 < A2：证明“组件组合方式”存在问题。
3. A3R2 失败 + A3R3 成功：完成单变量归因，锁定主因是 Inter 权重。
4. A3R3 > A2：证明修正后的 Balance 配置可作为优良主干。
5. 组合的崩溃（B1 vs C1）：跨样本 Attention (C1) 方案不仅未带来增益反而破坏表征，证明直接融合不可行。
6. **架构扩充决策（保三核）**：放弃 C1 路线，迅速引入当前主流的 LoRA（实验 C2）与特征流视角的 Local-Global Adapter（实验 C3）作为第二和第三大创新点，确保毕业论文的创新点厚度与工作量壁垒。

### 2.4 最终决策规则（当前已落地）
1. Loss 主干：锁定 A3R3 作为损失层唯一最佳创新。
2. 架构探索：放弃粗颗粒度的跨样本 Attention，全力落实和验证 LoRA 与 Local-Global Adapter。只要这两个结构创新点跑通任意一个，即可搭配 A3R3 形成满量顶会级医学分割论文的主线故事。

---

## 3. 章节-实验映射（写论文直接查这张表）

| 论文章节 | 章节目标 | 必须实验/证据 | 关键图表 | 来源文档 |
|---|---|---|---|---|
| 第1章 绪论 | 讲清问题与价值 | 任务背景、挑战定义 | 研究框架图 | `docs/THESIS_PLAN.md` |
| 第2章 相关工作 | 铺垫理论与差异 | Loss 与 SAM/MedSAM 综述 | 方法对比表 | `docs/THESIS_PLAN.md` |
| 第3章 Balance 方法 | 讲清 Inter/Intra/两阶段 | A0/A1/A2/A3/A3R1/A3R2/A3R3 | 主结果表、归因对照表 | `docs/EXPERIMENT_LOG.md` |
| 第4章 Attention 模块 | 讲清“可运行(B1)但组合退化(C1)”及原因分析 | EXP-005（B1/C1） | 对比表、失败案例图 | `docs/FULL_EXPERIMENT_PLAN.md` |
| 第5章 综合实验 | 展示完整方法效果 | Full、泛化、可视化 | 综合对比表、案例图 | `docs/FULL_EXPERIMENT_PLAN.md` |
| 第6章 总结展望 | 回答“做成了什么” | 结论追溯与不足 | 结论清单 | 本文档 + `docs/THESIS_PLAN.md` |

---

## 4. 证据索引（结果追溯入口）

### 4.1 指标文件
- `work_dir/eval_metrics/A0_summary.json`
- `work_dir/eval_metrics/A1_summary.json`
- `work_dir/eval_metrics/A2_summary.json`
- `work_dir/eval_metrics/A3_summary.json`
- `work_dir/eval_metrics/A3R1_summary.json`
- `work_dir/eval_metrics/A3R2_summary.json`
- `work_dir/eval_metrics/A3R3_summary.json`
- `work_dir/eval_metrics/B1_summary.json`
- `work_dir/eval_metrics/C1_summary.json`

### 4.2 训练日志
- `work_dir/A1_20260209-2026.log`
- `work_dir/A2_20260210-2309.log`
- `work_dir/A3_20260212-002344_bs1.log`
- `work_dir/exp_logs/A3R1_train.log`
- `work_dir/exp_logs/A3R2_train.log`
- `work_dir/exp_logs/A3R3_train.log`
- `work_dir/exp_logs/B1_train.log`
- `work_dir/exp_logs/C1_train.log`
- `work_dir/eval_metrics/logs/B1_eval.log`

### 4.3 关键模型权重
- `work_dir/MedSAM-Baseline-20260208-1953/medsam_model_best.pth`
- `work_dir/MedSAM-FLARE22-A2-IntraCBL-20260210-2309-20260210-2309/medsam_model_best.pth`
- `work_dir/MedSAM-FLARE22-A3R3-Balance-a0.5-b1.0-s50-*/medsam_model_best.pth`
- `work_dir/MedSAM-FLARE22-B1-AttnOnly-*/medsam_model_best.pth`
- `work_dir/MedSAM-FLARE22-C1-Attn-BalanceR3-20260224-2122/medsam_model_best.pth`

---

## 5. 写作工作流（建议）

1. 固化第3章最终结果表（将 A3R3 包装为核心亮点）。
2. 回写第1/2章（突出“扎实的对照分析”与“发现并探讨不同模块间的组合壁垒”）。
3. 第4章**调整叙事收口**：诚实陈述 C1 不如预期，利用消融结果分析 Attention 机制与基于分布假设的 Balance Loss 在优化目标上的内生冲突，这本身就是极好的讨论（Discussion）。
4. 随后直接进入第5章整合图表，准备结束整个实验跑动环节。
4. 第6章把 C1 作为局限性与后续工作入口。
5. 第5章最后整合，避免反复改表。

---

## 6. 文档导航

- 总体技术报告（联网核验）：`docs/THESIS_TECHNICAL_REPORT.md`
- 章节写作蓝图：`docs/THESIS_PLAN.md`
- 实验事实库：`docs/EXPERIMENT_LOG.md`
- 路线与决策门槛：`docs/FULL_EXPERIMENT_PLAN.md`
- 当前可执行命令：`docs/SERVER_COMMANDS.md`
- 本周执行面板：`docs/WEEKLY_TASKS.md`
- 历史口径与归档：`docs/APPENDIX_HISTORY.md`

---

## 7. 使用原则

1. 论文结论只引用“同口径指标 + 可追溯日志”。
2. 主线文档只保留当前有效方案；历史内容统一放附录。
3. 每次新实验完成后，按顺序更新：
   1. `docs/EXPERIMENT_LOG.md`
   2. `docs/THESIS_MASTER_GUIDE.md`
   3. `docs/WEEKLY_TASKS.md`
