# MedSAM 论文与实验总控指南（Master Guide）

> 更新时间：2026-02-14  
> 目标：用一份文档串起“论文叙事、实验证据、执行命令、下一步决策”。

---

## 1. 一页全局快照

### 1.1 当前研究主线
- 研究主题：基于 Balance Loss 与 Attention 融合的医学图像分割改进（以 MedSAM 为基座）。
- 当前数据主线：`FLARE22 -> data/npy/CT_Abd`。
- 当前优先级：先修正 Balance Loss（A3R1/R2/R3），再进入 Attention 模块。

### 1.2 实验状态（同口径）

| 实验 | 状态 | DSC | HD95 | ASD | 结论 |
|---|---|---:|---:|---:|---|
| A0 Baseline | 已完成 | 0.940741 | 4.830503 | 0.537757 | 基线已建立 |
| A1 Inter-CBL | 已完成 | 0.940596 | 4.790533 | 0.531697 | 与 A0 接近 |
| A2 Intra-CBL | 已完成 | 0.952554 | 3.368403 | 0.374899 | 当前最优 |
| A3 Balance(原始) | 已完成 | 0.903470 | 7.922879 | 0.886811 | 明显退化 |
| A3R1 修正 | 训练中 | 待回填 | 待回填 | 待回填 | 验证关键假设 |

### 1.3 当前阶段结论
1. Intra-CBL（A2）在当前口径下贡献最大。
2. 原始 Balance（A3）出现退化，提示 Inter 权重与阶段切换存在问题。
3. 下一步以 A3R1（`alpha=0.5, stage1=70`）为首选修正路径。

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
4. Attention（后续）：在损失稳定后提升特征融合质量。

### 2.3 关键证据链
1. A2 > A0/A1：证明 Intra-CBL 有效。
2. A3 < A2：证明“组件组合方式”存在问题，而非组件本身无效。
3. A3R1（进行中）：验证“弱化 Inter + 延后切换”能否恢复。

### 2.4 最终决策规则
1. 若 A3R1 三指标显著优于 A3 且接近/超过 A2：锁定修正 Balance，进入 Attention。
2. 若 A3R1 仅部分恢复：执行 R2/R3 继续剥离变量。
3. 若 R1/R2/R3 均不如 A2：以 A2 作为后续主干。

---

## 3. 章节-实验映射（写论文直接查这张表）

| 论文章节 | 章节目标 | 必须实验/证据 | 关键图表 | 来源文档 |
|---|---|---|---|---|
| 第1章 绪论 | 讲清问题与价值 | 任务背景、挑战定义 | 研究框架图 | `docs/THESIS_PLAN.md` |
| 第2章 相关工作 | 铺垫理论与差异 | Loss 与 SAM/MedSAM 综述 | 方法对比表 | `docs/THESIS_PLAN.md` |
| 第3章 Balance 方法 | 讲清 Inter/Intra/两阶段 | A0/A1/A2/A3/A3R1 | 主结果表、退化分析图 | `docs/EXPERIMENT_LOG.md` |
| 第4章 Attention 模块 | 讲清融合机制设计 | EXP-005 系列 | 模块结构图、消融表 | `docs/FULL_EXPERIMENT_PLAN.md` |
| 第5章 综合实验 | 展示完整方法效果 | Full、泛化、可视化 | 综合对比表、案例图 | `docs/FULL_EXPERIMENT_PLAN.md` |
| 第6章 总结展望 | 回答“做成了什么” | 结论追溯与不足 | 结论清单 | 本文档 + `docs/THESIS_PLAN.md` |

---

## 4. 证据索引（结果追溯入口）

### 4.1 指标文件
- `work_dir/eval_metrics/A0_summary.json`
- `work_dir/eval_metrics/A1_summary.json`
- `work_dir/eval_metrics/A2_summary.json`
- `work_dir/eval_metrics/A3_summary.json`
- `work_dir/eval_metrics/A3R1_summary.json`（待生成）

### 4.2 训练日志
- `work_dir/A1_20260209-2026.log`
- `work_dir/A2_20260210-2309.log`
- `work_dir/A3_20260212-002344_bs1.log`
- `work_dir/exp_logs/A3R1_train.log`

### 4.3 关键模型权重
- `work_dir/MedSAM-Baseline-20260208-1953/medsam_model_best.pth`
- `work_dir/MedSAM-FLARE22-A1-InterCBL-20260209-2026-20260209-2027/medsam_model_best.pth`
- `work_dir/MedSAM-FLARE22-A2-IntraCBL-20260210-2309-20260210-2309/medsam_model_best.pth`
- `work_dir/MedSAM-FLARE22-A3-BalanceLoss-20260212-002344-20260212-0023/medsam_model_best.pth`

---

## 5. 写作工作流（建议）

1. 先写第3章（方法 + 已有结果最完整）。
2. 再写第1/2章（用第3章反推绪论与相关工作聚焦点）。
3. A3R1 出结果后补第3章“修正实验与机制解释”。
4. Attention 实验稳定后写第4章。
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
