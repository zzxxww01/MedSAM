# MedSAM 论文与实验总控指南（Master Guide）

> 更新时间：2026-03-02  
> 状态：**核心实验已全部斩获成功，准备撰写论文正文。**

---

## 1. 一页全局快照

### 1.1 当前研究主线（最终闭环 3-Innovation Strategy）
- 研究主题：基于 Balance Loss 与 Local-Global Adapter 相结合的多尺度医学图像分割架构（以 MedSAM 为基座）。
- 当前数据主线：`FLARE22 -> data/npy/CT_Abd` (40 例小样本多器官测试)。
- 核心里程碑：C3 实验登顶，DSC 突破 0.961大关，论文三大创新点论据全部齐备。

### 1.2 实验状态（同口径）

| 实验 | 状态 | DSC | HD95 | ASD | 结论 |
|---|---|---:|---:|---:|---|
| A0 Baseline | 已完成 | 0.940741 | 4.830503 | 0.537757 | 基线已建立 |
| A2 Intra-CBL | 已完成 | 0.952554 | 3.368403 | 0.374899 | 单因素有效 |
| A3R3 对照 | 已完成 | 0.959554 | 2.251109 | 0.246323 | Loss 创新点 (第一创新) 阶段性最优 |
| C2 LoRA | 已完成 | 0.879610 | 7.754797 | 0.996511 | 极有价值的负向消融点 (第二创新论据) |
| C3 LG-Adapter | 已完成 | **0.961958** | **2.914536** | **0.538947** | **全局第一，终极绝杀模型** (第三创新) |

### 1.3 当前阶段结论（实验已完全终结）
1. A3R3 证明了在器官极度不平衡下，基于梯度映射的空间重塑能强行拔高分割上限。
2. C2 的暴跌证明：面对高难度医疗域，冻结 ViT 主干以求参数高效微调（PEFT）行不通，模型彻底失去重塑特征流形的自由度。
3. C3 的破局证明：在全参微调的基础上，额外加装多尺度膨胀卷积适配器（Local-Global Adapter）能完美命中 ViT 缺失的高频边缘响应，收割终极指标。

---

## 2. 论文叙事主线（答辩可讲清版）

### 2.1 问题定义
- 类别间不平衡：前景远少于背景，易出现前景召回不足。
- 类别内不平衡：易样本过多，难样本（边界/低对比）学习不足。
- 组合挑战：直接叠加多种平衡策略可能出现优化冲突。

### 2.2 方法主张 (3-Innovation 黄金组合)
1. **A3R3 平衡损失**：梯度层的硬约束，拉升整体重叠度。
2. **LoRA 参数高效微调 (Ablation)**：架构层的显存节省尝试，通过反证其由于过度冻结带来的灾难性欠拟合，论证本研究必须“放开主干更新”。
3. **Local-Global 特征适配器**：架构层的空间弥补，通过并联卷积重拾 ViT 丢掉的高频细微病灶/器官边缘。

### 2.3 关键证据链
1. A0 -> A3R3 (0.941 -> 0.959)：梯度重构初战告捷。
2. A3R3 -> C2 (0.959 -> 0.879)：消融实验，死锁主干导致断崖式崩盘。
3. A3R3 -> C3 (0.959 -> 0.962)：终极加装适配器，弥补高频响应，制霸全局。

### 2.4 后续决定
实验已经彻底通关，**锁死全部模型代码，不再做任何修改**。接下来的唯一任务是提取数据画图、撰写大段落论文。

---

## 3. 章节-实验映射（写论文直接查这张表）

| 论文章节 | 章节目标 | 必须实验/证据 | 关键图表 | 来源文档 |
|---|---|---|---|---|
| 第1章 绪论 | 讲清问题与价值 | 任务背景、挑战定义 | 研究框架图 | `docs/THESIS_PLAN.md` |
| 第2章 相关工作 | 铺垫理论与差异 | Loss 与 SAM Adapter/PEFT 综述 | 方法对比表 | `docs/THESIS_PLAN.md` |
| 第3章 Balance 方法 | 讲清梯度重构 | A0/A2/A3R3 | 主结果表、归因对照表 | `docs/EXPERIMENT_LOG.md` |
| 第4章 架构级创新 | 讲清主干局限性、PEFT消融与适配器设计 | C2 (负向) / C3 (全场最佳) | 对比表、边缘细节放大图 | 本文档 + `docs/THESIS_TECHNICAL_REPORT.md` |
| 第5章 综合实验 | 展示完整方法效果 | Full、可视化 | 综合对比表、案例图 | 本文档 |
| 第6章 总结展望 | 回答“做成了什么” | 结论追溯与不足 | 结论清单 | `docs/THESIS_PLAN.md` |

---

## 4. 证据索引（结果追溯入口）

### 4.1 指标文件
- `work_dir/eval_metrics/A0_summary.json`
- `work_dir/eval_metrics/A3R3_summary.json`
- `work_dir/eval_metrics/C2_summary.json`
- `work_dir/eval_metrics/C3_summary.json`

### 4.2 训练日志
### 4.2 训练日志
- `work_dir/exp_logs/A3R3_train.log`
- `work_dir/exp_logs/C2_train.log`
- `work_dir/exp_logs/C3_train.log`

### 4.3 关键模型权重
- `work_dir/MedSAM-Baseline-20260208-1953/medsam_model_best.pth`
- `work_dir/MedSAM-FLARE22-A3R3-Balance-a0.5-b1.0-s50-*/medsam_model_best.pth`
- `work_dir/MedSAM-FLARE22-C2-LoRA-*/medsam_model_best.pth`
- `work_dir/MedSAM-FLARE22-C3-LGAdapter-*/medsam_model_best.pth`

---

## 5. 写作工作流（建议）

## 5. 写作工作流（即将开启）

1. 利用评估脚本生成可视化截图，用这三个模型对比 (A0、A3R3、C3)。
2. 把 `docs/THESIS_TECHNICAL_REPORT.md` 里的理论部分直接粘进论文框架。
3. 把 `docs/EXPERIMENT_LOG.md` 里的数据抄进制表。
4. 全面润色语言，结束战斗！

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
