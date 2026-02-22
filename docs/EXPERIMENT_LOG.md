# 实验事实库（标准化卡片）

> 更新时间：2026-02-22  
> 作用：作为论文与答辩的“唯一实验事实来源”。  
> 总控入口：`docs/THESIS_MASTER_GUIDE.md`

---

## 0. 评估口径定义（统一）

1. 数据：`data/npy/CT_Abd`（40 例）。
2. 指标：DSC / HD95 / ASD。
3. 评估脚本：`eval_medsam_npz.py`。
4. 设备：默认 `cuda:0`。
5. 结论引用规则：必须来自 `work_dir/eval_metrics/*_summary.json`。

---

## 1. 状态总览

| ID | 实验 | 状态 | 结果摘要 | 产物 |
|---|---|---|---|---|
| EXP-001 | A0 Baseline | 完成 | 稳定基线 | `A0_summary.json` |
| EXP-002 | A1 Inter-CBL | 完成 | 与 A0 接近 | `A1_summary.json` |
| EXP-003 | A2 Intra-CBL | 完成 | 历史最优（修正前） | `A2_summary.json` |
| EXP-004 | A3 Balance(原始) | 完成 | 明显退化 | `A3_summary.json` |
| EXP-007 | A3R1 修正 | 完成 | 相比 A3 恢复，但仍显著落后 A2 | `A3R1_summary.json` |
| EXP-008 | A3R2 修正对照 | 完成 | 仅改切换时机后仍显著退化 | `A3R2_summary.json` |
| EXP-009 | A3R3 修正对照 | 完成 | 显著优于 A2，锁定新主干 | `A3R3_summary.json` |

---

## 2. 统一结果表（同口径）

| 实验 | DSC | HD95 | ASD | 对比结论 |
|---|---:|---:|---:|---|
| A0 | 0.940741 | 4.830503 | 0.537757 | 基线 |
| A1 | 0.940596 | 4.790533 | 0.531697 | 与 A0 近似 |
| A2 | 0.952554 | 3.368403 | 0.374899 | 历史最优 |
| A3 | 0.903470 | 7.922879 | 0.886811 | 退化 |
| A3R1 | 0.913660 | 6.638482 | 0.754714 | 部分恢复（未达 A2） |
| A3R2 | 0.904431 | 6.617903 | 0.801991 | 未恢复（不通过） |
| A3R3 | 0.959554 | 2.251109 | 0.246323 | 当前全局最优（通过） |

---

## 3. 标准化实验卡片

## 3.1 EXP-001：A0 Baseline

### Objective
- 建立同口径可比基线。

### Hypothesis
- 预训练 MedSAM 在 CT_Abd 上应提供稳定起点。

### Config
- Checkpoint：`work_dir/MedSAM-Baseline-20260208-1953/medsam_model_best.pth`
- Eval 输出：`work_dir/eval_metrics/A0_summary.json`

### Evidence
- 评估日志：`work_dir/eval_metrics/logs/A0_eval.log`
- 汇总 JSON：`work_dir/eval_metrics/A0_summary.json`

### Result
- DSC=0.940741, HD95=4.830503, ASD=0.537757

### Decision
- 作为全部后续实验统一参照。

---

## 3.2 EXP-002：A1 Inter-CBL

### Objective
- 验证“仅类别间平衡”收益。

### Hypothesis
- Inter-CBL 可降低假阳性并带来小幅提升。

### Config
- 日志：`work_dir/A1_20260209-2026.log`
- 权重：`work_dir/MedSAM-FLARE22-A1-InterCBL-20260209-2026-20260209-2027/medsam_model_best.pth`
- Eval 输出：`work_dir/eval_metrics/A1_summary.json`

### Evidence
- 训练终点：Epoch 199 完成，无 Traceback。
- 汇总 JSON：`work_dir/eval_metrics/A1_summary.json`

### Result
- DSC=0.940596, HD95=4.790533, ASD=0.531697

### Comparison
- 与 A0 基本持平。

### Decision
- Inter 单独不是主增益项。

---

## 3.3 EXP-003：A2 Intra-CBL

### Objective
- 验证“类内难样本加权”收益。

### Hypothesis
- Intra-CBL 可显著改善边界和困难样本表现。

### Config
- 日志：`work_dir/A2_20260210-2309.log`
- 权重：`work_dir/MedSAM-FLARE22-A2-IntraCBL-20260210-2309-20260210-2309/medsam_model_best.pth`
- Eval 输出：`work_dir/eval_metrics/A2_summary.json`

### Evidence
- 训练终点：Epoch 199 完成，无 Traceback。
- 汇总 JSON：`work_dir/eval_metrics/A2_summary.json`

### Result
- DSC=0.952554, HD95=3.368403, ASD=0.374899

### Comparison
- 明显优于 A0/A1/A3。

### Decision
- 修正实验前的默认最优模型与后续基准。

---

## 3.4 EXP-004：A3 Balance（原始配置）

### Objective
- 验证完整 Balance 组合收益。

### Hypothesis
- Inter + Intra + Dice 应优于单项损失。

### Config
- 成功日志：`work_dir/A3_20260212-002344_bs1.log`
- 权重：`work_dir/MedSAM-FLARE22-A3-BalanceLoss-20260212-002344-20260212-0023/medsam_model_best.pth`
- Eval 输出：`work_dir/eval_metrics/A3_summary.json`

### Failure Analysis
1. 首轮曾 OOM（batch_size=2）。
2. 修复后（batch_size=1）训练完成，但评估明显退化。
3. 说明“可训练完成”不等于“配置有效”。

### Result
- DSC=0.903470, HD95=7.922879, ASD=0.886811

### Decision
- 进入修正实验（R1/R2/R3），不直接进入 Attention。

---

## 3.5 EXP-007：A3R1 修正实验（已完成）

### Objective
- 验证“弱化 Inter + 延后切换”是否恢复性能。

### Hypothesis
- 原 A3 退化主要由 Inter 权重过强与阶段切换过早导致。

### Config
- `loss_type=balance`
- `balance_alpha=0.5`
- `balance_beta=1.0`
- `balance_gamma=1.0`
- `stage1_epochs=70`
- `balance_hard_threshold=0.9`
- `balance_hard_weight=2.0`
- `balance_neg_ratio=3.0`

### Evidence
- 日志：`work_dir/exp_logs/A3R1_train.log`
- 评估日志：`work_dir/eval_metrics/logs/A3R1_eval.log`
- 评估汇总：`work_dir/eval_metrics/A3R1_summary.json`

### Result
- DSC=0.913660, HD95=6.638482, ASD=0.754714

### Detailed Comparison（详细对比）

| 对比项 | DSC变化 | HD95变化 | ASD变化 | 解读 |
|---|---:|---:|---:|---|
| A3R1 vs A3 | +0.010190 | -1.284397 | -0.132097 | 明显恢复，修正方向有效 |
| A3R1 vs A2 | -0.038894 | +3.270079 | +0.379815 | 与当时最优仍有明显差距 |
| A3R1 vs A0 | -0.027081 | +1.807979 | +0.216957 | 未回到基线 |

### Decision
1. R1 判定为“部分恢复，不通过”。
2. 进入 R2/R3 单变量剥离。

---

## 3.6 EXP-008：A3R2 修正对照（仅改切换时机，已完成）

### Objective
- 验证 `stage1_epochs` 延后是否是 A3 退化主因。

### Config
- `loss_type=balance`
- `balance_alpha=1.0`
- `balance_beta=1.0`
- `balance_gamma=1.0`
- `stage1_epochs=100`
- 其余参数与 A3 基本一致。

### Evidence
- 训练日志：`work_dir/exp_logs/A3R2_train.log`（Epoch 199 完成）
- 评估日志：`work_dir/eval_metrics/logs/A3R2_eval.log`
- 评估汇总：`work_dir/eval_metrics/A3R2_summary.json`

### Result
- DSC=0.904431, HD95=6.617903, ASD=0.801991

### Comparison
- A3R2 vs A3R1：DSC -0.009229，HD95 -0.020579，ASD +0.047277（整体不如 A3R1）。
- A3R2 vs A2：DSC -0.048122，HD95 +3.249500，ASD +0.427092（仍显著退化）。

### Decision
- R2 不通过。
- “仅延后切换时机”不能解释或修复主问题。

---

## 3.7 EXP-009：A3R3 修正对照（仅改 Inter 权重，已完成）

### Objective
- 验证降低 `balance_alpha` 是否是恢复性能的主因。

### Config
- `loss_type=balance`
- `balance_alpha=0.5`
- `balance_beta=1.0`
- `balance_gamma=1.0`
- `stage1_epochs=50`
- `balance_hard_threshold=0.9`
- `balance_hard_weight=2.0`
- `balance_neg_ratio=3.0`

### Evidence
- 训练日志：`work_dir/exp_logs/A3R3_train.log`（Epoch 199 完成）
- 日志参数行已确认：`alpha=0.5,beta=1.0,gamma=1.0,stage1_epochs=50,...`
- 评估日志：`work_dir/eval_metrics/logs/A3R3_eval.log`
- 评估汇总：`work_dir/eval_metrics/A3R3_summary.json`

### Result
- DSC=0.959554, HD95=2.251109, ASD=0.246323

### Comparison
- A3R3 vs A2：DSC +0.007000，HD95 -1.117294，ASD -0.128576（全面优于 A2）。
- A3R3 vs A3R1：DSC +0.045894，HD95 -4.387373，ASD -0.508391（显著提升）。

### Decision
1. R3 通过，且成为当前全局最优配置。
2. 退化主因锁定为 Inter 权重强度（H1 获得强证据支持）。
3. 后续 EXP-005（Attention 阶段）默认采用 A3R3 配置作为损失主干。

---

## 4. 下一阶段实验（已切换）

## 4.1 EXP-005：Attention 模块阶段（进行中）
- 阶段入口条件：已满足（R3 通过且优于 A2）。
- 默认 Loss 主干：A3R3（`alpha=0.5, beta=1.0, gamma=1.0, stage1_epochs=50`）。
- 先做：B1（Attention 模块独立增益验证）；
- 再做：C1（Attention + A3R3 完整方案）。

---

## 5. 引用规范（论文写作）

1. 章节中给出的数值必须与本文件表格一致。
2. 每个结论至少对应一个 `summary.json` 和一个训练日志来源。
3. 对“主因”类结论，必须给出至少一个单变量对照（本轮为 R2/R3）。
