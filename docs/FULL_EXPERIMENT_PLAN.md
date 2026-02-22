# 实验路线图（当前主线版）

> 更新时间：2026-02-22  
> 目标：给出“接下来做什么、何时切换路线、用什么判据”的决策完整方案。  
> 实验事实来源：`docs/EXPERIMENT_LOG.md`

---

## 1. 当前目标与约束

## 1.1 当前核心目标
1. R2/R3 单变量剥离已完成并形成明确结论。
2. Loss 主干已锁定为 A3R3（优于 A2）。
3. 启动 Attention 阶段（EXP-005）：先 B1，再 C1。

## 1.2 约束
1. 当前主线数据固定：`data/npy/CT_Abd`。
2. 训练任务时长长，默认使用 `nohup` 后台执行。
3. 所有结论采用同口径评估（DSC/HD95/ASD）。
4. Attention 阶段须在 Loss 主干固定后运行，避免变量耦合。

---

## 2. 理论假设与验证映射（R2/R3 后）

| 假设ID | 假设内容 | 证据基础 | 判定 |
|---|---|---|---|
| H1 | Inter 权重过强导致退化 | A3 远弱于 A2；R3 显著优于 A2 | 通过 |
| H2 | 阶段切换过早导致退化 | R2（仅改 stage1）仍显著退化 | 不通过 |
| H3 | 中等 Inter + 合理切换可恢复并超越 | R3 全面优于 A2 | 通过 |

---

## 3. 修正阶段实验矩阵（已完成）

| 实验 | alpha | beta | gamma | stage1_epochs | 状态 | 核心结果 |
|---|---:|---:|---:|---:|---|---|
| A3R1 | 0.5 | 1.0 | 1.0 | 70 | 完成 | 部分恢复但不通过 |
| A3R2 | 1.0 | 1.0 | 1.0 | 100 | 完成 | DSC=0.904431，未恢复 |
| A3R3 | 0.5 | 1.0 | 1.0 | 50 | 完成 | DSC=0.959554，当前最优 |

---

## 4. 决策门槛与已落地结果

## 4.1 R2/R3 判定
1. R2 通过条件：至少接近 A2。  
判定：不通过（`DSC 0.904431 < 0.952554`）。
2. R3 通过条件：三指标接近或优于 A2。  
判定：通过（R3 三指标全面优于 A2）。

| 对比项 | ΔDSC | ΔHD95 | ΔASD | 结论 |
|---|---:|---:|---:|---|
| A3R2 vs A2 | -0.048122 | +3.249500 | +0.427092 | 失败 |
| A3R3 vs A2 | +0.007000 | -1.117294 | -0.128576 | 成功 |

## 4.2 路线切换（已执行）
1. Loss 主干从 A2 切换为 A3R3。
2. 进入 EXP-005（Attention 阶段）。
3. 先做 B1（模块独立增益），再做 C1（完整方案）。

---

## 5. 执行顺序（当前版本）

1. B1：AttentionCrossBlock + Dice/CE（模块独立有效性验证）。
2. C1：AttentionCrossBlock + A3R3 Loss 主干（完整方案验证）。
3. 对比 A3R3/B1/C1，确认 Attention 的净增益来源与幅度。
4. 将结论回填第4章与第5章实验表。

---

## 6. Attention 阶段准入条件（EXP-005）

满足以下任一条件可进入：
1. R1 通过；
2. R2/R3 中至少一项通过；
3. 若全部不通过，则以 A2 为主干进入 Attention。

当前状态：条件 2 已满足（R3 通过），准入已达成。

---

## 7. 产出要求（每个实验必须具备）

1. 训练日志：`work_dir/exp_logs/*_train.log`
2. 模型权重：`work_dir/MedSAM-*/medsam_model_best.pth`
3. 评估结果：`work_dir/eval_metrics/*_summary.json`
4. 实验记录更新：`docs/EXPERIMENT_LOG.md`
5. 章节联动更新：`docs/THESIS_MASTER_GUIDE.md`、`docs/THESIS_TECHNICAL_REPORT.md`

---

## 8. 风险与应对

| 风险 | 现象 | 应对 |
|---|---|---|
| DDP 环境变量遗漏 | `MASTER_ADDR expected, but not set` | 统一使用标准 `nohup bash -lc` 模板 |
| 显存不足 | OOM | 维持 `batch_size=1`，避免并发大任务 |
| 长任务中断 | SSH 断开后状态不明 | 使用 `nohup` + PID 文件 + 日志 tail |
| 变量耦合难解释 | 同时改 Loss 与 Attention | 固定 A3R3 先做 B1 再做 C1 |

---

## 9. 与论文章节的对应关系

1. R1/R2/R3 已完成，支撑论文第3章“失败机制与修正归因”。
2. Attention 阶段（B1/C1）支撑第4章。
3. Full 阶段与泛化支撑第5章。

---

## 10. 关联文档

- 总控：`docs/THESIS_MASTER_GUIDE.md`
- 命令：`docs/SERVER_COMMANDS.md`
- 周执行：`docs/WEEKLY_TASKS.md`
- 历史归档：`docs/APPENDIX_HISTORY.md`
