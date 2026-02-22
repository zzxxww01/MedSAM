# 本周执行面板（实验 + 论文联动）

> 周期：2026-02-17 ~ 2026-02-24  
> 主线：R2/R3 已完成，Balance 主干锁定 A3R3，进入 Attention 阶段（EXP-005）

---

## 1. 本周目标

1. 完成 A3R2/A3R3 训练与同口径评估，定位 A3 退化主因。
2. 锁定最终 Balance 主干配置，并形成可追溯证据链。
3. 启动 Attention 阶段首轮实验（B1/C1），承接第4章写作。

---

## 2. 已完成

- [x] A0/A1/A2/A3/A3R1 同口径评估回填（40 例）。
- [x] A3R2 训练与评估完成（`A3R2_summary.json`）。
- [x] A3R3 训练与评估完成（`A3R3_summary.json`）。
- [x] R2/R3 单变量剥离结论明确：主因是 Inter 权重，不是切换时机。
- [x] Balance 主干配置锁定为 A3R3。
- [x] `docs/EXPERIMENT_LOG.md` / `docs/FULL_EXPERIMENT_PLAN.md` / `docs/THESIS_MASTER_GUIDE.md` 已同步更新。

---

## 3. 进行中

- [ ] EXP-005 / B1：Attention 模块首轮增益验证（模块可运行 + 同口径评估）。
- [ ] EXP-005 / C1：Attention + A3R3 完整组合实验准备。
- [ ] 第3章 3.5/3.6 与第4章 4.1 的衔接写作（从“变量剥离”过渡到“融合模块验证”）。

---

## 4. 当前结果快照（CT_Abd，40例，同口径）

| 实验 | DSC | HD95 | ASD | 判定 |
|---|---:|---:|---:|---|
| A0 | 0.940741 | 4.830503 | 0.537757 | 基线 |
| A1 | 0.940596 | 4.790533 | 0.531697 | 与基线近似 |
| A2 | 0.952554 | 3.368403 | 0.374899 | 历史最优 |
| A3 | 0.903470 | 7.922879 | 0.886811 | 明显退化 |
| A3R1 | 0.913660 | 6.638482 | 0.754714 | 部分恢复（未达标） |
| A3R2 | 0.904431 | 6.617903 | 0.801991 | 不通过 |
| A3R3 | 0.959554 | 2.251109 | 0.246323 | 当前全局最优 |

关键解释（论文可直接使用）：
1. R2 失败说明“仅延后切换时机”不能修复 A3 退化。
2. R3 全面优于 A2，说明 `Inter` 权重强度是主导因素。
3. Balance 主干从 A2 迁移为 A3R3，有充分定量证据支撑。
4. Attention 阶段已满足准入条件，可在固定 Loss 主干后继续控制变量验证。

---

## 5. 已决策事项

1. 后续 Loss 主干：A3R3（`alpha=0.5, beta=1.0, gamma=1.0, stage1_epochs=50`）。
2. Attention 阶段入口：已打开（EXP-005）。
3. 下一步实验顺序：先 B1（模块独立验证），再 C1（完整组合）。

---

## 6. 每日检查命令（Attention 阶段）

```bash
cd ~/chengang/zxw/MedSAM
conda activate medsam
nvidia-smi

# A3R3 关键结果确认
grep -E '"(dice_mean|hd95_mean|asd_mean)"' work_dir/eval_metrics/A3R3_summary.json

# Attention 实验（按实际 pid/log 文件名替换）
ps -fp $(cat work_dir/exp_logs/B1_train.pid)
tail -n 40 work_dir/exp_logs/B1_train.log
ps -fp $(cat work_dir/exp_logs/C1_train.pid)
tail -n 40 work_dir/exp_logs/C1_train.log
```

---

## 7. 论文写作联动任务

- [ ] 第3章：补入 R2/R3 结果与“主因判定”段落（可直接答辩引用）。
- [ ] 第4章：Attention 模块实验设置与评价口径落地。
- [ ] 第1章创新点：从“提出修正假设”升级为“完成单变量证伪并锁定主干”。

---

## 8. 本周交付清单

1. `work_dir/eval_metrics/A3R2_summary.json`（已完成）
2. `work_dir/eval_metrics/A3R3_summary.json`（已完成）
3. Attention 首轮结果文件（`B1_summary.json`）
4. 更新后的 `docs/EXPERIMENT_LOG.md`
5. 更新后的 `docs/THESIS_TECHNICAL_REPORT.md`
