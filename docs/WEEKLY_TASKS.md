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
- [x] EXP-005 / B1 训练与评估完成（`B1_summary.json`，40 例）。
- [x] `docs/EXPERIMENT_LOG.md` / `docs/FULL_EXPERIMENT_PLAN.md` / `docs/THESIS_MASTER_GUIDE.md` 已同步更新。

---

## 3. 进行中

- [x] EXP-005 / C1：Attention + A3R3 完整组合实验（训练与评估完成，结果退化）。
- [ ] 第3章 3.5/3.6 与第4章 4.1-4.2 的衔接写作（从“变量剥离”过渡到“融合模块失败的归因”）。
- [ ] 完成第4章 B1/C1 对比小结与局限性段落（不再依赖 C2/C3 才能成稿）。

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
| A3R3 | 0.959554 | 2.251109 | 0.246323 | 当前全局最优 (核心主线) |
| B1 | 0.943297 | 3.602789 | 0.437852 | Attention 验证 (有效) |
| C1 | 0.942718 | 4.417467 | 0.501330 | 探索性组合 (无额外增益) |

关键解释（论文可直接使用）：
1. R2 失败说明“仅延后切换时机”不能修复 A3 退化。
2. R3 全面优于 A2，说明 `Inter` 权重强度是主导因素。
3. Balance 主干从 A2 迁移为 A3R3，有充分定量证据支撑。
4. B1 证明 Attention 模块可运行，但 C1 结合 A3R3 后发生了明确的性能崩溃。
5. **总体结论的进化**：在确保 A3R3 性能底盘的同时，响应更高的创新点数量要求，开启阶段二架构增强。放弃 C1 路线，全力投入 LoRA (参数高效微调) 与 Local-Global Adapter (特征流升级) 的开发与实验。

---

## 5. 已决策事项

1. 毕业论文三核体系构建：
   - 核心1 (Loss)：A3R3 梯度平衡策略 (已验证成功)。
   - 核心2 (PEFT)：LoRA 微调机制解决大模型微调的遗忘问题 (待验证)。
   - 核心3 (特征)：局部-全局高频适配器弥补 SAM 多尺度细节缺失 (待验证)。
2. 本周剩余动作：不再纠结旧 Attention，迅速编写 LoRA 层和 Adapter 层代码，在服务器并行拉起两项极具落地潜力的轻量级实验。

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
3. `work_dir/eval_metrics/B1_summary.json`（已完成）
4. `work_dir/eval_metrics/C1_summary.json`（已完成）
5. 汇总以上四个结果，提炼论文第四/五章的关键图表说明。
6. 更新后的 `docs/THESIS_TECHNICAL_REPORT.md`
