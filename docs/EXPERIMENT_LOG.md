# 实验事实库

> **本文件是全部实验数据的唯一权威来源。** 所有论文中引用的数值必须与本表一致。
> 更新时间：2026-03-03 | 状态：全部实验完成，代码冻结
> 详细解读见 `docs/THESIS_KNOWLEDGE_BASE.md`

---

## 评估口径

- 数据：`data/npy/CT_Abd`（FLARE22，40 例独立测试集）
- 指标：DSC / HD95 / ASD
- 脚本：`eval_medsam_npz.py`
- 结果来源：`work_dir/eval_metrics/*_summary.json`

---

## 统一结果表

| 编号 | 方法组合 | DSC | HD95 | ASD | 结论 |
|:---|:---|---:|---:|---:|:---|
| **A0** | Baseline (Dice+CE) | 0.940741 | 4.830503 | 0.537757 | 基线 |
| **A1** | Inter-CBL only | 0.940596 | 4.790533 | 0.531697 | 单独Inter无增益 |
| **A2** | Intra-CBL only | 0.952554 | 3.368403 | 0.374899 | 困难样本加权有效 |
| **A3** | Balance (α=1.0, stage=50) | 0.903470 | 7.922879 | 0.886811 | 严重退化 |
| **A3R1** | Balance (α=0.5, stage=70) | 0.913660 | 6.638482 | 0.754714 | 部分恢复 |
| **A3R2** | Balance (α=1.0, stage=100) | 0.904431 | 6.617903 | 0.801991 | 未恢复 → H2否定 |
| **A3R3** | Balance (α=0.5, stage=50) | **0.959554** | **2.251109** | **0.246323** | **损失层最优** → H1通过 |
| **B1** | Attention-only (dicece) | 0.943297 | 3.602789 | 0.437852 | Attention独立基线 |
| **C1** | Attention + A3R3 | 0.942719 | 4.417467 | 0.501330 | 退化 → 废弃Attention |
| **C2** | LoRA (r=4) + A3R3 | 0.879610 | 7.754797 | 0.996511 | 灾难退化 → 消融反证 |
| **C3** | LG-Adapter + A3R3 | **0.961958** | 2.914536 | 0.538947 | **全局最优 DSC** |

---

## 实验卡片

### A0 Baseline
- Checkpoint: `work_dir/MedSAM-Baseline-20260208-1953/medsam_model_best.pth`
- Eval JSON: `work_dir/eval_metrics/A0_summary.json`

### A1 Inter-CBL
- Config: `loss_type=inter_cbl`
- 日志: `work_dir/A1_20260209-2026.log`
- Checkpoint: `work_dir/MedSAM-FLARE22-A1-InterCBL-20260209-2026-20260209-2027/medsam_model_best.pth`
- Eval JSON: `work_dir/eval_metrics/A1_summary.json`

### A2 Intra-CBL
- Config: `loss_type=intra_cbl`
- 日志: `work_dir/A2_20260210-2309.log`
- Checkpoint: `work_dir/MedSAM-FLARE22-A2-IntraCBL-20260210-2309-20260210-2309/medsam_model_best.pth`
- Eval JSON: `work_dir/eval_metrics/A2_summary.json`

### A3 Balance (原始)
- Config: `loss_type=balance, alpha=1.0, beta=1.0, gamma=1.0, stage1_epochs=50`
- 日志: `work_dir/A3_20260212-002344_bs1.log`
- Eval JSON: `work_dir/eval_metrics/A3_summary.json`
- 备注: batch_size=1（首轮OOM后降低）

### A3R1 修正 (双变量)
- Config: `alpha=0.5, stage1_epochs=70`
- 日志: `work_dir/exp_logs/A3R1_train.log`
- Eval JSON: `work_dir/eval_metrics/A3R1_summary.json`

### A3R2 修正 (仅改切换时机，验证H2)
- Config: `alpha=1.0, stage1_epochs=100`
- 日志: `work_dir/exp_logs/A3R2_train.log`
- Eval JSON: `work_dir/eval_metrics/A3R2_summary.json`
- 结论: H2否定（切换时机非主因）

### A3R3 修正 (仅改Inter权重，验证H1)
- Config: `alpha=0.5, beta=1.0, gamma=1.0, stage1_epochs=50, threshold=0.9, hard_weight=2.0, neg_ratio=3.0`
- 日志: `work_dir/exp_logs/A3R3_train.log`
- Eval JSON: `work_dir/eval_metrics/A3R3_summary.json`
- 结论: **H1通过，锁定为损失主干**

### B1 Attention-Only
- Config: `loss_type=dicece, use_attention=true, batch_size=1`
- 日志: `work_dir/exp_logs/B1_train.log`
- Checkpoint: `work_dir/MedSAM-FLARE22-B1-AttnOnly-20260223-0018/medsam_model_best.pth`
- Eval JSON: `work_dir/eval_metrics/B1_summary.json`

### C1 Attention + A3R3
- Config: `loss_type=balance(A3R3), use_attention=true`
- 日志: `work_dir/exp_logs/C1_train.log`
- Checkpoint: `work_dir/MedSAM-FLARE22-C1-Attn-BalanceR3-20260224-2122/medsam_model_best.pth`
- Eval JSON: `work_dir/eval_metrics/C1_summary.json`
- 结论: 退化，废弃Attention路线

### C2 LoRA + A3R3
- Config: `loss_type=balance(A3R3), use_lora=true, lora_rank=4`
- 日志: `work_dir/exp_logs/C2_train.log`
- Eval JSON: `work_dir/eval_metrics/C2_summary.json`
- 结论: 灾难退化，冻结主干不可行

### C3 LG-Adapter + A3R3
- Config: `loss_type=balance(A3R3), use_lg_adapter=true`
- 日志: `work_dir/exp_logs/C3_train.log`
- Eval JSON: `work_dir/eval_metrics/C3_summary.json`
- 结论: **全局最优 DSC=0.9620**

---

## 差分证据表

| 对比 | ΔDSC | ΔHD95 | ΔASD | 解读 |
|:---|---:|---:|---:|:---|
| A1 vs A0 | -0.0001 | -0.0400 | -0.0061 | Inter-only 基本无净增益 |
| A2 vs A0 | +0.0118 | -1.4621 | -0.1629 | Intra-CBL 稳定有效 |
| A3 vs A2 | -0.0491 | +4.5545 | +0.5119 | 原始Balance严重退化 |
| A3R2 vs A2 | -0.0481 | +3.2495 | +0.4271 | 仅改切换时机无效(H2否定) |
| A3R3 vs A2 | +0.0070 | -1.1173 | -0.1286 | 仅改Inter权重即显著提升(H1通过) |
| A3R3 vs A0 | +0.0188 | -2.5794 | -0.2914 | Balance Loss全面提升 |
| C2 vs A3R3 | -0.0800 | +5.5037 | +0.7502 | 冻结主干导致崩溃 |
| C3 vs A3R3 | +0.0024 | +0.6634 | +0.2926 | DSC进一步突破 |
| C1 vs A3R3 | -0.0168 | +2.1664 | +0.2550 | Attention融合有害 |

---

## 引用规范

1. 章节中给出的数值必须与本表一致
2. 每个结论至少对应一个 `summary.json` 来源
3. 对"主因"类结论必须给出单变量对照（R2/R3）
