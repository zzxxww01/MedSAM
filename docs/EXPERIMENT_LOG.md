# 实验事实库（标准化卡片）

> 更新时间：2026-02-14  
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
| EXP-003 | A2 Intra-CBL | 完成 | 当前最优 | `A2_summary.json` |
| EXP-004 | A3 Balance(原始) | 完成 | 明显退化 | `A3_summary.json` |
| EXP-007 | A3R1 修正 | 进行中 | 等待评估 | `A3R1_summary.json`(待) |
| EXP-008 | A3R2 修正对照 | 未启动 | 待决策 | - |
| EXP-009 | A3R3 修正对照 | 未启动 | 待决策 | - |

---

## 2. 统一结果表（同口径）

| 实验 | DSC | HD95 | ASD | 对比结论 |
|---|---:|---:|---:|---|
| A0 | 0.940741 | 4.830503 | 0.537757 | 基线 |
| A1 | 0.940596 | 4.790533 | 0.531697 | 与 A0 近似 |
| A2 | 0.952554 | 3.368403 | 0.374899 | 最优 |
| A3 | 0.903470 | 7.922879 | 0.886811 | 退化 |
| A3R1 | 待回填 | 待回填 | 待回填 | 验证中 |

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
- 作为当前默认最优模型与后续基准。

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

## 3.5 EXP-007：A3R1 修正实验（运行中）

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

### Run Command（记录）
```bash
nohup bash -lc '
set -e
cd ~/chengang/zxw/MedSAM
conda activate medsam
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export MPLBACKEND=Agg
python train_multi_gpus_balance.py \
  -i data/npy/CT_Abd \
  -task_name MedSAM-FLARE22-A3R1-Balance-a0.5-b1.0-s70 \
  -model_type vit_b \
  -checkpoint work_dir/medsam_vit_b.pth \
  -num_epochs 200 \
  -batch_size 1 \
  -lr 0.0001 \
  --world_size 2 \
  -use_amp \
  -loss_type balance \
  -balance_alpha 0.5 \
  -balance_beta 1.0 \
  -balance_gamma 1.0 \
  -stage1_epochs 70 \
  -balance_hard_threshold 0.9 \
  -balance_hard_weight 2.0 \
  -balance_neg_ratio 3.0
' > work_dir/exp_logs/A3R1_train.log 2>&1 < /dev/null &
echo $! > work_dir/exp_logs/A3R1_train.pid
```

### Evidence（当前）
- PID：`work_dir/exp_logs/A3R1_train.pid`
- 日志：`work_dir/exp_logs/A3R1_train.log`
- 状态：Rank0/1 已启动并进入训练。

### Post-Train Evaluation（待执行）
```bash
mkdir -p work_dir/eval_metrics/logs
PY=/home/chengang/anaconda3/envs/medsam/bin/python
A3R1_CKPT=$(ls -dt work_dir/MedSAM-FLARE22-A3R1-Balance-a0.5-b1.0-s70-*/medsam_model_best.pth | head -n1)

nohup $PY eval_medsam_npz.py \
  --data_root data/npy/CT_Abd \
  --checkpoint "$A3R1_CKPT" \
  --exp_name A3R1 \
  --out_csv work_dir/eval_metrics/A3R1_case_metrics.csv \
  --out_json work_dir/eval_metrics/A3R1_summary.json \
  > work_dir/eval_metrics/logs/A3R1_eval.log 2>&1 < /dev/null &
echo $! > work_dir/eval_metrics/logs/A3R1_eval.pid
```

### Decision Rule
1. 若 A3R1 明显优于 A3 且接近/超过 A2：锁定 R1，进入 Attention。
2. 若 A3R1 仅部分恢复：继续 R2/R3 做变量剥离。

---

## 4. 下一批实验（待激活）

## 4.1 EXP-008：A3R2（只改切换时机）
- 目的：验证 stage1 延后是否是主因。
- 条件：A3R1 未达到预期时启动。

## 4.2 EXP-009：A3R3（只改 Inter 权重）
- 目的：验证 alpha 调低是否是主因。
- 条件：A3R1 未达到预期时启动。

---

## 5. 引用规范（论文写作）

1. 章节中给出的数值必须与本文件表格一致。
2. 每个结论至少对应一个 `summary.json` 和一个训练日志来源。
3. 未完成实验只能写“进行中/待验证”，不能写确定结论。
