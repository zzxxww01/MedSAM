# 服务器命令中心（当前主线）

> 当前主线：FLARE22 / `data/npy/CT_Abd`  
> 非当前主线历史命令见：`docs/APPENDIX_HISTORY.md`

---

## 0. 环境准备

```bash
cd ~/chengang/zxw/MedSAM
conda activate medsam
```

---

## 1. 当前进度快照（2026-02-27）

| 实验 | 状态 | 指标/说明 |
|---|---|---|
| A0 | 已完成 | DSC=0.940741, HD95=4.830503, ASD=0.537757 |
| A1 | 已完成 | DSC=0.940596, HD95=4.790533, ASD=0.531697 |
| A2 | 已完成 | DSC=0.952554, HD95=3.368403, ASD=0.374899 |
| A3 | 已完成 | DSC=0.903470, HD95=7.922879, ASD=0.886811 |
| A3R1 | 已完成 | DSC=0.913660, HD95=6.638482, ASD=0.754714 |
| A3R2 | 已完成 | DSC=0.904431, HD95=6.617903, ASD=0.801991 |
| A3R3 | 已完成 | DSC=0.959554, HD95=2.251109, ASD=0.246323（当前最优） |
| B1 | 已完成 | DSC=0.943297, HD95=3.602789, ASD=0.437852（Attention-only 基线） |
| C1 | 已完成 | DSC=0.942719, HD95=4.417467, ASD=0.501330（低于 A3R3 与 B1） |

---

## 2. 结果快速核验命令（A2/R1/R2/R3/B1/C1）

```bash
python - <<'PY'
import json
files = [
  "work_dir/eval_metrics/A2_summary.json",
  "work_dir/eval_metrics/A3R1_summary.json",
  "work_dir/eval_metrics/A3R2_summary.json",
  "work_dir/eval_metrics/A3R3_summary.json",
  "work_dir/eval_metrics/B1_summary.json",
  "work_dir/eval_metrics/C1_summary.json",
]
for p in files:
    d = json.load(open(p, "r", encoding="utf-8"))
    print(f"{d['exp_name']}: DSC={d['dice_mean']:.6f}, HD95={d['hd95_mean']:.6f}, ASD={d['asd_mean']:.6f}")
PY
```

---

## 3. Attention 阶段启动前检查（EXP-005）

```bash
ls -l train_fss.py models/medsam_fss.py modules/attention_cross_block.py
```

---

## 4. 架构创新实验启动模板

## 4.1 C2：A3R3 + LoRA 微调
```bash
mkdir -p work_dir/exp_logs

nohup bash -lc '
set -e
cd ~/chengang/zxw/MedSAM
conda activate medsam
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=localhost
export MASTER_PORT=12358
export MPLBACKEND=Agg
python train_fss.py \
  -i data/npy/CT_Abd \
  -task_name MedSAM-FLARE22-C2-LoRA \
  -model_type vit_b \
  -checkpoint work_dir/medsam_vit_b.pth \
  -num_epochs 200 \
  -batch_size 2 \
  -lr 0.0001 \
  --world_size 2 \
  -use_amp \
  -loss_type balance \
  -balance_alpha 0.5 \
  -balance_beta 1.0 \
  -balance_gamma 1.0 \
  -stage1_epochs 50 \
  -balance_hard_threshold 0.9 \
  -balance_hard_weight 2.0 \
  -balance_neg_ratio 3.0 \
  -use_lora True \
  -lora_rank 4
' > work_dir/exp_logs/C2_train.log 2>&1 < /dev/null &
echo $! > work_dir/exp_logs/C2_train.pid
```

## 4.2 C3：A3R3 + Local-Global Adapter
```bash
mkdir -p work_dir/exp_logs

nohup bash -lc '
set -e
cd ~/chengang/zxw/MedSAM
conda activate medsam
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=localhost
export MASTER_PORT=12359
export MPLBACKEND=Agg
python train_fss.py \
  -i data/npy/CT_Abd \
  -task_name MedSAM-FLARE22-C3-LGAdapter \
  -model_type vit_b \
  -checkpoint work_dir/medsam_vit_b.pth \
  -num_epochs 200 \
  -batch_size 2 \
  -lr 0.0001 \
  --world_size 2 \
  -use_amp \
  -loss_type balance \
  -balance_alpha 0.5 \
  -balance_beta 1.0 \
  -balance_gamma 1.0 \
  -stage1_epochs 50 \
  -balance_hard_threshold 0.9 \
  -balance_hard_weight 2.0 \
  -balance_neg_ratio 3.0 \
  -use_lg_adapter True
' > work_dir/exp_logs/C3_train.log 2>&1 < /dev/null &
echo $! > work_dir/exp_logs/C3_train.pid
```

---

## 5. 训练监控命令

```bash
ps -fp $(cat work_dir/exp_logs/C2_train.pid)
tail -n 40 work_dir/exp_logs/C2_train.log
ps -fp $(cat work_dir/exp_logs/C3_train.pid)
tail -n 40 work_dir/exp_logs/C3_train.log
watch -n 2 nvidia-smi
```

---

## 6. 评估模板（按实验名替换）

```bash
mkdir -p work_dir/eval_metrics/logs
PY=/home/chengang/anaconda3/envs/medsam/bin/python
EXP=B1
CKPT=$(ls -dt work_dir/MedSAM-FLARE22-${EXP}-*/medsam_model_best.pth | head -n1)

nohup $PY eval_medsam_npz.py \
  --data_root data/npy/CT_Abd \
  --checkpoint "$CKPT" \
  --exp_name "$EXP" \
  --out_csv work_dir/eval_metrics/${EXP}_case_metrics.csv \
  --out_json work_dir/eval_metrics/${EXP}_summary.json \
  > work_dir/eval_metrics/logs/${EXP}_eval.log 2>&1 < /dev/null &
echo $! > work_dir/eval_metrics/logs/${EXP}_eval.pid
```

---

## 7. 常见问题与处理

1. `MASTER_ADDR expected, but not set`
- 原因：DDP 环境变量未设置。
- 处理：必须使用本文件的 `nohup bash -lc` 模板。

2. `OutOfMemoryError`
- 处理：维持 `batch_size=1`，避免并发大任务。

3. `python: can't open file 'train_fss.py'`
- 原因：Attention 脚本尚未落地或路径错误。
- 处理：先执行第 3 节检查命令，确认脚本存在再启动。

---

## 8. 关联文档

- 实验事实：`docs/EXPERIMENT_LOG.md`
- 路线决策：`docs/FULL_EXPERIMENT_PLAN.md`
- 论文总控：`docs/THESIS_MASTER_GUIDE.md`
