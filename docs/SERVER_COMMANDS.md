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

## 1. 当前进度快照（2026-02-14）

| 实验 | 状态 | 指标/说明 |
|---|---|---|
| A0 | 已完成 | DSC=0.940741, HD95=4.830503, ASD=0.537757 |
| A1 | 已完成 | DSC=0.940596, HD95=4.790533, ASD=0.531697 |
| A2 | 已完成 | DSC=0.952554, HD95=3.368403, ASD=0.374899 |
| A3 | 已完成 | DSC=0.903470, HD95=7.922879, ASD=0.886811 |
| A3R1 | 训练中 | 修正实验进行中 |

---

## 2. A3R1 训练（非链式、nohup、换行式）

```bash
mkdir -p work_dir/exp_logs

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

---

## 3. 训练监控命令

```bash
ps -fp $(cat work_dir/exp_logs/A3R1_train.pid)
tail -n 30 work_dir/exp_logs/A3R1_train.log
watch -n 2 nvidia-smi
```

---

## 4. A3R1 训练完成后评估（非链式、nohup）

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

---

## 5. 可选：单独启动 A3R2 / A3R3（仅当 R1 不达标）

## 5.1 A3R2（只改切换时机）
```bash
mkdir -p work_dir/exp_logs

nohup bash -lc '
set -e
cd ~/chengang/zxw/MedSAM
conda activate medsam
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=localhost
export MASTER_PORT=12356
export MPLBACKEND=Agg
python train_multi_gpus_balance.py \
  -i data/npy/CT_Abd \
  -task_name MedSAM-FLARE22-A3R2-Balance-a1.0-b1.0-s100 \
  -model_type vit_b \
  -checkpoint work_dir/medsam_vit_b.pth \
  -num_epochs 200 \
  -batch_size 1 \
  -lr 0.0001 \
  --world_size 2 \
  -use_amp \
  -loss_type balance \
  -balance_alpha 1.0 \
  -balance_beta 1.0 \
  -balance_gamma 1.0 \
  -stage1_epochs 100 \
  -balance_hard_threshold 0.9 \
  -balance_hard_weight 2.0 \
  -balance_neg_ratio 3.0
' > work_dir/exp_logs/A3R2_train.log 2>&1 < /dev/null &
echo $! > work_dir/exp_logs/A3R2_train.pid
```

## 5.2 A3R3（只改 Inter 权重）
```bash
mkdir -p work_dir/exp_logs

nohup bash -lc '
set -e
cd ~/chengang/zxw/MedSAM
conda activate medsam
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=localhost
export MASTER_PORT=12357
export MPLBACKEND=Agg
python train_multi_gpus_balance.py \
  -i data/npy/CT_Abd \
  -task_name MedSAM-FLARE22-A3R3-Balance-a0.5-b1.0-s50 \
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
  -stage1_epochs 50 \
  -balance_hard_threshold 0.9 \
  -balance_hard_weight 2.0 \
  -balance_neg_ratio 3.0
' > work_dir/exp_logs/A3R3_train.log 2>&1 < /dev/null &
echo $! > work_dir/exp_logs/A3R3_train.pid
```

---

## 6. 统一汇总命令（A0/A1/A2/A3/A3R1）

```bash
python - <<'PY'
import json
files = [
  "work_dir/eval_metrics/A0_summary.json",
  "work_dir/eval_metrics/A1_summary.json",
  "work_dir/eval_metrics/A2_summary.json",
  "work_dir/eval_metrics/A3_summary.json",
  "work_dir/eval_metrics/A3R1_summary.json",
]
for p in files:
    d = json.load(open(p, "r", encoding="utf-8"))
    print(f"{d['exp_name']}: DSC={d['dice_mean']:.6f}, HD95={d['hd95_mean']:.6f}, ASD={d['asd_mean']:.6f}")
PY
```

---

## 7. 常见问题与处理

1. `MASTER_ADDR expected, but not set`
- 原因：DDP 环境变量未设置。
- 处理：必须使用本文件的 `nohup bash -lc` 模板。

2. `OutOfMemoryError`
- 处理：维持 `batch_size=1`，避免并发大任务。

3. SSH 断线后不确定任务是否在跑
- 处理：先 `ps -fp $(cat *.pid)`，再 `tail -n 30 *.log`。

---

## 8. 关联文档

- 实验事实：`docs/EXPERIMENT_LOG.md`
- 路线决策：`docs/FULL_EXPERIMENT_PLAN.md`
- 论文总控：`docs/THESIS_MASTER_GUIDE.md`
