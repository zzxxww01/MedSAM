# MedSAM 服务器命令汇总

## 环境配置

```bash
cd ~/chengang/zxw/MedSAM
conda activate medsam
```

---

## 1. 数据预处理

### AMOS22 数据集预处理
```bash
python pre_AMOS22.py
```

输出目录: `data/npy/CT_AMOS/`

---

## 2. 训练命令

### A0: Baseline (Dice + CE Loss)
```bash
CUDA_VISIBLE_DEVICES=0,1 \
MASTER_ADDR=localhost \
MASTER_PORT=12355 \
MPLBACKEND=Agg \
python train_multi_gpus.py \
    -i data/npy/CT_AMOS \
    -task_name MedSAM-AMOS-Baseline \
    -model_type vit_b \
    -checkpoint work_dir/medsam_vit_b.pth \
    -num_epochs 200 \
    -batch_size 2 \
    -lr 0.0001 \
    --world_size 2 \
    -use_amp
```

### A1: Inter-CBL + Dice
```bash
CUDA_VISIBLE_DEVICES=0,1 \
MASTER_ADDR=localhost \
MASTER_PORT=12355 \
MPLBACKEND=Agg \
python train_multi_gpus_balance.py \
    -i data/npy/CT_AMOS \
    -task_name MedSAM-AMOS-InterCBL \
    -model_type vit_b \
    -checkpoint work_dir/medsam_vit_b.pth \
    -num_epochs 200 \
    -batch_size 2 \
    -lr 0.0001 \
    --world_size 2 \
    -use_amp \
    -loss_type inter_cbl
```

### A2: Intra-CBL + Dice
```bash
CUDA_VISIBLE_DEVICES=0,1 \
MASTER_ADDR=localhost \
MASTER_PORT=12355 \
MPLBACKEND=Agg \
python train_multi_gpus_balance.py \
    -i data/npy/CT_AMOS \
    -task_name MedSAM-AMOS-IntraCBL \
    -model_type vit_b \
    -checkpoint work_dir/medsam_vit_b.pth \
    -num_epochs 200 \
    -batch_size 2 \
    -lr 0.0001 \
    --world_size 2 \
    -use_amp \
    -loss_type intra_cbl
```

### A3: Balance Loss (完整)
```bash
CUDA_VISIBLE_DEVICES=0,1 \
MASTER_ADDR=localhost \
MASTER_PORT=12355 \
MPLBACKEND=Agg \
python train_multi_gpus_balance.py \
    -i data/npy/CT_AMOS \
    -task_name MedSAM-AMOS-BalanceLoss \
    -model_type vit_b \
    -checkpoint work_dir/medsam_vit_b.pth \
    -num_epochs 200 \
    -batch_size 1 \
    -lr 0.0001 \
    --world_size 2 \
    -use_amp \
    -loss_type balance
```

> 备注（2026-02-13）：A3 在 FLARE22 上已验证 `batch_size=1` 可稳定完成 200 epochs；若改回 `batch_size=2`，需先确认显存余量。

### A3 修正实验（FLARE22 主线，推荐）

> 背景：当前同口径评估中 A3 指标劣于 A1/A2，优先做参数修正再进入 Attention 阶段。

#### R1（优先）: 降低 Inter 权重 + 延后阶段切换
```bash
CUDA_VISIBLE_DEVICES=0,1 \
MASTER_ADDR=localhost \
MASTER_PORT=12355 \
MPLBACKEND=Agg \
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
```

#### R2: 仅调整切换时机（对照）
```bash
CUDA_VISIBLE_DEVICES=0,1 \
MASTER_ADDR=localhost \
MASTER_PORT=12356 \
MPLBACKEND=Agg \
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
```

#### R3: 降低 Inter 权重（切换保持50）
```bash
CUDA_VISIBLE_DEVICES=0,1 \
MASTER_ADDR=localhost \
MASTER_PORT=12357 \
MPLBACKEND=Agg \
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
```

---

## 3. 监控与验证

### 查看训练日志
```bash
tail -f work_dir/MedSAM-AMOS-*/train.log
```

### 查看GPU使用情况
```bash
watch -n 1 nvidia-smi
```

### 检查模型文件
```bash
ls -la work_dir/MedSAM-AMOS-*/
```

---

## 4. 评估指标

评估指标包括:
- DSC (Dice Similarity Coefficient)
- HD95 (95% Hausdorff Distance)
- ASD (Average Surface Distance)
