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
    -batch_size 2 \
    -lr 0.0001 \
    --world_size 2 \
    -use_amp \
    -loss_type balance
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
