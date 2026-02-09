# 实验记录模板

> 本文档用于记录所有实验的详细信息，包括配置、结果和分析

---

## 实验索引

| 实验ID | 日期 | 类型 | 描述 | 状态 | 结果摘要 |
|--------|------|------|------|------|----------|
| EXP-001 | - | Baseline | MedSAM原始模型 | 待运行 | - |
| EXP-002 | - | Ablation | Inter-CBL only | 待运行 | - |
| EXP-003 | - | Ablation | Intra-CBL only | 待运行 | - |
| EXP-004 | - | Ablation | Balance Loss (完整) | 待运行 | - |
| EXP-005 | - | Ablation | AttentionCrossBlock only | 待运行 | - |
| EXP-006 | - | Full | Balance Loss + Attention | 待运行 | - |

---

## 实验详细记录

### EXP-001: Baseline (MedSAM原始模型)

#### 基本信息
- **日期**: YYYY-MM-DD
- **运行时长**: -
- **GPU**: -
- **状态**: 待运行

#### 配置
```yaml
model:
  type: vit_b
  checkpoint: work_dir/SAM/sam_vit_b_01ec64.pth

training:
  epochs: 200
  batch_size: 4
  learning_rate: 0.0001
  weight_decay: 0.01

loss:
  type: dice_ce
  # Dice Loss + BCE Loss

data:
  train_path: data/npy/CT_Abd
  num_workers: 4
```

#### 运行命令
```bash
python train_one_gpu.py \
    -i data/npy/CT_Abd \
    -task_name MedSAM-Baseline \
    -model_type vit_b \
    -checkpoint work_dir/SAM/sam_vit_b_01ec64.pth \
    -num_epochs 200 \
    -batch_size 4 \
    -lr 0.0001 \
    --use_wandb True
```

#### 结果

| 数据集 | DSC (%) | HD95 (mm) | ASD (mm) |
|--------|---------|-----------|----------|
| FLARE22 (val) | - | - | - |
| KiTS19 | - | - | - |
| NIH | - | - | - |

#### 训练曲线
- Loss曲线: `results/EXP-001/loss_curve.png`
- DSC曲线: `results/EXP-001/dice_curve.png`

#### 分析
- 收敛情况:
- 过拟合情况:
- 其他观察:

---

### EXP-002: Inter-CBL Only

#### 基本信息
- **日期**: YYYY-MM-DD
- **运行时长**: -
- **GPU**: -
- **状态**: 待运行

#### 配置
```yaml
loss:
  type: inter_cbl
  # 仅使用Inter-CBL + Dice

training:
  epochs: 200
  batch_size: 4
  learning_rate: 0.0001
```

#### 运行命令
```bash
python train_balance_loss.py \
    -i data/npy/CT_Abd \
    -task_name MedSAM-InterCBL \
    -loss_type inter_cbl \
    -num_epochs 200 \
    -batch_size 4 \
    --use_wandb True
```

#### 结果

| 数据集 | DSC (%) | HD95 (mm) | ΔBaseline |
|--------|---------|-----------|-----------|
| FLARE22 (val) | - | - | - |

#### 对比分析
- 相比Baseline的提升/下降:
- 对小目标的效果:
- 训练稳定性:

---

### EXP-003: Intra-CBL Only

#### 基本信息
- **日期**: YYYY-MM-DD
- **状态**: 待运行

#### 配置
```yaml
loss:
  type: intra_cbl
  threshold: 0.9
  hard_weight: 2.0
  easy_weight: 1.0
```

#### 运行命令
```bash
python train_balance_loss.py \
    -i data/npy/CT_Abd \
    -task_name MedSAM-IntraCBL \
    -loss_type intra_cbl \
    -intra_threshold 0.9 \
    --use_wandb True
```

#### 结果

| 数据集 | DSC (%) | HD95 (mm) | ΔBaseline |
|--------|---------|-----------|-----------|
| FLARE22 (val) | - | - | - |

---

### EXP-004: Balance Loss (完整)

#### 基本信息
- **日期**: YYYY-MM-DD
- **状态**: 待运行

#### 配置
```yaml
loss:
  type: balance
  alpha: 1.0      # Inter-CBL权重
  beta: 1.0       # Intra-CBL权重
  gamma: 1.0      # Dice权重
  threshold: 0.9
  stage_switch_epoch: 50
```

#### 运行命令
```bash
python train_balance_loss.py \
    -i data/npy/CT_Abd \
    -task_name MedSAM-BalanceLoss \
    -loss_type balance \
    -balance_alpha 1.0 \
    -balance_beta 1.0 \
    -balance_gamma 1.0 \
    -stage_switch_epoch 50 \
    --use_wandb True
```

#### 结果

| 数据集 | DSC (%) | HD95 (mm) | ΔBaseline |
|--------|---------|-----------|-----------|
| FLARE22 (val) | - | - | - |

#### 阶段分析
- Stage 1 (Epoch 0-49) 表现:
- Stage 2 (Epoch 50+) 表现:
- 切换时机是否合适:

---

### EXP-005: AttentionCrossBlock Only

#### 基本信息
- **日期**: YYYY-MM-DD
- **状态**: 待运行

#### 配置
```yaml
model:
  type: medsam_fss
  use_attention: true
  num_support: 5

loss:
  type: dice_ce  # 使用原始损失

attention:
  embed_dim: 256
  num_heads: 8
```

#### 运行命令
```bash
python train_fss.py \
    -i data/npy/CT_Abd \
    -task_name MedSAM-Attention \
    -num_support 5 \
    -use_attention True \
    --use_wandb True
```

#### 结果

| 数据集 | DSC (%) | HD95 (mm) | ΔBaseline |
|--------|---------|-----------|-----------|
| FLARE22 (val) | - | - | - |

#### 注意力可视化
- 权重分布: `results/EXP-005/attention_weights.png`
- 特征图对比: `results/EXP-005/feature_comparison.png`

---

### EXP-006: 完整方案 (Balance Loss + Attention)

#### 基本信息
- **日期**: YYYY-MM-DD
- **状态**: 待运行

#### 配置
```yaml
model:
  type: medsam_fss
  use_attention: true
  num_support: 5

loss:
  type: balance
  alpha: 1.0
  beta: 1.0
  gamma: 1.0
  stage_switch_epoch: 50
```

#### 运行命令
```bash
python train_fss.py \
    -i data/npy/CT_Abd \
    -task_name MedSAM-Full \
    -num_support 5 \
    -use_attention True \
    -loss_type balance \
    -balance_alpha 1.0 \
    -balance_beta 1.0 \
    -balance_gamma 1.0 \
    --use_wandb True
```

#### 结果

| 数据集 | DSC (%) | HD95 (mm) | ΔBaseline |
|--------|---------|-----------|-----------|
| FLARE22 (val) | - | - | - |
| KiTS19 | - | - | - |
| NIH | - | - | - |
| BUSI | - | - | - |
| CVC-ClinicDB | - | - | - |

---

## 超参数搜索记录

### Balance Loss 超参数

| α | β | γ | threshold | DSC (%) | 备注 |
|---|---|---|-----------|---------|------|
| 1.0 | 1.0 | 1.0 | 0.9 | - | 默认 |
| 0.5 | 1.0 | 1.0 | 0.9 | - | 降低Inter-CBL |
| 1.0 | 2.0 | 1.0 | 0.9 | - | 增强Intra-CBL |
| 1.0 | 1.0 | 0.5 | 0.9 | - | 降低Dice |
| 1.0 | 1.0 | 1.0 | 0.8 | - | 降低阈值 |
| 1.0 | 1.0 | 1.0 | 0.95 | - | 提高阈值 |

### Stage切换时机

| switch_epoch | 最终DSC (%) | 收敛速度 | 备注 |
|--------------|-------------|----------|------|
| 30 | - | - | 较早切换 |
| 50 | - | - | 默认 |
| 70 | - | - | 较晚切换 |
| 100 | - | - | 很晚切换 |

### Attention模块参数

| num_heads | embed_dim | num_support | DSC (%) | 显存 (GB) |
|-----------|-----------|-------------|---------|-----------|
| 4 | 256 | 3 | - | - |
| 8 | 256 | 5 | - | - |
| 8 | 256 | 7 | - | - |
| 16 | 256 | 5 | - | - |

---

## 结果汇总表

### 主实验结果

| 方法 | FLARE22 DSC | FLARE22 HD95 | KiTS19 DSC | NIH DSC |
|------|-------------|--------------|------------|---------|
| MedSAM (Baseline) | - | - | - | - |
| + Inter-CBL | - | - | - | - |
| + Intra-CBL | - | - | - | - |
| + Balance Loss | - | - | - | - |
| + Attention | - | - | - | - |
| **Ours (Full)** | - | - | - | - |

### 对比方法结果

| 方法 | FLARE22 DSC | FLARE22 HD95 | 参数量 | 推理时间 |
|------|-------------|--------------|--------|----------|
| MedSAM (Baseline) | - | - | 93.7M | - |
| MedSAM (Ours) | - | - | ~95M | - |
| SAM | - | - | 308M | - |
| UniverSeg | - | - | - | - |
| nnU-Net | - | - | - | - |
| DeepLabV3+ | - | - | - | - |

---

## 可视化结果

### 分割结果对比

```
results/
├── visualization/
│   ├── comparison_flare22/
│   │   ├── case_001_baseline.png
│   │   ├── case_001_ours.png
│   │   └── ...
│   ├── comparison_kits19/
│   └── comparison_nih/
```

### 注意力权重可视化

```
results/
├── attention_maps/
│   ├── case_001_attn.png
│   └── ...
```

### 损失曲线对比

```
results/
├── loss_curves/
│   ├── all_methods_comparison.png
│   ├── balance_loss_components.png
│   └── ...
```

---

## 问题记录

### 问题1: [问题描述]
- **日期**: YYYY-MM-DD
- **现象**:
- **原因分析**:
- **解决方案**:
- **状态**: 已解决/进行中/待处理

### 问题2: [问题描述]
- **日期**: YYYY-MM-DD
- **现象**:
- **原因分析**:
- **解决方案**:
- **状态**: 已解决/进行中/待处理

---

## 下一步计划

- [ ] 任务1
- [ ] 任务2
- [ ] 任务3

---

**文档结束**

> 最后更新: YYYY-MM-DD
