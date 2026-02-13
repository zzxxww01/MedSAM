# 实验记录模板

> 本文档用于记录所有实验的详细信息，包括配置、结果和分析

---

## 运行环境与协作方式（固定）

- 固定模式：**本地开发 + 远程服务器运行**。
- 本地目录仅用于代码和文档，不作为实验运行进度依据。
- 进度判定必须基于远程服务器证据（日志/GPU/产物）。
- 服务器固定信息：`~/chengang/zxw/MedSAM`、`conda: medsam`、`4 x V100`、默认2卡运行。

### 远程进度最小回传模板（每次实验都填）

```text
实验ID:
服务器时间:
执行命令:
最新日志(末尾20行):
GPU状态(nvidia-smi):
产物列表(work_dir相关目录):
当前结论: 待运行 / 运行中 / 已完成 / 失败
```

### 远程检查命令（可直接执行）

```bash
cd ~/chengang/zxw/MedSAM
conda activate medsam
nvidia-smi
tail -n 20 work_dir/<task_name>/train.log
ls -lah work_dir/<task_name>/
```

---

## 当前状态快照（2026-02-13，含评估回填）

| 模块 | 状态 | 服务器证据 | 下一动作 |
|------|------|------------|----------|
| Baseline | 已完成 | `MedSAM-Baseline-20260208-1953`，含 `medsam_model_best.pth` | 补跑同口径评估（用于ΔBaseline） |
| A1 Inter-CBL | 已完成 | `A1_20260209-2026.log` 结束于 `Epoch 199` | 已回填指标，保留为对照 |
| A2 Intra-CBL | 已完成 | `A2_20260210-2309.log` 结束于 `Epoch 199` | 当前最优，作为后续默认候选 |
| A3 Balance Loss | 已完成 | `A3_20260212-002344_bs1.log` 结束于 `Epoch 199`，best/latest 权重已生成 | 排查性能退化原因并做超参修正 |
| A5 超参数 | 未启动 | - | 优先围绕 A3（α/β/stage 切换）做修正实验 |
| EXP-005 Attention | 未启动 | - | 先做模块最小可运行验证 |
| EXP-006 Full | 未启动 | - | 待 Attention 实验稳定后启动 |

---

## 实验索引

| 实验ID | 日期 | 类型 | 描述 | 状态 | 结果摘要 |
|--------|------|------|------|------|----------|
| EXP-001 | 2026-02-08 | Baseline | MedSAM原始模型 | 已运行（可用基线） | 目录: MedSAM-Baseline-20260208-1953 |
| EXP-002 | 2026-02-09 ~ 2026-02-10 | Ablation | Inter-CBL only | 已完成 | 200 epoch完成，best/latest权重已保存 |
| EXP-003 | 2026-02-10 ~ 2026-02-11 | Ablation | Intra-CBL only | 已完成 | 200 epoch完成，best/latest权重已保存 |
| EXP-004 | 2026-02-12 ~ 2026-02-13 | Ablation | Balance Loss (完整) | 已完成 | 首轮OOM后改batch_size=1重跑完成（200 epochs） |
| EXP-005 | - | Ablation | AttentionCrossBlock only | 待运行 | - |
| EXP-006 | - | Full | Balance Loss + Attention | 待运行 | - |

---

## 实验详细记录

### EXP-001: Baseline (MedSAM原始模型)

#### 基本信息
- **日期**: 2026-02-08
- **运行时长**: -
- **GPU**: V100 (默认2卡策略)
- **状态**: 已运行（可用基线）

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

#### 当前证据（2026-02-09）
- Baseline运行目录共7个。
- `MedSAM-Baseline-20260208-1953` 已确认存在 `medsam_model_best.pth` 与 `MedSAM-Baselinetrain_loss.png`。
- `baseline_train.log` 当前内容对应一次失败启动（路径错误）：在 `~/chengang/zxw/MedSAM/work_dir` 下调用 `train_multi_gpus.py`，报错 `No such file or directory`。
- 其余目录是否完成需进一步查看日志/模型文件。

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
- **日期**: 2026-02-09 ~ 2026-02-10
- **运行时长**: 约23小时
- **GPU**: V100 x2 (CUDA_VISIBLE_DEVICES=0,1)
- **状态**: 已完成（200 epochs）

#### 配置
```yaml
loss:
  type: inter_cbl
  # 仅使用Inter-CBL + Dice

training:
  epochs: 200
  batch_size: 2
  learning_rate: 0.0001
```

#### 运行命令
```bash
python train_multi_gpus_balance.py \
    -i data/npy/CT_Abd \
    -task_name MedSAM-FLARE22-A1-InterCBL-<timestamp> \
    -loss_type inter_cbl \
    -num_epochs 200 \
    -batch_size 2 \
    --world_size 2 \
    -use_amp
```

#### 结果

| 数据集 | DSC (%) | HD95 (mm) | ΔBaseline |
|--------|---------|-----------|-----------|
| FLARE22 (val) | 94.06 | 4.79 | 待Baseline评估 |

#### 当前证据（2026-02-10）
- 日志文件: `work_dir/A1_20260209-2026.log`
- 产物目录: `work_dir/MedSAM-FLARE22-A1-InterCBL-20260209-2026-20260209-2027`
- 关键文件: `medsam_model_best.pth`, `medsam_model_latest.pth`, `MedSAM-FLARE22-A1-InterCBL-20260209-2026_train_loss.png`
- 结束日志: `Epoch 199`，无 `Traceback`
- 末轮损失（日志）: `0.0211279581` / `0.0211373862`（rank1/rank0）
- 评估汇总（服务器）: `work_dir/eval_metrics/A1_summary.json`
- 指标（40例）: DSC=`0.940596`，HD95=`4.790533`，ASD=`0.531697`

#### 对比分析
- 相比Baseline的提升/下降: 待Baseline同口径评估后补齐Δ值
- 对小目标的效果: 暂无器官级拆分结论（待case-level分析）
- 训练稳定性: 训练与评估过程稳定，结果可复现

---

### EXP-003: Intra-CBL Only

#### 基本信息
- **日期**: 2026-02-10 ~ 2026-02-11
- **运行时长**: 约23小时
- **GPU**: V100 x2 (CUDA_VISIBLE_DEVICES=0,1)
- **状态**: 已完成（200 epochs）

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
| FLARE22 (val) | 95.26 | 3.37 | 待Baseline评估 |

#### 当前证据（2026-02-11）
- 日志文件: `work_dir/A2_20260210-2309.log`
- 产物目录: `work_dir/MedSAM-FLARE22-A2-IntraCBL-20260210-2309-20260210-2309`
- 关键文件: `medsam_model_best.pth`, `medsam_model_latest.pth`, `MedSAM-FLARE22-A2-IntraCBL-20260210-2309_train_loss.png`
- 结束日志: `Epoch 199`，无 `Traceback`
- 末轮损失（日志）: `0.0089820238` / `0.0088780716`（rank0/rank1）
- 评估汇总（服务器）: `work_dir/eval_metrics/A2_summary.json`
- 指标（40例）: DSC=`0.952554`，HD95=`3.368403`，ASD=`0.374899`

---

### EXP-004: Balance Loss (完整)

#### 基本信息
- **日期**: 2026-02-12 ~ 2026-02-13
- **运行时长**: 首轮失败后重跑，完整训练至Epoch 199
- **GPU**: V100 x2（实际训练进程在GPU1/GPU3）
- **状态**: 已完成（200 epochs，batch_size=1重跑成功）

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
| FLARE22 (val) | 90.35 | 7.92 | 待Baseline评估 |

#### 当前证据（2026-02-13）
- 首轮失败日志: `work_dir/A3_20260212-0010.log`
- 首轮错误类型: `torch.cuda.OutOfMemoryError`（`image_encoder` 注意力计算阶段）
- 二次失败日志: `work_dir/A3_20260212-001844_bs1.log`
- 二次失败现象: `MASTER_ADDR expected, but not set`，并出现 `Rank 0~3`（环境变量缺失导致误触发4卡）
- 成功重跑日志: `work_dir/A3_20260212-002344_bs1.log`
- 结束日志证据: `[Rank 0/1] Epoch 199: 100%`，`Time: 20260213-0501, Epoch: 199`
- 末轮损失（日志）: `0.0339833474` / `0.0338141891`（rank1/rank0）
- 错误检索结果: `grep -E "Epoch 199|OutOfMemory|MASTER_ADDR|Traceback"` 仅命中 `Epoch 199`，未出现 OOM/Traceback
- 成功产物目录: `work_dir/MedSAM-FLARE22-A3-BalanceLoss-20260212-002344-20260212-0023`
- 关键文件: `medsam_model_best.pth`, `medsam_model_latest.pth`, `medsam_model_latest_step.pth`, `*_train_loss.png`
- 资源侧证据（训练中）: `nvidia-smi` 显示 `medsam` 两个训练进程位于 GPU1/GPU3，显存占用约 12GB/卡
- 评估汇总（服务器）: `work_dir/eval_metrics/A3_summary.json`
- 指标（40例）: DSC=`0.903470`，HD95=`7.922879`，ASD=`0.886811`

#### 阶段分析
- Stage 1 (Epoch 0-49) 表现: 训练稳定，未见中断报错（待结合指标判断早期收敛质量）
- Stage 2 (Epoch 50+) 表现: 完整跑通至 Epoch 199，loss持续下降
- 切换时机是否合适: 当前配置下整体指标劣于 A1/A2，需复查 `stage_switch_epoch` 与 α/β 权重设置

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
| + Inter-CBL | 0.9406 | 4.7905 | - | - |
| + Intra-CBL | 0.9526 | 3.3684 | - | - |
| + Balance Loss | 0.9035 | 7.9229 | - | - |
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

### 问题1: A3 首轮训练 OOM
- **日期**: 2026-02-12
- **现象**: `torch.cuda.OutOfMemoryError`，注意力计算阶段显存不足
- **原因分析**: A3 完整 Balance Loss 组合下，默认 `batch_size=2` 显存峰值超限
- **解决方案**: 将 A3 重跑参数调整为 `-batch_size 1`，其余超参保持不变
- **状态**: 已解决

### 问题2: A3 重启时分布式环境变量缺失
- **日期**: 2026-02-12
- **现象**: 日志报错 `MASTER_ADDR expected, but not set`，并出现 `Rank 0~3`
- **原因分析**: 启动命令拼接不规范，关键环境变量未显式设置，导致进程组配置错误
- **解决方案**: 使用标准化模板启动（先设置 `MASTER_ADDR/MASTER_PORT/CUDA_VISIBLE_DEVICES`，再 `nohup python ...`）
- **状态**: 已解决

---

## 下一步计划

- [x] 回填 A1/A2/A3 的 DSC/HD95/ASD（以服务器评估结果为准）
- [x] 输出 A1/A2/A3 对比表并补齐 `主实验结果` 表
- [ ] 补跑 Baseline 同口径评估（用于计算 A1/A2/A3 的 ΔBaseline）
- [ ] 启动 A3 修正实验（优先: `stage_switch_epoch`、α/β 比例、阈值）
- [ ] 启动 EXP-005（AttentionCrossBlock only）并建立日志与产物索引
- [ ] 在 EXP-006 启动前完成 Attention 模块的最小可运行验证（小规模试跑）

---

**文档结束**

> 最后更新: 2026-02-13
