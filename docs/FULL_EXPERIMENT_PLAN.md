# MedSAM Balance Loss 完整实验计划

## 一、研究背景与目标

### 1.1 研究问题
医学图像分割中存在的核心问题：
- **类别不平衡**：前景（器官）像素远少于背景像素
- **样本难度不均**：边界区域比内部区域更难分割
- **小目标问题**：小器官（如肾上腺）容易被忽略

### 1.2 研究目标
通过 Balance Loss 解决上述问题，提升 MedSAM 在多器官分割任务上的性能。

### 1.3 预期贡献
1. 提出 Balance Loss 损失函数
2. 在 AMOS22 数据集上验证有效性
3. 通过消融实验分析各组件贡献

---

## 二、实验总体流程

```
阶段1: 数据准备 → 阶段2: Baseline实验 → 阶段3: 消融实验 → 阶段4: 结果分析
```

| 阶段 | 目的 | 产出 |
|------|------|------|
| 阶段1 | 准备AMOS22数据集 | `data/npy/CT_AMOS/` |
| 阶段2 | 建立性能基准 | Baseline模型 + 指标 |
| 阶段3 | 验证各组件效果 | 4组实验结果 |
| 阶段4 | 分析与总结 | 论文实验部分 |

---

## 三、阶段1：数据准备

### 3.1 目的
将 AMOS22 原始 NIfTI 格式转换为 MedSAM 训练所需的 NPY 格式。

### 3.2 AMOS22 数据集信息
- **来源**: MICCAI 2022 Challenge
- **规模**: 500例CT + 100例MRI
- **标签**: 15个腹部器官
- **我们使用**: 仅CT图像部分

### 3.3 操作步骤

**服务器操作：**
```bash
# 1. 拉取最新代码
cd ~/chengang/zxw/MedSAM
git pull

# 2. 运行预处理
python pre_AMOS22.py
```

### 3.4 预期输出
```
data/npy/CT_AMOS/
├── imgs/    # ~50000+ 张 1024x1024 RGB图像
└── gts/     # 对应的分割标签
```

### 3.5 验证方法
```bash
# 检查生成的文件数量
ls data/npy/CT_AMOS/imgs/ | wc -l
ls data/npy/CT_AMOS/gts/ | wc -l
```

---

## 四、阶段2：Baseline实验 (A0)

### 4.1 目的
建立性能基准，作为后续改进的对比参照。

### 4.2 实验配置
| 参数 | 值 |
|------|-----|
| 损失函数 | Dice + CE (原始MedSAM) |
| 数据集 | CT_AMOS |
| Epochs | 200 |
| Batch Size | 2 |
| Learning Rate | 0.0001 |
| GPUs | 2 |

### 4.3 操作步骤

**服务器操作：**
```bash
CUDA_VISIBLE_DEVICES=0,1 \
MASTER_ADDR=localhost \
MASTER_PORT=12355 \
MPLBACKEND=Agg \
python train_multi_gpus.py \
    -i data/npy/CT_AMOS \
    -task_name MedSAM-AMOS-A0-Baseline \
    -model_type vit_b \
    -checkpoint work_dir/medsam_vit_b.pth \
    -num_epochs 200 \
    -batch_size 2 \
    -lr 0.0001 \
    --world_size 2 \
    -use_amp
```

### 4.4 预期产出
- 模型文件: `work_dir/MedSAM-AMOS-A0-Baseline-*/medsam_model_best.pth`
- 损失曲线: `*train_loss.png`

### 4.5 为什么需要这一步？
- 验证数据预处理正确性
- 获得基准性能指标
- 为后续实验提供对比基线

---

## 五、阶段3：消融实验

### 5.1 实验设计总览

| 实验ID | 损失函数 | 目的 | 验证假设 |
|--------|----------|------|----------|
| A0 | Dice + CE | Baseline | - |
| A1 | Inter-CBL + Dice | 类别间平衡 | 困难背景挖掘有效 |
| A2 | Intra-CBL + Dice | 类别内平衡 | 困难样本加权有效 |
| A3 | Balance Loss | 完整方案 | 组合效果最优 |

### 5.2 实验A1：Inter-CBL + Dice

**目的：** 验证类别间平衡（前景vs背景）的效果

**核心思想：**
- 挖掘困难背景样本（被误判为前景的背景像素）
- 使前景和背景的损失贡献相等

**操作命令：**
```bash
CUDA_VISIBLE_DEVICES=0,1 \
MASTER_ADDR=localhost \
MASTER_PORT=12356 \
MPLBACKEND=Agg \
python train_multi_gpus_balance.py \
    -i data/npy/CT_AMOS \
    -task_name MedSAM-AMOS-A1-InterCBL \
    -model_type vit_b \
    -checkpoint work_dir/medsam_vit_b.pth \
    -num_epochs 200 \
    -batch_size 2 \
    -lr 0.0001 \
    --world_size 2 \
    -use_amp \
    -loss_type inter_cbl
```

**预期效果：** 减少假阳性，提高精确率

### 5.3 实验A2：Intra-CBL + Dice

**目的：** 验证类别内平衡（简单vs困难样本）的效果

**核心思想：**
- 根据预测置信度划分难易样本
- 困难样本（边界区域）给予更高权重

**操作命令：**
```bash
CUDA_VISIBLE_DEVICES=0,1 \
MASTER_ADDR=localhost \
MASTER_PORT=12357 \
MPLBACKEND=Agg \
python train_multi_gpus_balance.py \
    -i data/npy/CT_AMOS \
    -task_name MedSAM-AMOS-A2-IntraCBL \
    -model_type vit_b \
    -checkpoint work_dir/medsam_vit_b.pth \
    -num_epochs 200 \
    -batch_size 2 \
    -lr 0.0001 \
    --world_size 2 \
    -use_amp \
    -loss_type intra_cbl
```

**预期效果：** 改善边界分割质量

### 5.4 实验A3：完整Balance Loss

**目的：** 验证完整方案的综合效果

**核心思想：**
- 结合 Inter-CBL 和 Intra-CBL
- 两阶段训练策略：
  - Stage 1 (Epoch 0-50): Intra-CBL + Dice（稳定启动）
  - Stage 2 (Epoch 50+): Inter-CBL + Intra-CBL + Dice（完整损失）

**操作命令：**
```bash
CUDA_VISIBLE_DEVICES=0,1 \
MASTER_ADDR=localhost \
MASTER_PORT=12358 \
MPLBACKEND=Agg \
python train_multi_gpus_balance.py \
    -i data/npy/CT_AMOS \
    -task_name MedSAM-AMOS-A3-BalanceLoss \
    -model_type vit_b \
    -checkpoint work_dir/medsam_vit_b.pth \
    -num_epochs 200 \
    -batch_size 2 \
    -lr 0.0001 \
    --world_size 2 \
    -use_amp \
    -loss_type balance \
    -stage1_epochs 50
```

**预期效果：** 综合提升，DSC最高

---

## 六、阶段4：结果分析与评估

### 6.1 评估指标

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| DSC | Dice相似系数 | 2×|A∩B| / (|A|+|B|) |
| HD95 | 95%豪斯多夫距离 | 边界距离的95百分位 |
| ASD | 平均表面距离 | 边界点平均距离 |

### 6.2 预期结果表格

| 实验 | DSC ↑ | HD95 ↓ | ASD ↓ |
|------|-------|--------|-------|
| A0 (Baseline) | ~0.85 | ~15 | ~3.0 |
| A1 (Inter-CBL) | ~0.86 | ~14 | ~2.8 |
| A2 (Intra-CBL) | ~0.86 | ~13 | ~2.7 |
| A3 (Balance) | ~0.88 | ~12 | ~2.5 |

### 6.3 分析要点

1. **A1 vs A0**: Inter-CBL是否减少假阳性？
2. **A2 vs A0**: Intra-CBL是否改善边界？
3. **A3 vs A1/A2**: 组合是否优于单独使用？
4. **各器官分析**: 小器官（肾上腺）提升是否更明显？

---

## 七、执行时间线

### 7.1 推荐执行顺序

```
Day 1: 数据预处理 + 启动A0
Day 2-3: A0训练完成，启动A1
Day 3-4: A1训练完成，启动A2
Day 4-5: A2训练完成，启动A3
Day 5-6: A3训练完成，开始评估
Day 7: 结果分析与总结
```

### 7.2 并行执行方案（如有多台服务器）

```
服务器1: A0 → A2
服务器2: A1 → A3
```

---

## 八、快速参考卡片

### 8.1 服务器登录后第一步
```bash
cd ~/chengang/zxw/MedSAM
git pull
conda activate medsam
```

### 8.2 查看训练进度
```bash
# 查看GPU使用
watch -n 1 nvidia-smi

# 查看最新日志
tail -f work_dir/MedSAM-AMOS-*/train.log
```

### 8.3 检查训练结果
```bash
# 列出所有实验
ls -la work_dir/ | grep AMOS

# 查看损失曲线
ls work_dir/MedSAM-AMOS-*/*loss.png
```

---

## 九、常见问题处理

| 问题 | 解决方案 |
|------|----------|
| CUDA OOM | 减小batch_size到1 |
| 端口占用 | 修改MASTER_PORT |
| 训练中断 | 使用--resume恢复 |
| 数据路径错误 | 检查data/npy/CT_AMOS是否存在 |

---

## 十、总结

本实验计划通过4组对照实验，系统验证Balance Loss各组件的有效性：

1. **Baseline (A0)** - 建立基准
2. **Inter-CBL (A1)** - 验证类别间平衡
3. **Intra-CBL (A2)** - 验证类别内平衡
4. **Balance Loss (A3)** - 验证完整方案

预期结论：Balance Loss通过同时解决类别不平衡和样本难度不均问题，能够显著提升医学图像分割性能。
