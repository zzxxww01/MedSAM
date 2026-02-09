# MedSAM改进项目 - 毕业论文完整规划

> **目标**: 产出一篇完整的硕士毕业论文
> **主题**: 基于平衡损失与注意力融合的医学图像分割方法研究

---

## 第一部分：论文结构与实验对应关系

### 论文整体结构

```
论文标题: 基于平衡损失与注意力特征融合的医学图像分割方法研究
         (Research on Medical Image Segmentation Method Based on
          Balanced Loss and Attention Feature Fusion)

摘要 (300-500字)
Abstract (英文摘要)

第一章 绪论 ........................... 需要: 文献调研
  1.1 研究背景与意义
  1.2 国内外研究现状
  1.3 现有方法的问题与挑战
  1.4 本文研究内容与创新点
  1.5 论文组织结构

第二章 相关技术与理论基础 ............. 需要: 理论理解
  2.1 医学图像分割概述
  2.2 深度学习分割方法
  2.3 Segment Anything Model (SAM)
  2.4 类别不平衡问题与解决方案
  2.5 注意力机制
  2.6 本章小结

第三章 基于平衡损失的医学图像分割 ..... 需要: 实验A1-A4
  3.1 问题分析：双重类别不平衡
  3.2 类别间平衡损失 (Inter-CBL)
  3.3 类别内平衡损失 (Intra-CBL)
  3.4 两阶段训练策略
  3.5 实验验证与分析
  3.6 本章小结

第四章 基于注意力的特征融合机制 ....... 需要: 实验B1-B3
  4.1 特征融合问题分析
  4.2 交叉注意力融合模块设计
  4.3 支持集机制设计
  4.4 实验验证与分析
  4.5 本章小结

第五章 完整方法与综合实验 ............. 需要: 实验C1-C5
  5.1 整体框架设计
  5.2 实验设置
  5.3 消融实验
  5.4 对比实验
  5.5 跨数据集泛化实验
  5.6 可视化分析
  5.7 本章小结

第六章 总结与展望
  6.1 工作总结
  6.2 研究不足
  6.3 未来展望

参考文献
致谢
附录（可选）
```

---

## 第二部分：分阶段详细规划

### 阶段总览

| 阶段 | 时间 | 核心任务 | 论文产出 | 实验产出 |
|------|------|----------|----------|----------|
| 阶段1 | 1月中-1月底 | 基线复现+文献调研 | 第1-2章初稿 | Baseline结果 |
| 阶段2 | 2月1日-2月15日 | Balance Loss实现 | 第3章初稿 | 实验A1-A4 |
| 阶段3 | 2月16日-2月28日 | Attention模块实现 | 第4章初稿 | 实验B1-B3 |
| 阶段4 | 3月1日-3月15日 | 完整实验+论文整合 | 第5章+全文整合 | 实验C1-C5 |
| 阶段5 | 3月16日-4月15日 | 论文修改完善 | 终稿 | 补充实验 |
| 阶段6 | 4月16日-5月 | 查重+答辩 | 定稿 | - |

---

## 阶段1：基线复现与文献调研（1月中-1月底）

### 1.1 本阶段目标

```
目标1: 成功运行MedSAM，获得Baseline性能数据
目标2: 完成文献调研，理解研究现状
目标3: 完成论文第1-2章初稿
```

### 1.2 具体任务清单

#### Week 1: 环境搭建与基线复现

| 天数 | 任务 | 产出物 | 验收标准 |
|------|------|--------|----------|
| Day 1-2 | 环境配置 | 可运行环境 | 能import所有依赖 |
| Day 3 | 下载预训练权重 | sam_vit_b.pth | 文件完整 |
| Day 4 | 准备FLARE22数据 | data/npy/CT_Abd/ | 数据格式正确 |
| Day 5-6 | 运行Baseline训练 | 训练日志+模型 | 训练loss下降 |
| Day 7 | 评估Baseline性能 | DSC, HD95数值 | 记录到实验表 |

**Baseline训练命令**:
```bash
python train_one_gpu.py \
    -i data/npy/CT_Abd \
    -task_name Baseline-FLARE22 \
    -model_type vit_b \
    -checkpoint work_dir/SAM/sam_vit_b_01ec64.pth \
    -num_epochs 200 \
    -batch_size 4 \
    -lr 0.0001 \
    -use_wandb True
```

**需要记录的Baseline结果**:
| 器官 | DSC (%) | HD95 (mm) |
|------|---------|-----------|
| 肝脏 | - | - |
| 脾脏 | - | - |
| 胰腺 | - | - |
| 肾脏 | - | - |
| ... | ... | ... |
| **平均** | **-** | **-** |

#### Week 2: 文献调研与论文第1-2章

| 天数 | 任务 | 产出物 |
|------|------|--------|
| Day 1-2 | 调研医学图像分割方法 | 文献笔记20篇+ |
| Day 3-4 | 调研类别不平衡解决方案 | 文献笔记10篇+ |
| Day 5 | 调研SAM/MedSAM相关工作 | 文献笔记10篇+ |
| Day 6-7 | 撰写第1章绪论 | 5000字初稿 |
| Day 8-9 | 撰写第2章理论基础 | 6000字初稿 |

**必读核心文献**:

```
医学图像分割:
[1] U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)
[2] nnU-Net: A Self-Configuring Method for Deep Learning-based Biomedical Image Segmentation (2021)
[3] TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation (2021)

SAM相关:
[4] Segment Anything (2023)
[5] MedSAM: Segment Anything in Medical Images (2024)
[6] SAM-Med2D (2023)

类别不平衡:
[7] Focal Loss for Dense Object Detection (2017)
[8] Class-Balanced Loss Based on Effective Number of Samples (2019)
[9] Dice Loss and Cross Entropy Loss for Medical Image Segmentation (2020)

注意力机制:
[10] Attention Is All You Need (2017)
[11] CBAM: Convolutional Block Attention Module (2018)
[12] UniverSeg: Universal Medical Image Segmentation (2023)
```

### 1.3 阶段1产出清单

```
□ Baseline模型权重: work_dir/Baseline-FLARE22/medsam_model_best.pth
□ Baseline性能记录: docs/EXPERIMENT_LOG.md 中 EXP-001
□ 文献阅读笔记: docs/literature_notes/
□ 论文第1章初稿: thesis/chapter1_introduction.docx (5000字)
□ 论文第2章初稿: thesis/chapter2_background.docx (6000字)
```

---

## 阶段2：Balance Loss实现与验证（2月1日-2月15日）

### 2.1 本阶段目标

```
目标1: 完成Balance Loss代码实现
目标2: 通过消融实验验证每个组件的有效性
目标3: 完成论文第3章初稿
```

### 2.2 具体任务清单

#### Week 3: Balance Loss代码实现

| 天数 | 任务 | 产出物 | 验收标准 |
|------|------|--------|----------|
| Day 1 | 实现Inter-CBL | balance_loss.py | 单元测试通过 |
| Day 2 | 实现Intra-CBL | balance_loss.py | 单元测试通过 |
| Day 3 | 实现完整BalanceLoss | balance_loss.py | 单元测试通过 |
| Day 4 | 集成到训练脚本 | train_balance_loss.py | 能正常训练 |
| Day 5-7 | 运行实验A1-A3 | 实验结果 | 记录到表格 |

**代码实现要点**:

```python
# Inter-CBL核心逻辑
def inter_cbl(pred, target):
    # 1. 统计前景数量
    fg_count = (target == 1).sum()

    # 2. 在背景中找最难的fg_count个样本
    bg_probs = sigmoid(pred[target == 0])
    hard_bg_idx = bg_probs.topk(fg_count).indices

    # 3. 平衡计算损失
    fg_loss = BCE(pred[fg_mask], target[fg_mask])
    hard_bg_loss = BCE(pred[hard_bg_idx], target[hard_bg_idx])
    return (fg_loss + hard_bg_loss) / 2

# Intra-CBL核心逻辑
def intra_cbl(pred, target, threshold=0.9):
    # 1. 计算每个像素的"正确性"
    prob = sigmoid(pred)
    correctness = target * prob + (1-target) * (1-prob)

    # 2. 划分难易样本
    easy = (correctness > threshold)
    hard = ~easy

    # 3. 加权损失
    return easy_weight * loss[easy].mean() + hard_weight * loss[hard].mean()
```

#### Week 4: 消融实验与论文第3章

**实验设计**:

| 实验ID | 配置 | 目的 | 预期结果 |
|--------|------|------|----------|
| A1 | Dice+CE (Baseline) | 对照基准 | - |
| A2 | **Inter-CBL** + Dice | 验证类别间平衡 | 小目标DSC提升 |
| A3 | **Intra-CBL** + Dice | 验证类别内平衡 | 边界精度提升 |
| A4 | **Balance Loss完整** | 验证组合效果 | 综合最优 |
| A5 | 不同α,β,γ参数 | 超参数敏感性 | 找到最优参数 |
| A6 | 不同切换时机 | 两阶段策略验证 | 找到最优切换点 |

**实验运行命令**:

```bash
# A1: Baseline
python train_one_gpu.py -task_name A1-Baseline ...

# A2: Inter-CBL
python train_balance_loss.py -task_name A2-InterCBL \
    -loss_type balance -balance_alpha 1.0 -balance_beta 0.0 ...

# A3: Intra-CBL
python train_balance_loss.py -task_name A3-IntraCBL \
    -loss_type balance -balance_alpha 0.0 -balance_beta 1.0 ...

# A4: Full Balance Loss
python train_balance_loss.py -task_name A4-BalanceLoss \
    -loss_type balance -balance_alpha 1.0 -balance_beta 1.0 ...
```

**论文第3章结构**:

```
第三章 基于平衡损失的医学图像分割方法 (约8000字)

3.1 问题分析 (1500字)
    3.1.1 类别间不平衡现象
          - 用统计数据说明：前景占比通常<1%
          - 图表：不同器官的前景/背景比例分布
    3.1.2 类别内不平衡现象
          - 用可视化说明：边界像素vs内部像素的难度差异
          - 图表：预测置信度分布图

3.2 类别间平衡损失 (Inter-CBL) (1500字)
    3.2.1 设计动机
    3.2.2 困难样本挖掘策略
    3.2.3 数学形式化
          - 公式推导
    3.2.4 与现有方法对比
          - 与Focal Loss的区别

3.3 类别内平衡损失 (Intra-CBL) (1500字)
    3.3.1 设计动机
    3.3.2 难度评估策略
    3.3.3 加权损失设计
    3.3.4 与OHEM的对比

3.4 两阶段训练策略 (1000字)
    3.4.1 策略设计
    3.4.2 切换时机选择

3.5 实验验证 (2000字)
    3.5.1 实验设置
    3.5.2 消融实验结果 (表格：A1-A4结果)
    3.5.3 超参数分析 (表格+图：不同参数的影响)
    3.5.4 可视化分析 (图：分割结果对比)

3.6 本章小结 (500字)
```

### 2.3 阶段2产出清单

```
□ 代码: losses/balance_loss.py
□ 代码: losses/__init__.py
□ 代码: train_balance_loss.py (或修改后的train_one_gpu.py)
□ 实验结果: A1-A6的DSC, HD95数据
□ 可视化: 分割结果对比图 (至少5个case)
□ 可视化: 损失曲线图
□ 论文第3章初稿: thesis/chapter3_balance_loss.docx (8000字)
```

**第3章需要的图表**:

| 图表编号 | 内容 | 用途 |
|----------|------|------|
| 图3-1 | 类别不平衡统计图 | 说明问题 |
| 图3-2 | Inter-CBL示意图 | 方法说明 |
| 图3-3 | Intra-CBL示意图 | 方法说明 |
| 图3-4 | 两阶段训练曲线 | 策略验证 |
| 图3-5 | 分割结果可视化对比 | 定性分析 |
| 表3-1 | 消融实验结果 | 定量分析 |
| 表3-2 | 超参数敏感性分析 | 参数选择 |

---

## 阶段3：注意力模块实现与验证（2月16日-2月28日）

### 3.1 本阶段目标

```
目标1: 完成AttentionCrossBlock代码实现
目标2: 设计并实现Few-Shot数据加载器
目标3: 通过实验验证注意力机制的有效性
目标4: 完成论文第4章初稿
```

### 3.2 具体任务清单

#### Week 5: 注意力模块实现

| 天数 | 任务 | 产出物 |
|------|------|--------|
| Day 1-2 | 实现AttentionCrossBlock | attention_cross_block.py |
| Day 3 | 实现SupportAggregator | attention_cross_block.py |
| Day 4 | 实现FewShotDataset | fewshot_dataset.py |
| Day 5 | 实现MedSAM_FSS模型 | medsam_fss.py |
| Day 6-7 | 集成测试 | 完整训练流程 |

**代码实现要点**:

```python
# AttentionCrossBlock核心逻辑
class AttentionCrossBlock(nn.Module):
    def forward(self, query, support):
        """
        query: [B, C, H, W] 查询图像特征
        support: [B, N, C, H, W] N个支持样本特征
        """
        # 展平
        Q = query.flatten(2).transpose(1,2)  # [B, HW, C]
        K = support.flatten(3)  # [B, N, C, HW]
        V = K

        # 多头注意力
        attn = softmax(Q @ K.T / sqrt(d))  # [B, HW, N*HW]
        out = attn @ V  # [B, HW, C]

        return out.reshape(B, C, H, W)

# Few-Shot数据加载
class FewShotDataset:
    def __getitem__(self, idx):
        query_img, query_gt = load(idx)
        support_indices = sample_same_class(idx, k=5)
        support_imgs = [load(i)[0] for i in support_indices]
        support_gts = [load(i)[1] for i in support_indices]

        return {
            'query_image': query_img,
            'query_gt': query_gt,
            'support_images': stack(support_imgs),
            'support_gts': stack(support_gts)
        }
```

#### Week 6: 实验与论文第4章

**实验设计**:

| 实验ID | 配置 | 目的 |
|--------|------|------|
| B1 | MedSAM + Attention (N=1) | 单支持样本 |
| B2 | MedSAM + Attention (N=5) | 多支持样本 |
| B3 | MedSAM + Attention (N=10) | 更多支持样本 |
| B4 | 不同注意力头数 | 超参数分析 |
| B5 | 有/无mask加权 | 消融分析 |

**论文第4章结构**:

```
第四章 基于注意力的特征融合机制 (约6000字)

4.1 问题分析 (1000字)
    4.1.1 现有特征融合方法的局限性
    4.1.2 支持集质量差异问题

4.2 交叉注意力融合模块 (2000字)
    4.2.1 模块设计
          - 图：模块结构示意图
    4.2.2 多头注意力机制
    4.2.3 位置编码设计
    4.2.4 与自注意力的区别

4.3 支持集机制设计 (1000字)
    4.3.1 支持集采样策略
    4.3.2 支持集聚合方法

4.4 实验验证 (1500字)
    4.4.1 实验设置
    4.4.2 不同支持样本数量的影响
    4.4.3 注意力可视化分析
    4.4.4 计算开销分析

4.5 本章小结 (500字)
```

### 3.3 阶段3产出清单

```
□ 代码: modules/attention_cross_block.py
□ 代码: modules/__init__.py
□ 代码: datasets/fewshot_dataset.py
□ 代码: models/medsam_fss.py
□ 代码: train_fss.py
□ 实验结果: B1-B5的DSC, HD95数据
□ 可视化: 注意力权重热力图
□ 论文第4章初稿: thesis/chapter4_attention.docx (6000字)
```

---

## 阶段4：完整实验与论文整合（3月1日-3月15日）

### 4.1 本阶段目标

```
目标1: 完成所有综合实验（对比实验、跨数据集实验）
目标2: 完成论文第5章
目标3: 整合全文，形成完整论文初稿
```

### 4.2 具体任务清单

#### Week 7: 综合实验

**实验设计**:

| 实验ID | 内容 | 数据集 |
|--------|------|--------|
| C1 | 最终消融实验 | FLARE22 |
| C2 | 与MedSAM原始对比 | FLARE22 |
| C3 | 与nnU-Net对比 | FLARE22 |
| C4 | 与其他方法对比 | FLARE22 |
| C5 | 跨数据集泛化 | KiTS19, BUSI |

**完整消融实验表**:

| 方法 | Inter-CBL | Intra-CBL | Attention | DSC↑ | HD95↓ |
|------|:---------:|:---------:|:---------:|------|-------|
| Baseline | | | | - | - |
| +Inter-CBL | ✓ | | | - | - |
| +Intra-CBL | | ✓ | | - | - |
| +Balance Loss | ✓ | ✓ | | - | - |
| +Attention | | | ✓ | - | - |
| **Ours (Full)** | ✓ | ✓ | ✓ | - | - |

**对比实验表**:

| 方法 | 年份 | DSC (%) | HD95 (mm) | 参数量 |
|------|------|---------|-----------|--------|
| U-Net | 2015 | - | - | ~7M |
| nnU-Net | 2021 | - | - | ~30M |
| TransUNet | 2021 | - | - | ~105M |
| SAM | 2023 | - | - | ~308M |
| MedSAM | 2024 | - | - | ~93M |
| **Ours** | 2025 | - | - | ~95M |

#### Week 8: 论文整合

**论文第5章结构**:

```
第五章 综合实验与分析 (约8000字)

5.1 实验设置 (1500字)
    5.1.1 数据集介绍
          - 表：各数据集统计信息
    5.1.2 评价指标
    5.1.3 实现细节
    5.1.4 对比方法

5.2 消融实验 (2000字)
    5.2.1 各组件贡献分析
          - 表：完整消融实验结果
    5.2.2 超参数分析
    5.2.3 训练策略分析

5.3 对比实验 (2000字)
    5.3.1 与现有方法对比
          - 表：主要对比结果
    5.3.2 不同器官的性能分析
    5.3.3 计算效率对比

5.4 泛化性实验 (1500字)
    5.4.1 跨数据集测试
    5.4.2 跨模态测试
          - 表：泛化实验结果

5.5 可视化分析 (1000字)
    5.5.1 分割结果可视化
    5.5.2 注意力权重可视化
    5.5.3 失败案例分析
```

### 4.3 阶段4产出清单

```
□ 实验结果: C1-C5完整数据
□ 可视化: 所有对比图、表格
□ 论文第5章: thesis/chapter5_experiments.docx (8000字)
□ 论文第6章: thesis/chapter6_conclusion.docx (2000字)
□ 摘要: thesis/abstract.docx (中英文)
□ 完整论文初稿: thesis/full_thesis_v1.docx
```

---

## 阶段5：论文修改与完善（3月16日-4月15日）

### 5.1 本阶段目标

```
目标1: 根据导师反馈修改论文
目标2: 完善实验，补充遗漏数据
目标3: 论文格式规范化
目标4: 准备查重
```

### 5.2 具体任务

#### Week 9-10: 导师初审修改

| 任务 | 说明 |
|------|------|
| 提交初稿给导师 | 3月16日 |
| 收集反馈意见 | 等待3-5天 |
| 修改论文内容 | 根据反馈逐条修改 |
| 补充实验 | 如有新要求 |
| 完善图表 | 提高专业性 |

#### Week 11-12: 格式规范化

| 任务 | 检查点 |
|------|--------|
| 格式检查 | 符合学校模板 |
| 参考文献 | 格式统一，50篇以上 |
| 图表规范 | 清晰、有标号、有说明 |
| 公式编号 | 按章节编号 |
| 语言润色 | 通顺、专业 |

### 5.3 论文检查清单

```
内容完整性:
□ 摘要包含背景、方法、结果、结论
□ 每章有引言和小结
□ 实验部分有完整的设置说明
□ 结论与摘要呼应

格式规范:
□ 页眉页脚正确
□ 目录自动生成
□ 图表编号连续
□ 参考文献格式统一
□ 字体字号符合要求

学术规范:
□ 引用标注完整
□ 无抄袭内容
□ 图片来源说明
□ 数据真实可复现
```

---

## 阶段6：查重、盲审与答辩（4月16日-5月）

### 6.1 查重准备

```
提前自查:
- 使用知网/维普自查
- 重复率目标: <15%
- 重点检查: 相关工作部分

降重技巧:
- 用自己的话重新表述
- 增加原创性分析
- 合理引用并标注
```

### 6.2 盲审准备

```
盲审版本要求:
- 去除个人信息
- 去除致谢
- 保留技术内容
```

### 6.3 答辩准备

**PPT结构** (20-25页):

```
1. 封面 (1页)
2. 目录 (1页)
3. 研究背景与意义 (2-3页)
4. 相关工作 (2页)
5. 创新点1: Balance Loss (4-5页)
6. 创新点2: Attention (3-4页)
7. 实验结果 (5-6页)
8. 结论与展望 (2页)
9. 致谢 (1页)
```

**答辩常见问题准备**:

```
1. 为什么选择MedSAM作为基础模型？
2. Balance Loss与Focal Loss的区别是什么？
3. 两阶段训练策略的切换依据是什么？
4. 注意力模块的计算复杂度如何？
5. 方法的局限性是什么？
6. 未来可以如何改进？
```

---

## 第三部分：实验详细设计

### 实验矩阵总表

| ID | 名称 | 类型 | 对应论文章节 | 优先级 |
|----|------|------|--------------|--------|
| A1 | Baseline | 基线 | 3.5, 5.2 | 必须 |
| A2 | Inter-CBL | 消融 | 3.5 | 必须 |
| A3 | Intra-CBL | 消融 | 3.5 | 必须 |
| A4 | Balance Loss | 消融 | 3.5, 5.2 | 必须 |
| A5 | 超参数α,β,γ | 分析 | 3.5 | 重要 |
| A6 | 切换时机 | 分析 | 3.5 | 重要 |
| B1 | Attention N=1 | 消融 | 4.4 | 必须 |
| B2 | Attention N=5 | 消融 | 4.4, 5.2 | 必须 |
| B3 | Attention N=10 | 消融 | 4.4 | 重要 |
| B4 | 注意力头数 | 分析 | 4.4 | 可选 |
| C1 | 完整消融 | 消融 | 5.2 | 必须 |
| C2 | vs MedSAM | 对比 | 5.3 | 必须 |
| C3 | vs nnU-Net | 对比 | 5.3 | 必须 |
| C4 | vs 其他方法 | 对比 | 5.3 | 重要 |
| C5 | 跨数据集 | 泛化 | 5.4 | 必须 |

### 实验结果记录模板

```markdown
## 实验 [ID]: [名称]

### 配置
- 日期: YYYY-MM-DD
- GPU:
- 训练时长:
- 超参数:

### 定量结果
| 指标 | 值 |
|------|-----|
| DSC  | xx.x% |
| HD95 | xx.x mm |

### 定性结果
- 收敛曲线: [截图]
- 分割示例: [截图]

### 分析
- 观察:
- 结论:
```

---

## 第四部分：论文字数规划

| 章节 | 字数 | 状态 |
|------|------|------|
| 摘要(中) | 500 | 待写 |
| 摘要(英) | 300词 | 待写 |
| 第一章 绪论 | 5,000 | 待写 |
| 第二章 理论基础 | 6,000 | 待写 |
| 第三章 Balance Loss | 8,000 | 待写 |
| 第四章 Attention | 6,000 | 待写 |
| 第五章 综合实验 | 8,000 | 待写 |
| 第六章 总结展望 | 2,000 | 待写 |
| 参考文献 | 50篇+ | 待整理 |
| **总计** | **~35,000** | - |

---

## 第五部分：风险预案

### 实验未达预期

| 情况 | 应对 |
|------|------|
| Balance Loss效果不显著 | 1. 调整超参数 2. 分析具体哪类样本改善/退化 3. 如实报告并分析原因 |
| Attention计算量过大 | 1. 使用高效注意力 2. 减少支持样本数 3. 分析效率-性能权衡 |
| 泛化性不佳 | 1. 增加数据增强 2. 分析失败案例 3. 讨论方法局限性 |

### 时间不足

| 情况 | 应对 |
|------|------|
| 代码开发延迟 | 简化实现，先保证核心功能 |
| 实验时间不足 | 减少超参数搜索范围 |
| 论文撰写延迟 | 边实验边写，每完成一个实验立即记录 |

---

## 检查清单总表

### 阶段1完成检查
- [ ] Baseline训练完成并记录结果
- [ ] 40+篇文献阅读笔记
- [ ] 第1章初稿完成
- [ ] 第2章初稿完成

### 阶段2完成检查
- [ ] Balance Loss代码完成并测试
- [ ] 实验A1-A6完成并记录
- [ ] 第3章初稿完成
- [ ] 相关可视化图表制作完成

### 阶段3完成检查
- [ ] Attention模块代码完成
- [ ] Few-Shot训练流程完成
- [ ] 实验B1-B5完成并记录
- [ ] 第4章初稿完成

### 阶段4完成检查
- [ ] 所有实验C1-C5完成
- [ ] 第5章完成
- [ ] 第6章完成
- [ ] 全文初稿整合完成

### 阶段5完成检查
- [ ] 导师审阅反馈处理完成
- [ ] 格式规范化完成
- [ ] 查重通过 (<15%)

### 阶段6完成检查
- [ ] 盲审通过
- [ ] 答辩PPT完成
- [ ] 答辩准备充分

---

**文档结束**

> 按照此规划执行，可确保产出一篇完整、规范的硕士毕业论文
