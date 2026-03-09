# 论文待办清单

> 最后更新：2026-03-03
> 状态：论文骨架完成（51页），代码冻结，待补充图片和细节打磨

---

## 📋 总览

| 类别 | 待办项 | 优先级 | 预计耗时 | 状态 |
|------|--------|--------|---------|------|
| 图片 | 8 幅图 | ⭐⭐⭐ | 4-6 小时 | ⏳ 未开始 |
| 论文 | 细节打磨 | ⭐⭐ | 2-3 小时 | ⏳ 未开始 |
| 答辩 | PPT 制作 | ⭐⭐ | 3-4 小时 | ⏳ 未开始 |

---

## 🎨 图片制作清单（8 幅）

### ⭐⭐⭐ 优先级最高（必须完成）

#### ✅ **Fig.4: LG-Adapter 架构示意图** (Ch4)

**位置**：`thesis-medsam/pages/chapter4.tex:60-65`（注释占位符）

**需要展示**：
```
输入 x [B,256,64,64]
    ↓
┌─────────────────┬─────────────────┐
│  DWConv 3×3     │  DWConv 3×3     │
│  dilation=1     │  dilation=2     │
│  groups=256     │  groups=256     │
│      ↓          │      ↓          │
│    GELU         │    GELU         │
│      ↓          │      ↓          │
│  f1 [256,64,64] │ f2 [256,64,64]  │
└────────┬────────┴────────┬────────┘
         │                 │
         └──── Concat ─────┘
               ↓
         [B,512,64,64]
               ↓
         Conv 1×1 (512→256)
               ↓
         f_out [B,256,64,64]
               ↓
         y = x + f_out (残差)
               ↓
         输出 [B,256,64,64]
```

**制作方式**：
- [ ] **方案A（推荐）**：PowerPoint 手绘
  - 使用形状工具绘制矩形框和箭头
  - 字体：Arial 或 Times New Roman，12-14pt
  - 颜色：黑白或浅蓝色系
  - 导出为 PDF（1200×800 像素）

- [ ] **方案B**：Python matplotlib 代码生成
  - 运行 `scripts/generate_fig4_adapter.py`（待创建）
  - 自动生成 `thesis-medsam/figures/lg_adapter_arch.pdf`

**参考资料**：
- MobileNet 论文 Fig.3（深度可分离卷积示意图）
- DeepLabV3+ 论文 Fig.2（ASPP 模块）

**保存路径**：`thesis-medsam/figures/lg_adapter_arch.pdf`

**验证**：在 `chapter4.tex:60` 取消注释 `\includegraphics` 行，重新编译 LaTeX

---

#### ✅ **Fig.1: 三核驱动技术路线总览图** (Ch1)

**位置**：`thesis-medsam/pages/chapter1.tex:40-45`（注释占位符）

**需要展示**：
```
┌─────────────────────────────────────────┐
│          研究问题                        │
│  • 双重不平衡（类间+类内）               │
│  • 高频特征丢失（ViT 平滑边缘）          │
└──────────────┬──────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  第一核：Balance Loss (损失层)           │
│  ─────────────────────────────────────   │
│  Inter-CBL + Intra-CBL + Dice            │
│  两阶段训练策略                          │
│  实验路径：A0→A1→A2→A3→A3R1/R2/R3        │
│  结果：DSC 0.941 → 0.960 (+2.0%)         │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  第二核：LoRA 消融 (架构消融层)          │
│  ─────────────────────────────────────   │
│  冻结 Image Encoder + LoRA 旁路          │
│  实验：C2                                │
│  结果：DSC 0.879 (退化 -8.3%)            │
│  结论：反证全参微调必要性 ✓              │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  第三核：LG-Adapter (特征流层)           │
│  ─────────────────────────────────────   │
│  双路径膨胀卷积 + 残差连接               │
│  实验：C3                                │
│  结果：DSC 0.962 (历史最优) ★            │
└──────────────────────────────────────────┘
```

**制作方式**：
- [ ] PowerPoint 手绘（推荐）
- [ ] draw.io / Figma

**保存路径**：`thesis-medsam/figures/tech_roadmap.pdf`

---

### ⭐⭐ 优先级中等（从数据生成）

#### ✅ **Fig.6: A0/A3R3/C2/C3 预测掩码对比** (Ch5)

**位置**：`thesis-medsam/pages/chapter5.tex:95-100`（注释占位符）

**数据来源**：评估时保存的预测结果

**生成步骤**：

**Step 1：检查数据是否存在**
```bash
# 在服务器上运行
ls work_dir/eval_results/A0/
ls work_dir/eval_results/A3R3/
ls work_dir/eval_results/C2/
ls work_dir/eval_results/C3/
```

**如果不存在**，需要重新运行评估并保存预测：
```bash
# 重新评估 A0（保存预测结果）
python eval_medsam_npz.py \
  --checkpoint work_dir/MedSAM-Baseline-20260208-1953/medsam_model_best.pth \
  --data_root data/npy/CT_Abd \
  --save_dir work_dir/eval_results/A0 \
  --save_pred

# 同理运行 A3R3, C2, C3
```

**Step 2：运行可视化脚本**
```bash
# 在服务器上运行
python scripts/visualize_masks.py \
  --case_id case_0001 \
  --output thesis-medsam/figures/mask_comparison.pdf
```

**我需要你提供的信息**：
- [ ] 评估结果的实际存储路径（`work_dir/eval_results/` 下是否有 NPZ 文件？）
- [ ] NPZ 文件的键名（`masks` 还是 `pred` 还是其他？）
- [ ] Ground Truth 的存储路径

**我会生成的脚本**：
- [ ] `scripts/visualize_masks.py`（完整可执行）

**保存路径**：`thesis-medsam/figures/mask_comparison.pdf`

---

#### ✅ **Fig.3: A0 vs A3R3 训练曲线对比** (Ch3)

**位置**：`thesis-medsam/pages/chapter3.tex:85-90`（注释占位符）

**数据来源**：训练日志文件

**生成步骤**：

**Step 1：确认日志格式**
```bash
# 在服务器上运行，查看日志格式
head -20 work_dir/A0_train.log
head -20 work_dir/exp_logs/A3R3_train.log
```

**我需要你提供的信息**：
- [ ] 日志中损失值的格式（给我看 1-2 行示例）
- [ ] 日志文件的实际路径

**示例日志格式**：
```
Epoch 1/200, Loss: 0.3456, Dice: 0.8234
Epoch 2/200, Loss: 0.2987, Dice: 0.8567
```

**Step 2：运行绘图脚本**
```bash
python scripts/plot_training_curves.py \
  --log_a0 work_dir/A0_train.log \
  --log_a3r3 work_dir/exp_logs/A3R3_train.log \
  --output thesis-medsam/figures/training_curves.pdf
```

**我会生成的脚本**：
- [ ] `scripts/plot_training_curves.py`（根据你提供的日志格式定制）

**保存路径**：`thesis-medsam/figures/training_curves.pdf`

---

### ⭐ 优先级较低（可选或简化）

#### ✅ **Fig.2: Balance Loss 组件流程图** (Ch3)

**位置**：`thesis-medsam/pages/chapter3.tex:50-55`（注释占位符）

**需要展示**：
```
输入：pred, target, epoch
    ↓
┌───────────────────────────────────┐
│  Dice Loss (始终启用)              │
│  L_dice = 1 - 2|Y∩Ŷ|/(|Y|+|Ŷ|)   │
└───────────────┬───────────────────┘
                ↓
┌───────────────────────────────────┐
│  Intra-CBL (始终启用)              │
│  按置信度误差划分难易样本          │
│  L_intra = w_e·L_easy + w_h·L_hard│
└───────────────┬───────────────────┘
                ↓
        epoch < 50?
         ↙     ↘
       Yes      No
        ↓       ↓
    Stage 1  Stage 2
    (稳定)   (强化)
        ↓       ↓
        │   ┌───────────────────────┐
        │   │  Inter-CBL (启用)      │
        │   │  挖掘困难背景样本      │
        │   │  L_inter = ...         │
        │   └───────────┬───────────┘
        └───────────────┘
                ↓
    L = α·L_inter + β·L_intra + γ·L_dice
```

**制作方式**：
- [ ] PowerPoint 手绘
- [ ] Python matplotlib 代码生成

**保存路径**：`thesis-medsam/figures/balance_loss_flow.pdf`

---

#### ✅ **Fig.5: Adapter 在 MedSAM 管线中的位置** (Ch4)

**位置**：`thesis-medsam/pages/chapter4.tex:70-75`（注释占位符）

**需要展示**：
```
输入图像 (1024×1024×3)
    ↓
┌─────────────────────────┐
│  Image Encoder (ViT)    │
│  输出：[B,256,64,64]     │
└──────────┬──────────────┘
           ↓
    ┌──────────────┐
    │ LG-Adapter   │  ← 插入位置
    │ (本文提出)   │
    └──────┬───────┘
           ↓
    [B,256,64,64] (增强后)
           ↓
┌──────────────────────────┐
│  Prompt Encoder          │
│  (Box → Embedding)       │
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│  Mask Decoder            │
│  (Cross-Attention Fusion)│
└──────────┬───────────────┘
           ↓
    输出掩码 (1024×1024)
```

**制作方式**：
- [ ] PowerPoint 手绘

**保存路径**：`thesis-medsam/figures/adapter_pipeline.pdf`

---

#### ✅ **Fig.7: 边界区域放大对比** (Ch5)

**位置**：`thesis-medsam/pages/chapter5.tex:105-110`（注释占位符）

**需要展示**：选择一个器官边界区域（如肝脏-胃交界），放大对比 A0/A3R3/C2/C3 的预测精度

**生成步骤**：
```bash
# 在服务器上运行
python scripts/visualize_boundary_detail.py \
  --case_id case_0001 \
  --organ liver \
  --zoom_region 200,200,100,100 \
  --output thesis-medsam/figures/boundary_detail.pdf
```

**我会生成的脚本**：
- [ ] `scripts/visualize_boundary_detail.py`

**保存路径**：`thesis-medsam/figures/boundary_detail.pdf`

---

#### ✅ **Fig.8: MedSAM 架构图** (Ch2, 可选)

**位置**：`thesis-medsam/pages/chapter2.tex:20-25`（注释占位符）

**制作方式**：
- [ ] **方案A（推荐）**：直接复用 SAM 论文原图
  - 下载 SAM 论文 PDF，截取 Fig.2
  - 标注 "Image Encoder (ViT)" / "Prompt Encoder" / "Mask Decoder"

- [ ] **方案B**：重新绘制简化版

**保存路径**：`thesis-medsam/figures/medsam_architecture.pdf`

---

## 📝 论文细节打磨清单

### ✅ **语句通顺性检查**

- [ ] 通读 Ch1-Ch6，检查是否有语句不通顺、逻辑跳跃的地方
- [ ] 检查专业术语的一致性（如 "Image Encoder" vs "图像编码器"）
- [ ] 检查数值引用是否与 `docs/EXPERIMENT_LOG.md` 一致

### ✅ **格式统一性检查**

- [ ] 检查所有表格的 caption 格式是否一致
- [ ] 检查所有公式的编号是否正确
- [ ] 检查所有引用的格式（`\cite{}`）

### ✅ **图表交叉引用**

- [ ] 确保所有图片都在正文中被引用（如 "如图~\ref{fig:xxx}所示"）
- [ ] 确保所有表格都在正文中被引用

### ✅ **参考文献检查**

- [ ] 确认 30 篇参考文献都在正文中被 `\cite{}` 引用
- [ ] 检查参考文献格式是否符合学校要求

---

## 🎤 答辩准备清单

### ✅ **PPT 制作**

**内容结构**：
1. 研究背景与问题（2-3 页）
2. 相关工作（1-2 页）
3. 第一核心：Balance Loss（3-4 页）
4. 第二核心：LoRA 消融（2-3 页）
5. 第三核心：LG-Adapter（3-4 页）
6. 实验结果（3-4 页）
7. 总结与展望（1-2 页）

**关键图表**：
- [ ] 技术路线图（Fig.1）
- [ ] LG-Adapter 架构图（Fig.4）
- [ ] 实验结果对比表（Table 5.1）
- [ ] 预测掩码对比图（Fig.6）

**预计页数**：20-25 页

### ✅ **答辩稿准备**

- [ ] 熟读 `docs/THESIS_KNOWLEDGE_BASE.md § 10 答辩Q&A`
- [ ] 准备 7 个核心问题的回答（每个 1-2 分钟）
- [ ] 准备演示 Demo（可选）

### ✅ **时间控制**

- [ ] 答辩陈述：10-15 分钟
- [ ] 提问环节：5-10 分钟

---

## 📅 时间规划建议

### 第 1 天（4-6 小时）：图片制作

- [ ] 上午：Fig.4 (LG-Adapter) + Fig.1 (技术路线)
- [ ] 下午：Fig.6 (掩码对比) + Fig.3 (训练曲线)

### 第 2 天（2-3 小时）：论文打磨

- [ ] 上午：语句通顺性 + 格式统一性
- [ ] 下午：交叉引用 + 参考文献

### 第 3 天（3-4 小时）：答辩准备

- [ ] 上午：PPT 制作
- [ ] 下午：答辩稿准备 + 模拟答辩

---

## 🚀 立即行动

### 现在就可以做的：

1. **告诉我以下信息**（我会生成对应脚本）：
   - [ ] 评估结果是否保存了 NPZ？路径格式？
   - [ ] 训练日志的格式示例（给我看 1-2 行）
   - [ ] Ground Truth 的存储路径

2. **我会立即生成**：
   - [ ] `scripts/visualize_masks.py`（Fig.6）
   - [ ] `scripts/plot_training_curves.py`（Fig.3）
   - [ ] `scripts/visualize_boundary_detail.py`（Fig.7）
   - [ ] `scripts/generate_fig4_adapter.py`（Fig.4，可选）

3. **你在服务器运行脚本**，下载生成的 PDF

4. **手绘图片**（Fig.1, Fig.4）：
   - 我提供 ASCII 版本或 matplotlib 代码版本作为参考
   - 你用 PPT 照着画（更美观）

---

## ✅ 完成标志

- [ ] 8 幅图片全部生成并插入论文
- [ ] LaTeX 重新编译无错误
- [ ] 论文细节打磨完成
- [ ] 答辩 PPT 制作完成
- [ ] 模拟答辩 1-2 次

---

**预计总耗时**：9-13 小时（分 3 天完成）

**当前进度**：0/8 图片，0/3 打磨项，0/1 PPT

**下一步**：告诉我评估结果和训练日志的路径格式，我立即生成脚本！
