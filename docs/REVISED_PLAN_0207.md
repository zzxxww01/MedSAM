# 修订版计划 - 从2月7日开始

> **起始日期**: 2025年2月7日
> **论文初稿截止**: 3月底
> **查重/盲审**: 4月
> **答辩**: 5月

---

## 协作模式（固定约束）

- 开发模式固定为：**本地开发 + 远程服务器运行**。
- 本地仓库用于：代码编写、脚本准备、文档更新，不用于判断远程训练是否已运行/完成。
- 训练、预处理、评估进度以服务器端日志和产物为准。
- 当无法直接访问服务器状态时，默认采用两种方式之一：
  1. 向你确认当前服务器进度。
  2. 直接给出可执行的服务器命令（启动/检查/回传）。
- 每个阶段开始前，先提供“服务器执行清单”，避免重复遗漏该模式。

---

## 服务器固定信息（2026-02-09 更新）

- 服务器路径：`~/chengang/zxw/MedSAM`
- Conda环境：`medsam`
- GPU资源：`4 x NVIDIA V100`
- 默认资源策略：训练/推理默认使用2张GPU；无特殊冲突时保持该配置，必要时可扩展到更多GPU。
- 代码状态：默认以“服务器代码已是最新（已 `git pull`）”为前提执行实验。
- 启动规则：训练命令必须在项目根目录 `~/chengang/zxw/MedSAM` 执行，不在 `work_dir` 下直接启动脚本。

### 当前远程产物快照（来自用户回传）

```text
work_dir/
baseline_train.log
MedSAM
MedSAM-Baseline-20260208-1844
MedSAM-Baseline-20260208-1908
MedSAM-Baseline-20260208-1919
MedSAM-Baseline-20260208-1924
MedSAM-Baseline-20260208-1935
MedSAM-Baseline-20260208-1940
MedSAM-Baseline-20260208-1953
medsam_vit_b.pth
```

### 2026-02-09 追加核验（目录级）

- Baseline目录按时间排序（新→旧）：
  - `MedSAM-Baseline-20260208-1953`
  - `MedSAM-Baseline-20260208-1940`
  - `MedSAM-Baseline-20260208-1935`
  - `MedSAM-Baseline-20260208-1924`
  - `MedSAM-Baseline-20260208-1919`
  - `MedSAM-Baseline-20260208-1908`
  - `MedSAM-Baseline-20260208-1844`
- 目前已确认：仅 `MedSAM-Baseline-20260208-1953` 目录可见
  - `medsam_model_best.pth`
  - `MedSAM-Baselinetrain_loss.png`
- `baseline_train.log` 出现过一次失败启动记录：在 `work_dir` 下执行导致找不到 `train_multi_gpus.py`。
- 判定：`20260208-1953` 为当前最优 Baseline 候选目录（待日志与指标最终核验）。

### 2026-02-12 进度更新（Ablation实跑）

- A1 `Inter-CBL` 已完成（200 epochs）
  - 日志: `work_dir/A1_20260209-2026.log`
  - 目录: `work_dir/MedSAM-FLARE22-A1-InterCBL-20260209-2026-20260209-2027`
- A2 `Intra-CBL` 已完成（200 epochs）
  - 日志: `work_dir/A2_20260210-2309.log`
  - 目录: `work_dir/MedSAM-FLARE22-A2-IntraCBL-20260210-2309-20260210-2309`
- A3 `Balance Loss` 首轮失败（OOM）
  - 日志: `work_dir/A3_20260212-0010.log`
  - 结论: 维持2卡策略下，将 A3 重跑参数改为 `-batch_size 1`。
- A3 二次重启失败（命令环境变量缺失）
  - 现象: `MASTER_ADDR expected, but not set`，并出现 `Rank 0~3`（误触发4卡）
  - 结论: A3 重跑必须使用标准化命令模板，先显式 `export` 关键环境变量。

### 2026-02-13 进度更新（A3重跑完成）

- A3 `Balance Loss`（`batch_size=1`）已完成 200 epochs
  - 成功日志: `work_dir/A3_20260212-002344_bs1.log`
  - 完成标记: `[Rank 0/1] Epoch 199: 100%`，时间 `20260213-0501`
  - 产物目录: `work_dir/MedSAM-FLARE22-A3-BalanceLoss-20260212-002344-20260212-0023`
  - 关键文件: `medsam_model_best.pth`、`medsam_model_latest.pth`、`*_train_loss.png`
- A1/A2/A3 评估回填（40例，CT_Abd，同口径）
  - A1 Inter-CBL: DSC=`0.940596`，HD95=`4.790533`，ASD=`0.531697`
  - A2 Intra-CBL: DSC=`0.952554`，HD95=`3.368403`，ASD=`0.374899`
  - A3 Balance: DSC=`0.903470`，HD95=`7.922879`，ASD=`0.886811`
- 训练资源侧证据：运行中 `medsam` 进程稳定占用2卡（GPU1/GPU3），无新增 OOM 迹象。
- 当前节奏：A1/A2/A3 指标已回填，A2 当前最优；下一步先补 Baseline 同口径评估，再启动 A3 修正实验与 Attention 模块准备。
- 当前完成度（实验主线）：`Baseline + A1 + A2 + A3` 已完成训练与阶段评估，进入“修正优化 + 下一阶段实验”。

### 进度判定规则（补充）

- 只要 `work_dir` 下已出现对应实验目录和日志，即判定为“已启动/已运行过”。
- 是否“已完成”必须再看：`train.log` 末尾、最佳模型文件、评估指标。

---

## 时间总览

```
2月7日 ─────────────────────────────────────────────── 3月31日
   │                                                      │
   ├─ Phase 1 (2.7-2.14)   : 环境+Baseline+文献 [8天]     │
   ├─ Phase 2 (2.15-2.23)  : Balance Loss实现+实验 [9天]  │
   ├─ Phase 3 (2.24-3.5)   : Attention实现+实验 [10天]    │
   ├─ Phase 4 (3.6-3.15)   : 综合实验+对比 [10天]         │
   └─ Phase 5 (3.16-3.31)  : 论文撰写+整合 [16天]         │
                                                          ▼
                                                    论文初稿完成
```

---

## Phase 1: 环境搭建+Baseline+文献 (2月7日-2月14日)

**目标**: 跑通Baseline，完成文献调研，产出论文1-2章框架

### 2月7日 (周五) - 环境搭建

```
上午:
□ 确认GPU环境 (型号、显存、CUDA版本)
□ 创建conda环境: conda create -n medsam python=3.10
□ 安装PyTorch (匹配CUDA版本)

下午:
□ 安装依赖: pip install monai scikit-image matplotlib tqdm wandb
□ 下载SAM权重: sam_vit_b_01ec64.pth
□ 验证环境: python -c "import torch; from segment_anything import sam_model_registry"
```

### 2月8日 (周六) - 数据准备+理解代码

```
上午:
□ 准备FLARE22数据集 (或已有数据)
□ 检查数据格式: imgs/*.npy [1024,1024,3], gts/*.npy [256,256]
□ 运行sanity check确认数据正确

下午:
□ 通读 train_one_gpu.py，理解训练流程
□ 通读 mask_decoder.py，理解模型结构
□ 记录关键代码位置，为后续修改做准备
```

### 2月9日 (周日) - 启动Baseline训练

```
□ 启动Baseline训练 (后台运行，预计1-2天)
  python train_one_gpu.py -i data/npy/CT_Abd -task_name Baseline -num_epochs 150

□ 同时开始文献阅读 (不浪费等待时间):
  - SAM原论文
  - MedSAM论文
  - Focal Loss论文
```

### 2月10-12日 (周一-周三) - 文献调研 (训练同时进行)

```
必读论文清单 (按优先级):
□ [核心] MedSAM: Segment Anything in Medical Images
□ [核心] SAM: Segment Anything
□ [损失] Focal Loss for Dense Object Detection
□ [损失] Class-Balanced Loss Based on Effective Number of Samples
□ [注意力] Attention Is All You Need
□ [对比] UniverSeg: Universal Medical Image Segmentation
□ [对比] nnU-Net
□ [背景] U-Net

每篇论文记录:
- 核心方法 (1-2句话)
- 与本工作的关系
- 可借鉴的点
```

### 2月13-14日 (周四-周五) - 论文1-2章框架 + Baseline结果

```
2月13日:
□ 整理Baseline实验结果 (DSC, HD95)
□ 生成训练曲线图
□ 撰写论文第1章框架 (2000字)

2月14日:
□ 撰写论文第2章框架 (2500字)
□ 整理参考文献列表
□ Phase 1 总结，准备Phase 2
```

**Phase 1 产出检查清单**:
```
□ 可运行的开发环境
□ Baseline模型权重 + 性能数据
□ 20+篇文献阅读笔记
□ 论文第1-2章框架 (共4500字)
```

---

## Phase 2: Balance Loss实现+实验 (2月15日-2月23日)

**目标**: 完成Balance Loss代码，运行消融实验，产出论文第3章

### 2月15-16日 (周六-周日) - 代码实现

```
2月15日:
□ 创建 losses/ 目录
□ 实现 InterClassBalanceLoss
□ 单元测试验证

2月16日:
□ 实现 IntraClassBalanceLoss
□ 实现完整 BalanceLoss 类
□ 实现两阶段切换逻辑
```

### 2月17-18日 (周一-周二) - 集成+启动实验

```
□ 修改训练脚本集成Balance Loss
□ 启动实验A2 (Inter-CBL)
□ 启动实验A3 (Intra-CBL)
□ 启动实验A4 (完整Balance Loss)
```

### 2月19-21日 (周三-周五) - 实验运行+超参数

```
□ 监控实验进度
□ 运行超参数实验A5
□ 记录所有结果
```

### 2月22-23日 (周六-周日) - 分析+论文第3章

```
□ 整理实验数据表格
□ 生成可视化图表
□ 撰写论文第3章 (6000字)
```

**Phase 2 产出检查清单**:
```
□ losses/balance_loss.py
□ 实验A2-A5结果
□ 论文第3章初稿
```

---

## Phase 3: Attention模块实现+实验 (2月24日-3月5日)

**目标**: 完成注意力融合模块，运行实验，产出论文第4章

### 2月24-26日 - 代码实现

```
□ 创建 modules/ 目录
□ 实现 AttentionCrossBlock
□ 集成到训练流程
□ 单元测试验证
```

### 2月27日-3月2日 - 实验运行

```
□ B1: Attention模块效果验证
□ B2: 不同配置对比
□ 监控并记录结果
```

### 3月3-5日 - 分析+论文第4章

```
□ 整理实验结果
□ 生成注意力可视化图
□ 撰写论文第4章 (5000字)
```

**Phase 3 产出检查清单**:
```
□ modules/attention_cross_block.py
□ 实验B系列结果
□ 论文第4章初稿
```

---

## Phase 4: 综合实验+对比 (3月6日-3月15日)

**目标**: 完成所有对比实验，产出论文第5章

### 3月6-8日 - 完整模型实验

```
□ C1: Balance Loss + Attention (完整方法)
□ 与Baseline对比
□ 记录完整消融表
```

### 3月9-12日 - 对比实验

```
□ C2: 与MedSAM原始对比
□ C3: 与其他方法对比 (如有)
□ C4: 跨数据集泛化测试
```

### 3月13-15日 - 论文第5章

```
□ 整理所有实验数据
□ 制作对比表格和图
□ 撰写论文第5章 (6000字)
```

**Phase 4 产出检查清单**:
```
□ 完整消融实验表
□ 对比实验结果
□ 论文第5章初稿
```

---

## Phase 5: 论文撰写+整合 (3月16日-3月31日)

**目标**: 完成论文全文，准备提交

### 3月16-20日 - 论文整合

```
□ 撰写第6章总结 (2000字)
□ 撰写摘要 (中英文)
□ 整合第1-6章
□ 统一格式和符号
```

### 3月21-25日 - 修改完善

```
□ 通读全文，修改逻辑问题
□ 完善图表质量
□ 补充参考文献 (50篇+)
□ 提交导师初审
```

### 3月26-31日 - 定稿

```
□ 根据导师反馈修改
□ 格式规范化
□ 自查查重
□ 准备送审版本
```

**Phase 5 产出检查清单**:
```
□ 完整论文初稿 (~30000字)
□ 格式规范的终稿
□ 查重报告 (<15%)
```
---

## 实验总表

| ID | 名称 | 阶段 | 论文章节 |
|----|------|------|----------|
| A1 | Baseline | Phase 1 | 3.5, 5.2 |
| A2 | Inter-CBL | Phase 2 | 3.5 |
| A3 | Intra-CBL | Phase 2 | 3.5 |
| A4 | Balance Loss | Phase 2 | 3.5 |
| A5 | 超参数分析 | Phase 2 | 3.5 |
| B1 | Attention | Phase 3 | 4.4 |
| B2 | 配置对比 | Phase 3 | 4.4 |
| C1 | 完整方法 | Phase 4 | 5.2 |
| C2 | 对比实验 | Phase 4 | 5.3 |
| C4 | 泛化实验 | Phase 4 | 5.4 |

---

## 论文字数规划

| 章节 | 字数 | 截止日期 |
|------|------|----------|
| 第1章 绪论 | 4000 | 2月14日 |
| 第2章 理论基础 | 4000 | 2月14日 |
| 第3章 Balance Loss | 6000 | 2月23日 |
| 第4章 Attention | 5000 | 3月5日 |
| 第5章 实验 | 6000 | 3月15日 |
| 第6章 总结 | 2000 | 3月20日 |
| 摘要 | 500+300词 | 3月20日 |
| **总计** | **~28000** | 3月31日 |

---

## 关键里程碑

| 日期 | 里程碑 | 验收标准 |
|------|--------|----------|
| 2月14日 | Phase 1完成 | Baseline结果+论文1-2章框架 |
| 2月23日 | Phase 2完成 | Balance Loss代码+实验+第3章 |
| 3月5日 | Phase 3完成 | Attention代码+实验+第4章 |
| 3月15日 | Phase 4完成 | 所有实验+第5章 |
| 3月31日 | 论文初稿 | 完整论文提交导师 |

---

**今日任务 (2月7日)**:
```
□ 确认GPU环境
□ 创建conda环境
□ 安装PyTorch和依赖
□ 下载SAM预训练权重
□ 验证环境可用
```

