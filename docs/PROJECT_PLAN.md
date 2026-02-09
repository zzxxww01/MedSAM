# MedSAM 改进项目详细规划文档

> **文档版本**: v1.0
> **创建日期**: 2025年2月6日
> **项目名称**: 基于MedSAM的通用医学图像分割模型改进
> **研究方向**: 医学图像分割 + 少样本学习 + 不平衡学习

---

## 目录

1. [项目概述](#1-项目概述)
2. [代码架构分析](#2-代码架构分析)
3. [创新点详细方案](#3-创新点详细方案)
4. [代码修改规划](#4-代码修改规划)
5. [实验设计](#5-实验设计)
6. [时间规划](#6-时间规划)
7. [风险与应对](#7-风险与应对)
8. [Git分支管理策略](#8-git分支管理策略)
9. [待确认事项](#9-待确认事项)

---

## 1. 项目概述

### 1.1 研究背景

医学图像分割是计算机辅助诊断的核心任务。现有方法面临以下挑战：
- **类别不平衡**: 前景目标（病灶、器官）在图像中占比极低（常<1%）
- **标注稀缺**: 医学图像标注需要专业知识，成本高昂
- **泛化性差**: 模型难以跨数据集、跨模态迁移

### 1.2 基础模型

**MedSAM (Segment Anything in Medical Images)**
- 基于SAM (Segment Anything Model) 的医学图像分割模型
- 使用ViT作为图像编码器
- 支持边界框(bounding box)提示
- 在多种医学图像上展现了良好的泛化能力

### 1.3 研究目标

1. 解决医学图像分割中的**双重类别不平衡**问题
2. 引入**注意力机制**改进特征融合
3. 在多个医学图像数据集上验证改进效果

### 1.4 数据集

| 数据集 | 类型 | 用途 | 说明 |
|--------|------|------|------|
| FLARE22 | 腹部CT | 主要实验 | 13个腹部器官分割 |
| KiTS19 | 肾脏CT | 对比实验 | 肾脏肿瘤分割 |
| NIH | 胰腺CT | 对比实验 | 胰腺分割 |
| BUSI | 乳腺超声 | 跨模态验证 | 乳腺肿瘤分割 |
| CVC-ClinicDB | 结肠镜 | 跨模态验证 | 息肉分割 |

---

## 2. 代码架构分析

### 2.1 当前项目结构

```
MedSAM/
├── segment_anything/           # SAM核心模块
│   ├── modeling/
│   │   ├── image_encoder.py   # ViT图像编码器 (不修改)
│   │   ├── mask_decoder.py    # 掩码解码器 (需修改：加入注意力融合)
│   │   ├── prompt_encoder.py  # 提示编码器 (不修改)
│   │   ├── transformer.py     # Transformer模块 (可能需修改)
│   │   ├── common.py          # 通用组件
│   │   └── sam.py            # 主模型类
│   ├── build_sam.py           # 模型构建
│   ├── predictor.py           # 预测器
│   └── utils/                 # 工具函数
├── train_one_gpu.py           # 单GPU训练脚本 (需修改：加入Balance Loss)
├── train_multi_gpus.py        # 多GPU训练脚本
├── MedSAM_Inference.py        # 推理脚本
├── utils/                     # 工具函数
│   ├── SurfaceDice.py        # Surface Dice评估
│   └── ...
├── comparisons/               # 对比模型
│   ├── DeepLabV3+/
│   ├── SAM/
│   └── nnU-Net/
└── extensions/                # 扩展功能
    ├── point_prompt/
    └── text_prompt/
```

### 2.2 核心模块功能分析

#### 2.2.1 MedSAM模型 (`train_one_gpu.py` 中的 `MedSAM` 类)

```python
class MedSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        # image_encoder: ViT编码器，输出 [B, 256, 64, 64]
        # mask_decoder: 掩码解码器
        # prompt_encoder: 提示编码器（冻结）

    def forward(self, image, box):
        # 1. 图像编码: image -> image_embedding [B, 256, 64, 64]
        # 2. 提示编码: box -> sparse_embeddings, dense_embeddings
        # 3. 掩码解码: -> low_res_masks
        # 4. 上采样: -> ori_res_masks
```

#### 2.2.2 MaskDecoder (`segment_anything/modeling/mask_decoder.py`)

```python
class MaskDecoder(nn.Module):
    # 核心组件:
    # - transformer: TwoWayTransformer，处理图像和提示特征
    # - output_upscaling: 上采样网络
    # - output_hypernetworks_mlps: 生成掩码的MLP
    # - iou_prediction_head: IoU预测头

    def forward(self, image_embeddings, image_pe,
                sparse_prompt_embeddings, dense_prompt_embeddings,
                multimask_output):
        # 返回: masks, iou_pred
```

#### 2.2.3 TwoWayTransformer (`segment_anything/modeling/transformer.py`)

```python
class TwoWayTransformer(nn.Module):
    # 双向Transformer:
    # - 查询(prompt) 到 图像 的注意力
    # - 图像 到 查询(prompt) 的注意力

    def forward(self, image_embedding, image_pe, point_embedding):
        # 返回: queries, keys (处理后的特征)
```

### 2.3 当前损失函数

```python
# train_one_gpu.py 第284-286行
seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

# 训练时 (第324, 333行)
loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
```

---

## 3. 创新点详细方案

### 3.1 创新点1: Balance Loss（解决双重类别不平衡）

#### 3.1.1 问题分析

**类别间不平衡 (Inter-class Imbalance)**
- 医学图像中前景目标占比极低（常<1%）
- 标准BCE/Dice Loss对此不敏感
- 模型倾向于预测"全背景"

**类别内不平衡 (Intra-class Imbalance)**
- 每个类别内部，简单样本远多于困难样本
- 简单样本：内部区域、高对比度区域
- 困难样本：边界、低对比度区域、异质区域

#### 3.1.2 技术方案

**Balance Loss = α × Inter-CBL + β × Intra-CBL + γ × Dice Loss**

**Inter-CBL (类别间平衡损失)**
```
算法流程:
1. 统计前景像素数量 N_fg
2. 在背景像素中，找出预测概率最高的 N_fg 个像素（最容易被误判的背景）
3. 仅在 前景像素 + 困难背景像素 上计算BCE损失
4. 两部分损失等权相加

效果: 前景和背景对损失的贡献相等
```

**Intra-CBL (类别内平衡损失)**
```
算法流程:
1. 计算每个像素的"正确性概率"
   - 前景像素: correctness = pred_prob
   - 背景像素: correctness = 1 - pred_prob
2. 按阈值θ (如0.9) 划分:
   - 简单样本: correctness > θ
   - 困难样本: correctness ≤ θ
3. 分别计算损失，加权组合:
   loss = w_easy × L_easy + w_hard × L_hard

效果: 困难样本获得更高权重
```

#### 3.1.3 两阶段训练策略

| 阶段 | 训练轮次 | 损失函数 | 说明 |
|------|----------|----------|------|
| Stage 1 | Epoch 0-50 | Intra-CBL + Dice | 初期稳定训练 |
| Stage 2 | Epoch 50+ | Inter-CBL + Intra-CBL + Dice | 完整Balance Loss |

**切换时机判断**:
- 当验证集Dice > 0.5 时切换
- 或固定在第50个epoch切换

#### 3.1.4 代码设计

```python
# 文件: losses/balance_loss.py

class InterClassBalanceLoss(nn.Module):
    """类别间平衡损失"""
    def __init__(self, min_hard_samples=16):
        ...

    def forward(self, pred, target):
        # pred: [B, 1, H, W] logits
        # target: [B, 1, H, W] binary mask

        # 1. 统计前景像素数
        fg_mask = (target == 1)
        fg_count = fg_mask.sum()

        # 2. 挖掘困难背景
        bg_mask = (target == 0)
        bg_probs = sigmoid(pred[bg_mask])
        k = min(fg_count, bg_probs.numel())
        hard_bg_indices = bg_probs.topk(k).indices

        # 3. 计算平衡损失
        fg_loss = BCE(pred[fg_mask], target[fg_mask])
        hard_bg_loss = BCE(pred[hard_bg_indices], target[hard_bg_indices])

        return (fg_loss + hard_bg_loss) / 2


class IntraClassBalanceLoss(nn.Module):
    """类别内平衡损失"""
    def __init__(self, threshold=0.9, hard_weight=2.0, easy_weight=1.0):
        ...

    def forward(self, pred, target):
        # 1. 计算正确性概率
        pred_prob = sigmoid(pred)
        correctness = target * pred_prob + (1-target) * (1-pred_prob)

        # 2. 划分难易样本
        easy_mask = (correctness > self.threshold)
        hard_mask = ~easy_mask

        # 3. 加权损失
        per_sample_loss = BCE(pred, target, reduction='none')
        easy_loss = per_sample_loss[easy_mask].mean()
        hard_loss = per_sample_loss[hard_mask].mean()

        return (self.easy_weight * easy_loss + self.hard_weight * hard_loss) / total_weight


class BalanceLoss(nn.Module):
    """完整Balance Loss"""
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, stage=1):
        self.inter_cbl = InterClassBalanceLoss()
        self.intra_cbl = IntraClassBalanceLoss()
        self.stage = stage  # 1 or 2

    def set_stage(self, stage):
        self.stage = stage

    def dice_loss(self, pred, target):
        # 标准Dice Loss
        ...

    def forward(self, pred, target):
        loss = 0
        loss_dict = {}

        # Intra-CBL (始终使用)
        intra = self.intra_cbl(pred, target)
        loss += self.beta * intra
        loss_dict['intra_cbl'] = intra.item()

        # Inter-CBL (仅Stage 2)
        if self.stage == 2:
            inter = self.inter_cbl(pred, target)
            loss += self.alpha * inter
            loss_dict['inter_cbl'] = inter.item()

        # Dice Loss
        dice = self.dice_loss(pred, target)
        loss += self.gamma * dice
        loss_dict['dice'] = dice.item()

        return loss, loss_dict
```

---

### 3.2 创新点2: AttentionCrossBlock（注意力特征融合）

#### 3.2.1 问题分析

**当前MedSAM的局限**:
- 使用边界框(box)作为唯一提示
- 缺乏对目标外观的先验知识
- 相似目标的区分能力有限

**UniverSeg的启发**:
- 使用支持集(support set)提供目标样例
- 但采用简单平均融合，无法区分样本质量
- 噪声样本和相关样本被"一视同仁"

#### 3.2.2 技术方案

**使用交叉注意力机制替代简单平均**

```
输入:
  - query_features: 查询图像特征 [B, C, H, W]
  - support_features: 支持集特征 [B, N, C, H, W] (N个支持样本)

输出:
  - fused_features: 融合后的特征 [B, C, H, W]

交叉注意力:
  Q = Linear(query_features)
  K = Linear(support_features)
  V = Linear(support_features)

  Attention(Q, K, V) = softmax(QK^T / √d) × V
```

**设计优势**:
1. 自适应权重：根据query与support的相关性分配权重
2. 噪声抑制：不相关的支持样本权重自动降低
3. 保留细节：空间维度上的精细匹配

#### 3.2.3 集成方案

**方案A: MaskDecoder内部集成**

```
位置: mask_decoder.py 的 predict_masks 方法中
时机: 在transformer处理之前融合支持集特征

修改流程:
1. 编码查询图像 -> query_features
2. 编码支持集 -> support_features  [新增]
3. 注意力融合 -> fused_features    [新增]
4. Transformer处理 (使用fused_features)
5. 生成掩码
```

**方案B: 独立融合模块**

```
位置: MedSAM主模型中
时机: 在image_encoder之后、mask_decoder之前

修改流程:
1. image_encoder(query) -> query_features
2. image_encoder(support) -> support_features  [新增]
3. AttentionCrossBlock(query, support) -> fused_features  [新增]
4. mask_decoder(fused_features) -> masks
```

**推荐方案B**：更模块化，易于调试和消融实验

#### 3.2.4 代码设计

```python
# 文件: modules/attention_cross_block.py

class AttentionCrossBlock(nn.Module):
    """交叉注意力特征融合模块"""

    def __init__(self, embed_dim=256, num_heads=8, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # 层归一化
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, query_features, support_features, support_masks=None):
        """
        Args:
            query_features: [B, C, H, W]
            support_features: [B, N, C, H, W] 或 [B, C, H, W]
            support_masks: [B, N, 1, H, W] 可选，用于加权

        Returns:
            fused_features: [B, C, H, W]
        """
        B, C, H, W = query_features.shape

        # 处理单个支持样本的情况
        if support_features.dim() == 4:
            support_features = support_features.unsqueeze(1)

        N = support_features.shape[1]

        # 展平空间维度
        q = query_features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        kv = support_features.flatten(3).permute(0, 1, 3, 2)  # [B, N, H*W, C]
        kv = kv.reshape(B, N * H * W, C)  # [B, N*H*W, C]

        # 层归一化
        q = self.norm_q(q)
        kv = self.norm_kv(kv)

        # 投影
        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        # 多头注意力
        q = q.reshape(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N*H*W, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N*H*W, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力计算
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, H*W, C)
        out = self.out_proj(out)

        # 残差连接
        out = out + query_features.flatten(2).transpose(1, 2)
        out = self.norm_out(out)

        # MLP
        out = out + self.mlp(out)

        # 恢复形状
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return out
```

---

### 3.3 MedSAM适配方案

#### 3.3.1 支持集机制设计

**当前MedSAM**: 使用边界框(box)作为提示
**改进目标**: 增加支持集(support set)作为额外提示

```python
# 文件: 新模型定义

class MedSAM_FSS(nn.Module):
    """Few-Shot Segmentation版本的MedSAM"""

    def __init__(self, medsam_model, num_support=5, use_attention=True):
        super().__init__()
        self.image_encoder = medsam_model.image_encoder
        self.mask_decoder = medsam_model.mask_decoder
        self.prompt_encoder = medsam_model.prompt_encoder

        self.num_support = num_support
        self.use_attention = use_attention

        # 新增: 注意力融合模块
        if use_attention:
            self.attention_fusion = AttentionCrossBlock(embed_dim=256)

        # 冻结prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, query_image, boxes, support_images=None, support_masks=None):
        """
        Args:
            query_image: [B, 3, 1024, 1024] 查询图像
            boxes: [B, 4] 边界框提示
            support_images: [B, N, 3, 1024, 1024] 支持集图像 (可选)
            support_masks: [B, N, 1, 256, 256] 支持集掩码 (可选)
        """
        # 编码查询图像
        query_features = self.image_encoder(query_image)  # [B, 256, 64, 64]

        # 如果提供了支持集，进行特征融合
        if support_images is not None and self.use_attention:
            # 编码支持集
            B, N = support_images.shape[:2]
            support_flat = support_images.reshape(B * N, *support_images.shape[2:])
            support_features = self.image_encoder(support_flat)
            support_features = support_features.reshape(B, N, *support_features.shape[1:])

            # 注意力融合
            query_features = self.attention_fusion(
                query_features,
                support_features,
                support_masks
            )

        # 提示编码
        with torch.no_grad():
            box_torch = torch.as_tensor(boxes, dtype=torch.float32, device=query_image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=box_torch, masks=None
            )

        # 掩码解码
        low_res_masks, iou_pred = self.mask_decoder(
            image_embeddings=query_features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # 上采样
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(query_image.shape[2], query_image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        return ori_res_masks
```

#### 3.3.2 数据加载器修改

```python
# 文件: datasets/fewshot_dataset.py

class FewShotNpyDataset(Dataset):
    """支持少样本学习的数据集"""

    def __init__(self, data_root, num_support=5, bbox_shift=20, same_class_support=True):
        self.data_root = data_root
        self.num_support = num_support
        self.bbox_shift = bbox_shift
        self.same_class_support = same_class_support

        # 加载文件列表
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(glob.glob(join(self.gt_path, "**/*.npy"), recursive=True))

        # 按类别组织样本（用于same_class采样）
        self.class_to_samples = self._organize_by_class()

    def _organize_by_class(self):
        """按类别组织样本索引"""
        class_dict = {}
        for idx, path in enumerate(self.gt_path_files):
            # 从文件名或内容提取类别信息
            # 具体实现取决于数据集格式
            ...
        return class_dict

    def _sample_support(self, query_idx, query_class):
        """采样支持集"""
        if self.same_class_support:
            # 从同类别样本中采样
            candidates = self.class_to_samples.get(query_class, [])
            candidates = [c for c in candidates if c != query_idx]
        else:
            # 随机采样
            candidates = list(range(len(self)))
            candidates.remove(query_idx)

        if len(candidates) < self.num_support:
            # 如果候选不足，允许重复采样
            support_indices = random.choices(candidates, k=self.num_support)
        else:
            support_indices = random.sample(candidates, self.num_support)

        return support_indices

    def __getitem__(self, index):
        # 加载查询样本
        query_img, query_gt, query_class = self._load_sample(index)

        # 计算边界框
        boxes = self._get_boxes(query_gt)

        # 采样并加载支持集
        support_indices = self._sample_support(index, query_class)
        support_imgs = []
        support_gts = []
        for idx in support_indices:
            img, gt, _ = self._load_sample(idx)
            support_imgs.append(img)
            support_gts.append(gt)

        return {
            'query_image': query_img,           # [3, 1024, 1024]
            'query_gt': query_gt,               # [1, 256, 256]
            'boxes': boxes,                     # [4]
            'support_images': torch.stack(support_imgs),  # [N, 3, 1024, 1024]
            'support_gts': torch.stack(support_gts),      # [N, 1, 256, 256]
        }
```

---

## 4. 代码修改规划

### 4.1 新建文件清单

| 序号 | 文件路径 | 功能说明 | 优先级 | 依赖 |
|------|----------|----------|--------|------|
| 1 | `losses/__init__.py` | 损失函数模块初始化 | 高 | - |
| 2 | `losses/balance_loss.py` | Balance Loss实现 | **高** | - |
| 3 | `modules/__init__.py` | 模块初始化 | 高 | - |
| 4 | `modules/attention_cross_block.py` | 注意力融合模块 | **高** | - |
| 5 | `datasets/__init__.py` | 数据集模块初始化 | 中 | - |
| 6 | `datasets/fewshot_dataset.py` | Few-Shot数据加载器 | 中 | - |
| 7 | `models/__init__.py` | 模型模块初始化 | 中 | - |
| 8 | `models/medsam_fss.py` | MedSAM-FSS模型定义 | 中 | 4 |
| 9 | `train_balance_loss.py` | Balance Loss训练脚本 | **高** | 2 |
| 10 | `train_fss.py` | Few-Shot训练脚本 | 中 | 6, 8 |
| 11 | `configs/balance_loss.yaml` | 配置文件 | 中 | - |
| 12 | `utils/metrics.py` | 评估指标 | 中 | - |

### 4.2 修改文件清单

| 序号 | 文件路径 | 修改内容 | 优先级 | 影响范围 |
|------|----------|----------|--------|----------|
| 1 | `train_one_gpu.py` | 集成Balance Loss | **高** | 训练流程 |
| 2 | `segment_anything/modeling/mask_decoder.py` | 可选：集成注意力模块 | 中 | 推理流程 |
| 3 | `segment_anything/modeling/sam.py` | 可选：支持Few-Shot模式 | 低 | 模型架构 |

### 4.3 文件修改详细说明

#### 4.3.1 `train_one_gpu.py` 修改

**修改位置1**: 导入部分 (第1-26行附近)
```python
# 新增导入
from losses import BalanceLoss, get_balance_loss
```

**修改位置2**: 参数解析 (第144-183行附近)
```python
# 新增参数
parser.add_argument("-loss_type", type=str, default="balance",
                    choices=["dice_ce", "balance", "focal_balance"],
                    help="loss function type")
parser.add_argument("-balance_alpha", type=float, default=1.0,
                    help="weight for Inter-CBL")
parser.add_argument("-balance_beta", type=float, default=1.0,
                    help="weight for Intra-CBL")
parser.add_argument("-balance_gamma", type=float, default=1.0,
                    help="weight for Dice Loss")
parser.add_argument("-stage_switch_epoch", type=int, default=50,
                    help="epoch to switch from stage 1 to stage 2")
```

**修改位置3**: 损失函数定义 (第284-286行附近)
```python
# 原代码:
# seg_loss = monai.losses.DiceLoss(...)
# ce_loss = nn.BCEWithLogitsLoss(...)

# 新代码:
if args.loss_type == "dice_ce":
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    use_balance_loss = False
else:
    balance_loss = BalanceLoss(
        alpha=args.balance_alpha,
        beta=args.balance_beta,
        gamma=args.balance_gamma,
        stage=1  # 初始Stage 1
    )
    use_balance_loss = True
```

**修改位置4**: 训练循环 (第314-336行附近)
```python
for epoch in range(start_epoch, num_epochs):
    # 阶段切换
    if use_balance_loss and epoch == args.stage_switch_epoch:
        balance_loss.set_stage(2)
        print(f"Switched to Stage 2 at epoch {epoch}")

    for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)):
        # ... 前向传播 ...

        if use_balance_loss:
            loss, loss_dict = balance_loss(medsam_pred, gt2D)
            # 可选：记录各损失分量
            if args.use_wandb:
                wandb.log(loss_dict)
        else:
            loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())

        # ... 反向传播 ...
```

#### 4.3.2 `mask_decoder.py` 修改 (可选)

**方案**: 添加可选的注意力融合接口

```python
# 在 MaskDecoder.__init__ 中添加
self.use_support_attention = False
self.attention_fusion = None

def enable_support_attention(self, embed_dim=256, num_heads=8):
    """启用支持集注意力融合"""
    from modules import AttentionCrossBlock
    self.attention_fusion = AttentionCrossBlock(embed_dim, num_heads)
    self.use_support_attention = True

# 在 predict_masks 方法中添加
def predict_masks(self, image_embeddings, ..., support_features=None):
    # 如果提供了支持集特征，进行融合
    if self.use_support_attention and support_features is not None:
        image_embeddings = self.attention_fusion(image_embeddings, support_features)

    # ... 原有代码 ...
```

---

## 5. 实验设计

### 5.1 评估指标

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| DSC | Dice相似系数 | 2\|A∩B\| / (\|A\|+\|B\|) |
| HD95 | 95%豪斯多夫距离 | 边界点距离的95百分位 |
| ASD | 平均表面距离 | 边界点平均距离 |
| Precision | 精确率 | TP / (TP + FP) |
| Recall | 召回率 | TP / (TP + FN) |

### 5.2 消融实验设计

| 实验ID | 模型 | Loss | 模块 | 目的 |
|--------|------|------|------|------|
| A0 | MedSAM | Dice + CE | 原始 | Baseline |
| A1 | MedSAM | **Inter-CBL** | 原始 | 验证Inter-CBL效果 |
| A2 | MedSAM | **Intra-CBL** | 原始 | 验证Intra-CBL效果 |
| A3 | MedSAM | **Balance Loss** | 原始 | 验证完整损失函数 |
| B1 | MedSAM | Dice + CE | **AttentionCrossBlock** | 验证注意力模块效果 |
| C1 | MedSAM | **Balance Loss** | **AttentionCrossBlock** | **完整创新方案** |

### 5.3 对比实验设计

| 模型 | 类型 | 说明 |
|------|------|------|
| MedSAM (ours) | 提示式分割 | 我们的改进模型 |
| MedSAM (original) | 提示式分割 | 原始MedSAM |
| SAM | 提示式分割 | 通用分割模型 |
| UniverSeg | 少样本分割 | 对比通用分割 |
| nnU-Net | 任务特定 | 对比任务特定模型 |
| DeepLabV3+ | 语义分割 | 经典分割模型 |

### 5.4 跨数据集实验

**训练**: FLARE22
**测试**: KiTS19, NIH, BUSI, CVC-ClinicDB

目的：验证模型泛化能力

---

## 6. 时间规划

### 6.1 总体时间表

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1月中旬    1月底    2月初    2月中    2月底    3月初    3月中    3月底   │
│     │        │        │        │        │        │        │        │    │
│  ┌──┴────────┴──┐  ┌──┴────────┴──┐  ┌──┴────────┴──┐  ┌──┴────────┴──┐ │
│  │   Phase 1   │  │   Phase 2   │  │   Phase 3   │  │   Phase 4   │ │
│  │  基线复现    │  │ Balance Loss│  │ Attention   │  │  论文撰写   │ │
│  │  框架搭建    │  │   实现验证   │  │  Block实现  │  │  实验完善   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
│                                                                         │
│  4月初        4月中        4月底        5月初        5月中        5月底  │
│     │           │           │           │           │           │      │
│  ┌──┴───────────┴───────────┴──┐  ┌──────┴───────────┴───────────┴──┐  │
│  │         Phase 5            │  │         Phase 6               │  │
│  │   论文修改 + 查重 + 盲审    │  │       答辩准备 + 答辩          │  │
│  └─────────────────────────────┘  └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 详细任务分解

#### Phase 1: 基线复现与框架搭建 (1月中旬 - 1月底)

**Week 1: 环境与基线**
- [ ] 配置开发环境（Python, PyTorch, CUDA）
- [ ] 下载预训练权重
- [ ] 运行MedSAM原始训练流程
- [ ] 在FLARE22上验证baseline性能
- [ ] 记录baseline指标

**Week 2: 框架搭建**
- [ ] 设置wandb实验监控
- [ ] 完善git版本控制
- [ ] 准备额外数据集
- [ ] 设计代码目录结构
- [ ] 编写基础工具函数

#### Phase 2: Balance Loss实现 (2月1日 - 2月15日)

**Week 3: Loss实现**
- [ ] 实现Inter-CBL
- [ ] 实现Intra-CBL
- [ ] 实现完整BalanceLoss类
- [ ] 编写单元测试
- [ ] 验证损失函数数值正确性

**Week 4: 集成与验证**
- [ ] 集成到训练脚本
- [ ] 实现两阶段训练策略
- [ ] 运行消融实验A1-A3
- [ ] 对比实验：Dice+CE vs Balance Loss
- [ ] 调试和优化超参数

#### Phase 3: AttentionCrossBlock实现 (2月16日 - 2月28日)

**Week 5: 模块实现**
- [ ] 实现AttentionCrossBlock
- [ ] 实现SupportAggregator
- [ ] 设计Few-Shot数据加载器
- [ ] 编写单元测试

**Week 6: 完整模型**
- [ ] 实现MedSAM_FSS模型
- [ ] 集成注意力模块
- [ ] 运行消融实验B1
- [ ] 运行完整实验C1
- [ ] 多数据集验证

#### Phase 4: 论文撰写 (3月1日 - 3月31日)

**Week 7-8: 论文主体**
- [ ] 撰写绪论（背景、意义、现状）
- [ ] 撰写相关工作
- [ ] 撰写方法章节
- [ ] 整理实验结果

**Week 9-10: 论文完善**
- [ ] 制作可视化图表
- [ ] 撰写结论与展望
- [ ] 论文格式规范化
- [ ] 导师初审修改

#### Phase 5: 论文定稿 (4月1日 - 4月20日)

- [ ] 导师复审修改
- [ ] 查重检测
- [ ] 格式最终检查
- [ ] 盲审提交

#### Phase 6: 答辩 (5月1日 - 5月20日)

- [ ] 制作答辩PPT
- [ ] 准备答辩稿
- [ ] 模拟答辩
- [ ] 正式答辩

---

## 7. 风险与应对

### 7.1 技术风险

| 风险 | 可能性 | 影响 | 应对策略 |
|------|--------|------|----------|
| GPU显存不足 | 中 | 高 | 使用混合精度训练、减小batch size、梯度累积 |
| Balance Loss训练不稳定 | 中 | 中 | 调节阶段切换时机、监控loss曲线、添加warmup |
| 注意力模块显存过大 | 中 | 中 | 使用高效注意力(Flash Attention)、分块计算 |
| 跨数据集泛化差 | 低 | 中 | 增加数据增强、调整支持集采样策略 |

### 7.2 进度风险

| 风险 | 可能性 | 影响 | 应对策略 |
|------|--------|------|----------|
| 基线复现困难 | 低 | 高 | 提前开始、参考官方issue、必要时联系作者 |
| 实验时间超预期 | 中 | 中 | 并行化实验、提前准备多GPU环境 |
| 论文撰写拖延 | 中 | 高 | 边实验边撰写、设置中间deadline |

### 7.3 资源需求

| 资源 | 最低需求 | 推荐配置 |
|------|----------|----------|
| GPU | 1x RTX 3090 (24GB) | 2x A100 (40GB) |
| 内存 | 32GB | 64GB |
| 存储 | 500GB SSD | 1TB NVMe SSD |
| 训练时间 | ~48h/实验 | ~24h/实验 |

---

## 8. Git分支管理策略

### 8.1 分支结构

```
main (稳定版本)
├── develop (开发主分支)
│   ├── feature/balance-loss (创新点1开发)
│   ├── feature/attention-block (创新点2开发)
│   ├── feature/fewshot-dataset (数据集开发)
│   └── experiment/ablation-xxx (实验分支)
└── release/v1.0 (发布版本)
```

### 8.2 分支说明

| 分支 | 用途 | 合并目标 |
|------|------|----------|
| `main` | 稳定代码，随时可运行 | - |
| `develop` | 日常开发集成 | main |
| `feature/*` | 新功能开发 | develop |
| `experiment/*` | 实验代码（可能不合并） | - |
| `release/*` | 版本发布 | main |

### 8.3 提交规范

```
格式: <type>(<scope>): <subject>

类型:
- feat: 新功能
- fix: 修复bug
- docs: 文档更新
- refactor: 重构
- test: 测试相关
- exp: 实验相关

示例:
- feat(loss): implement Inter-CBL
- fix(train): fix stage switching logic
- docs: update README with usage examples
- exp(ablation): add A1 experiment results
```

### 8.4 开发流程

```
1. 从develop创建feature分支
   git checkout develop
   git checkout -b feature/balance-loss

2. 开发并提交
   git add .
   git commit -m "feat(loss): implement Inter-CBL"

3. 推送并创建PR
   git push origin feature/balance-loss
   # 创建Pull Request到develop

4. 代码审查后合并
   # 在GitHub/GitLab上合并PR

5. 删除feature分支
   git branch -d feature/balance-loss
```

---

## 9. 待确认事项

### 9.1 硬件资源
- [ ] 确认可用GPU型号和显存大小
- [ ] 确认服务器访问权限
- [ ] 确认存储空间

### 9.2 技术方案
- [ ] 支持集大小N的选择（建议5）
- [ ] 是否需要完全复现UniverSeg的支持集机制
- [ ] Balance Loss超参数初始值

### 9.3 时间节点
- [ ] 确认论文提交截止日期
- [ ] 确认答辩时间
- [ ] 确认中期检查时间

### 9.4 论文相关
- [ ] 论文格式要求
- [ ] 是否可以将创新点单独发表
- [ ] 开题报告时间

---

## 附录A: 参考文献

1. **MedSAM**: Ma J, et al. "Segment Anything in Medical Images." Nature Communications, 2024.
2. **SAM**: Kirillov A, et al. "Segment Anything." ICCV, 2023.
3. **UniverSeg**: Butoi V, et al. "UniverSeg: Universal Medical Image Segmentation." ICCV, 2023.
4. **Focal Loss**: Lin T, et al. "Focal Loss for Dense Object Detection." ICCV, 2017.
5. **Class-Balanced Loss**: Cui Y, et al. "Class-Balanced Loss Based on Effective Number of Samples." CVPR, 2019.

---

## 附录B: 环境配置

```bash
# 创建conda环境
conda create -n medsam python=3.10 -y
conda activate medsam

# 安装PyTorch (根据CUDA版本选择)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装依赖
pip install monai
pip install scikit-image
pip install matplotlib
pip install tqdm
pip install wandb
pip install nibabel
pip install SimpleITK

# 安装MedSAM
cd MedSAM
pip install -e .
```

---

## 附录C: 文件模板

### C.1 配置文件模板 (`configs/balance_loss.yaml`)

```yaml
# 训练配置
training:
  epochs: 200
  batch_size: 4
  learning_rate: 0.0001
  weight_decay: 0.01

# 损失函数配置
loss:
  type: "balance"
  alpha: 1.0  # Inter-CBL权重
  beta: 1.0   # Intra-CBL权重
  gamma: 1.0  # Dice权重
  threshold: 0.9
  hard_weight: 2.0
  easy_weight: 1.0
  stage_switch_epoch: 50

# 数据配置
data:
  train_path: "data/npy/CT_Abd"
  num_workers: 4

# 模型配置
model:
  type: "vit_b"
  checkpoint: "work_dir/SAM/sam_vit_b_01ec64.pth"
```

---

**文档结束**

> 最后更新: 2025年2月6日
> 作者: AI Assistant
> 版本: v1.0
