# 代码实现详细规范

> 本文档详细描述每个需要新建/修改的文件的具体实现细节

---

## 目录

1. [新建文件详细规范](#1-新建文件详细规范)
2. [修改文件详细规范](#2-修改文件详细规范)
3. [代码风格规范](#3-代码风格规范)
4. [测试规范](#4-测试规范)

---

## 1. 新建文件详细规范

### 1.1 `losses/balance_loss.py`

#### 文件概述
- **功能**: 实现Balance Loss及其子组件
- **优先级**: 高
- **预计代码行数**: 300-400行

#### 类设计

```python
# =============================================================================
# 类: InterClassBalanceLoss
# =============================================================================
class InterClassBalanceLoss(nn.Module):
    """
    类别间平衡损失 (Inter-Class Balance Loss)

    解决问题: 前景像素数量远少于背景像素
    策略: 在背景中挖掘与前景数量相等的困难样本

    Attributes:
        min_hard_samples (int): 最小困难样本数量，避免极端情况
        smooth (float): 平滑因子，避免除零

    Example:
        >>> loss_fn = InterClassBalanceLoss(min_hard_samples=16)
        >>> pred = torch.randn(2, 1, 256, 256)
        >>> target = torch.zeros(2, 1, 256, 256)
        >>> target[:, :, 100:150, 100:150] = 1
        >>> loss = loss_fn(pred, target)
    """

    def __init__(self, min_hard_samples: int = 16, smooth: float = 1e-6):
        pass

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测logits, shape [B, 1, H, W]
            target: 真值掩码, shape [B, 1, H, W], 值为0或1

        Returns:
            loss: 标量损失值
        """
        pass


# =============================================================================
# 类: IntraClassBalanceLoss
# =============================================================================
class IntraClassBalanceLoss(nn.Module):
    """
    类别内平衡损失 (Intra-Class Balance Loss)

    解决问题: 简单样本远多于困难样本
    策略: 按置信度划分难易样本，加权组合损失

    Attributes:
        threshold (float): 置信度阈值，用于划分难易样本
        hard_weight (float): 困难样本权重
        easy_weight (float): 简单样本权重
    """

    def __init__(self,
                 threshold: float = 0.9,
                 hard_weight: float = 2.0,
                 easy_weight: float = 1.0,
                 smooth: float = 1e-6):
        pass

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass


# =============================================================================
# 类: BalanceLoss
# =============================================================================
class BalanceLoss(nn.Module):
    """
    完整平衡损失 (Balance Loss)

    组合公式: L = α * Inter-CBL + β * Intra-CBL + γ * Dice

    支持两阶段训练:
    - Stage 1: 仅使用 Intra-CBL + Dice (稳定启动)
    - Stage 2: 使用完整 Balance Loss

    Attributes:
        alpha, beta, gamma: 各损失分量权重
        stage: 当前训练阶段 (1 或 2)
    """

    def __init__(self,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 gamma: float = 1.0,
                 threshold: float = 0.9,
                 hard_weight: float = 2.0,
                 easy_weight: float = 1.0,
                 use_dice: bool = True,
                 stage: int = 1,
                 smooth: float = 1e-6):
        pass

    def set_stage(self, stage: int):
        """切换训练阶段"""
        pass

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算Dice Loss"""
        pass

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            loss: 总损失标量
            loss_dict: 各分量损失字典，用于日志记录
        """
        pass
```

#### 实现要点

1. **边界情况处理**:
   - 当前景像素数为0时，回退到标准BCE
   - 当背景像素数不足时，使用全部背景

2. **数值稳定性**:
   - 使用`clamp`避免log(0)
   - 添加`smooth`项

3. **效率考虑**:
   - 使用`torch.where`代替循环
   - 避免不必要的tensor复制

---

### 1.2 `modules/attention_cross_block.py`

#### 文件概述
- **功能**: 实现注意力特征融合模块
- **优先级**: 高
- **预计代码行数**: 400-500行

#### 类设计

```python
# =============================================================================
# 类: AttentionCrossBlock
# =============================================================================
class AttentionCrossBlock(nn.Module):
    """
    交叉注意力特征融合模块

    使用多头交叉注意力机制融合查询特征和支持集特征

    Architecture:
        Input(query) ─┬─> Q_proj ─────────────────┐
                      │                           ├─> Attention ─> Output
        Input(support) ─> K_proj, V_proj ─────────┘

    Attributes:
        embed_dim (int): 嵌入维度
        num_heads (int): 注意力头数
        use_residual (bool): 是否使用残差连接
    """

    def __init__(self,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 use_layer_norm: bool = True,
                 use_residual: bool = True):
        pass

    def forward(self,
                query_features: torch.Tensor,
                support_features: torch.Tensor,
                support_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query_features: [B, C, H, W] 查询图像特征
            support_features: [B, N, C, H, W] 或 [B, C, H, W] 支持集特征
            support_masks: [B, N, 1, H, W] 可选的掩码权重

        Returns:
            fused_features: [B, C, H, W] 融合后的特征
        """
        pass


# =============================================================================
# 类: SpatialCrossAttention
# =============================================================================
class SpatialCrossAttention(nn.Module):
    """
    空间交叉注意力模块

    仅在空间维度上计算注意力，内存效率更高
    """
    pass


# =============================================================================
# 类: ChannelCrossAttention
# =============================================================================
class ChannelCrossAttention(nn.Module):
    """
    通道交叉注意力模块

    使用全局池化进行通道重标定
    """
    pass


# =============================================================================
# 类: SupportAggregator
# =============================================================================
class SupportAggregator(nn.Module):
    """
    支持集聚合模块

    将多个支持样本聚合为单一表示
    """
    pass
```

#### 实现要点

1. **内存优化**:
   - 支持分块计算
   - 可选Flash Attention

2. **灵活性**:
   - 支持单个或多个支持样本
   - 可选掩码加权

3. **可调试性**:
   - 返回注意力权重用于可视化

---

### 1.3 `datasets/fewshot_dataset.py`

#### 文件概述
- **功能**: Few-Shot学习数据加载器
- **优先级**: 中
- **预计代码行数**: 200-250行

#### 类设计

```python
class FewShotNpyDataset(Dataset):
    """
    Few-Shot学习数据集

    继承自原NpyDataset，增加支持集采样功能

    Attributes:
        num_support (int): 支持样本数量
        same_class_support (bool): 是否从同类别采样支持集
    """

    def __init__(self,
                 data_root: str,
                 num_support: int = 5,
                 bbox_shift: int = 20,
                 same_class_support: bool = True):
        pass

    def _organize_by_class(self) -> Dict[str, List[int]]:
        """按类别组织样本索引"""
        pass

    def _sample_support(self, query_idx: int, query_class: str) -> List[int]:
        """采样支持集索引"""
        pass

    def _load_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """加载单个样本"""
        pass

    def _get_boxes(self, gt: torch.Tensor) -> torch.Tensor:
        """从掩码计算边界框"""
        pass

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
                - query_image: [3, 1024, 1024]
                - query_gt: [1, 256, 256]
                - boxes: [4]
                - support_images: [N, 3, 1024, 1024]
                - support_gts: [N, 1, 256, 256]
        """
        pass
```

---

### 1.4 `models/medsam_fss.py`

#### 文件概述
- **功能**: Few-Shot Segmentation版本的MedSAM
- **优先级**: 中
- **预计代码行数**: 150-200行

#### 类设计

```python
class MedSAM_FSS(nn.Module):
    """
    Few-Shot Segmentation版本的MedSAM

    在原MedSAM基础上增加:
    1. 支持集编码
    2. 注意力特征融合
    """

    def __init__(self,
                 medsam_model: nn.Module,
                 num_support: int = 5,
                 use_attention: bool = True,
                 freeze_encoder: bool = False):
        pass

    def encode_support(self,
                       support_images: torch.Tensor,
                       support_masks: torch.Tensor) -> torch.Tensor:
        """编码支持集"""
        pass

    def forward(self,
                query_image: torch.Tensor,
                boxes: torch.Tensor,
                support_images: Optional[torch.Tensor] = None,
                support_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass
```

---

### 1.5 `train_balance_loss.py`

#### 文件概述
- **功能**: 使用Balance Loss的训练脚本
- **优先级**: 高
- **基于**: `train_one_gpu.py`
- **预计代码行数**: 400-450行

#### 主要修改点

```python
# 1. 新增导入
from losses import BalanceLoss

# 2. 新增命令行参数
parser.add_argument("-loss_type", type=str, default="balance")
parser.add_argument("-balance_alpha", type=float, default=1.0)
parser.add_argument("-balance_beta", type=float, default=1.0)
parser.add_argument("-balance_gamma", type=float, default=1.0)
parser.add_argument("-stage_switch_epoch", type=int, default=50)
parser.add_argument("-intra_threshold", type=float, default=0.9)

# 3. 损失函数初始化
if args.loss_type == "balance":
    criterion = BalanceLoss(
        alpha=args.balance_alpha,
        beta=args.balance_beta,
        gamma=args.balance_gamma,
        threshold=args.intra_threshold,
        stage=1
    )

# 4. 训练循环中的阶段切换
if epoch == args.stage_switch_epoch:
    criterion.set_stage(2)
    print(f"Switched to Stage 2 at epoch {epoch}")

# 5. 损失计算
loss, loss_dict = criterion(medsam_pred, gt2D)

# 6. 日志记录
if args.use_wandb:
    wandb.log({
        "total_loss": loss_dict['total'],
        "inter_cbl": loss_dict.get('inter_cbl', 0),
        "intra_cbl": loss_dict['intra_cbl'],
        "dice": loss_dict.get('dice', 0),
        "stage": criterion.stage
    })
```

---

## 2. 修改文件详细规范

### 2.1 `train_one_gpu.py` 修改

#### 修改清单

| 行号范围 | 修改类型 | 描述 |
|----------|----------|------|
| 1-26 | 新增 | 导入Balance Loss |
| 144-183 | 新增 | 添加损失函数相关参数 |
| 284-286 | 修改 | 损失函数初始化逻辑 |
| 314-336 | 修改 | 训练循环中的损失计算 |

#### 详细修改

**位置1: 导入部分 (约第20行后)**
```python
# 新增
try:
    from losses import BalanceLoss
    BALANCE_LOSS_AVAILABLE = True
except ImportError:
    BALANCE_LOSS_AVAILABLE = False
    print("Warning: Balance Loss not available, using default loss")
```

**位置2: 参数解析 (约第177行后)**
```python
# === Balance Loss 参数 ===
parser.add_argument(
    "-loss_type", type=str, default="dice_ce",
    choices=["dice_ce", "balance"],
    help="loss function type: dice_ce (original) or balance"
)
parser.add_argument(
    "-balance_alpha", type=float, default=1.0,
    help="weight for Inter-CBL in Balance Loss"
)
parser.add_argument(
    "-balance_beta", type=float, default=1.0,
    help="weight for Intra-CBL in Balance Loss"
)
parser.add_argument(
    "-balance_gamma", type=float, default=1.0,
    help="weight for Dice Loss in Balance Loss"
)
parser.add_argument(
    "-stage_switch_epoch", type=int, default=50,
    help="epoch to switch from stage 1 to stage 2"
)
parser.add_argument(
    "-intra_threshold", type=float, default=0.9,
    help="confidence threshold for Intra-CBL"
)
```

**位置3: 损失函数定义 (约第284行)**
```python
# === 损失函数初始化 ===
if args.loss_type == "balance" and BALANCE_LOSS_AVAILABLE:
    balance_loss = BalanceLoss(
        alpha=args.balance_alpha,
        beta=args.balance_beta,
        gamma=args.balance_gamma,
        threshold=args.intra_threshold,
        stage=1
    )
    use_balance_loss = True
    print(f"Using Balance Loss with alpha={args.balance_alpha}, "
          f"beta={args.balance_beta}, gamma={args.balance_gamma}")
else:
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    use_balance_loss = False
    print("Using original Dice + CE Loss")
```

**位置4: 训练循环 (约第314行)**
```python
for epoch in range(start_epoch, num_epochs):
    # === 阶段切换 ===
    if use_balance_loss and epoch == args.stage_switch_epoch:
        balance_loss.set_stage(2)
        print(f"[Epoch {epoch}] Switched Balance Loss to Stage 2")

    epoch_loss = 0
    epoch_loss_dict = {'inter_cbl': 0, 'intra_cbl': 0, 'dice': 0}

    for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        boxes_np = boxes.detach().cpu().numpy()
        image, gt2D = image.to(device), gt2D.to(device)

        if args.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                medsam_pred = medsam_model(image, boxes_np)
                if use_balance_loss:
                    loss, loss_dict = balance_loss(medsam_pred, gt2D)
                else:
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            medsam_pred = medsam_model(image, boxes_np)
            if use_balance_loss:
                loss, loss_dict = balance_loss(medsam_pred, gt2D)
                # 累积各分量损失
                for k, v in loss_dict.items():
                    if k in epoch_loss_dict:
                        epoch_loss_dict[k] += v
            else:
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        iter_num += 1

    # === 记录日志 ===
    epoch_loss /= (step + 1)
    if use_balance_loss:
        for k in epoch_loss_dict:
            epoch_loss_dict[k] /= (step + 1)

    if args.use_wandb:
        log_dict = {"epoch_loss": epoch_loss, "epoch": epoch}
        if use_balance_loss:
            log_dict.update(epoch_loss_dict)
            log_dict["stage"] = balance_loss.stage
        wandb.log(log_dict)
```

---

### 2.2 `segment_anything/modeling/mask_decoder.py` 修改 (可选)

此修改为可选，用于深度集成注意力模块。

#### 修改清单

| 行号范围 | 修改类型 | 描述 |
|----------|----------|------|
| 8-13 | 新增 | 导入注意力模块 |
| 44-50 | 新增 | 添加注意力融合相关属性 |
| 76-83 | 修改 | forward方法增加support参数 |

#### 详细修改

```python
# 位置1: 导入 (约第8行后)
try:
    from modules import AttentionCrossBlock
    ATTENTION_AVAILABLE = True
except ImportError:
    ATTENTION_AVAILABLE = False

# 位置2: __init__ (约第74行后)
def __init__(self, ..., use_support_attention: bool = False):
    ...
    # 新增
    self.use_support_attention = use_support_attention
    if use_support_attention and ATTENTION_AVAILABLE:
        self.attention_fusion = AttentionCrossBlock(
            embed_dim=transformer_dim,
            num_heads=8
        )

# 位置3: forward (约第76行)
def forward(
    self,
    image_embeddings: torch.Tensor,
    image_pe: torch.Tensor,
    sparse_prompt_embeddings: torch.Tensor,
    dense_prompt_embeddings: torch.Tensor,
    multimask_output: bool,
    support_features: Optional[torch.Tensor] = None,  # 新增
) -> Tuple[torch.Tensor, torch.Tensor]:

    # 新增: 支持集特征融合
    if self.use_support_attention and support_features is not None:
        image_embeddings = self.attention_fusion(
            image_embeddings,
            support_features
        )

    # 原有代码继续...
```

---

## 3. 代码风格规范

### 3.1 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 类名 | PascalCase | `BalanceLoss`, `AttentionCrossBlock` |
| 函数名 | snake_case | `compute_loss`, `get_hard_samples` |
| 变量名 | snake_case | `fg_mask`, `support_features` |
| 常量 | UPPER_SNAKE_CASE | `DEFAULT_THRESHOLD`, `MIN_SAMPLES` |
| 私有方法 | _prefix | `_compute_weights`, `_validate_input` |

### 3.2 文档字符串

```python
def function_name(arg1: Type1, arg2: Type2 = default) -> ReturnType:
    """
    简短描述（一行）

    详细描述（可选，多行）

    Args:
        arg1: 参数1描述
        arg2: 参数2描述，默认值说明

    Returns:
        返回值描述

    Raises:
        ExceptionType: 异常触发条件

    Example:
        >>> result = function_name(value1, value2)
    """
    pass
```

### 3.3 类型注解

所有公共函数和方法必须有类型注解：

```python
from typing import Optional, Tuple, List, Dict, Union

def forward(
    self,
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ...
```

---

## 4. 测试规范

### 4.1 单元测试

每个模块应有对应的测试文件：

```
tests/
├── test_balance_loss.py
├── test_attention_cross_block.py
├── test_fewshot_dataset.py
└── test_medsam_fss.py
```

### 4.2 测试用例设计

```python
# tests/test_balance_loss.py

import unittest
import torch
from losses import BalanceLoss, InterClassBalanceLoss, IntraClassBalanceLoss

class TestInterClassBalanceLoss(unittest.TestCase):

    def setUp(self):
        self.loss_fn = InterClassBalanceLoss()
        self.batch_size = 2
        self.height, self.width = 256, 256

    def test_output_shape(self):
        """测试输出为标量"""
        pred = torch.randn(self.batch_size, 1, self.height, self.width)
        target = torch.zeros(self.batch_size, 1, self.height, self.width)
        target[:, :, 100:150, 100:150] = 1

        loss = self.loss_fn(pred, target)
        self.assertEqual(loss.dim(), 0, "Loss should be a scalar")

    def test_no_foreground(self):
        """测试全背景情况"""
        pred = torch.randn(self.batch_size, 1, self.height, self.width)
        target = torch.zeros(self.batch_size, 1, self.height, self.width)

        loss = self.loss_fn(pred, target)
        self.assertFalse(torch.isnan(loss), "Loss should not be NaN")

    def test_all_foreground(self):
        """测试全前景情况"""
        pred = torch.randn(self.batch_size, 1, self.height, self.width)
        target = torch.ones(self.batch_size, 1, self.height, self.width)

        loss = self.loss_fn(pred, target)
        self.assertFalse(torch.isnan(loss), "Loss should not be NaN")

    def test_gradient_flow(self):
        """测试梯度流动"""
        pred = torch.randn(self.batch_size, 1, self.height, self.width, requires_grad=True)
        target = torch.zeros(self.batch_size, 1, self.height, self.width)
        target[:, :, 100:150, 100:150] = 1

        loss = self.loss_fn(pred, target)
        loss.backward()

        self.assertIsNotNone(pred.grad, "Gradient should exist")
        self.assertFalse(torch.isnan(pred.grad).any(), "Gradient should not contain NaN")


class TestBalanceLoss(unittest.TestCase):

    def test_stage_switching(self):
        """测试阶段切换"""
        loss_fn = BalanceLoss(stage=1)
        self.assertEqual(loss_fn.stage, 1)

        loss_fn.set_stage(2)
        self.assertEqual(loss_fn.stage, 2)

    def test_loss_dict_keys(self):
        """测试返回字典包含所有键"""
        loss_fn = BalanceLoss(stage=2)
        pred = torch.randn(2, 1, 64, 64)
        target = torch.zeros(2, 1, 64, 64)
        target[:, :, 20:40, 20:40] = 1

        loss, loss_dict = loss_fn(pred, target)

        expected_keys = ['inter_cbl', 'intra_cbl', 'dice', 'total']
        for key in expected_keys:
            self.assertIn(key, loss_dict)


if __name__ == '__main__':
    unittest.main()
```

### 4.3 运行测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_balance_loss.py

# 带覆盖率报告
python -m pytest tests/ --cov=losses --cov=modules
```

---

**文档结束**
