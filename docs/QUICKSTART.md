# 快速开始指南

> 本指南帮助你快速开始MedSAM改进项目的开发

---

## 1. 环境准备

### 1.1 创建虚拟环境

```bash
# 使用conda创建环境
conda create -n medsam python=3.10 -y
conda activate medsam
```

### 1.2 安装依赖

```bash
# 安装PyTorch (根据CUDA版本选择)
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install monai
pip install scikit-image
pip install matplotlib
pip install tqdm
pip install wandb
pip install nibabel
pip install SimpleITK

# 安装MedSAM (开发模式)
cd MedSAM
pip install -e .
```

### 1.3 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from segment_anything import sam_model_registry; print('MedSAM: OK')"
```

---

## 2. 数据准备

### 2.1 下载预训练权重

```bash
# 创建目录
mkdir -p work_dir/SAM

# 下载SAM-ViT-B权重
wget -O work_dir/SAM/sam_vit_b_01ec64.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 2.2 准备数据集

数据应按以下格式组织：

```
data/
└── npy/
    └── CT_Abd/           # 数据集名称
        ├── imgs/         # 图像文件
        │   ├── case001_slice001.npy
        │   ├── case001_slice002.npy
        │   └── ...
        └── gts/          # 标签文件
            ├── case001_slice001.npy
            ├── case001_slice002.npy
            └── ...
```

数据格式要求：
- 图像: `[1024, 1024, 3]`, 值范围 `[0, 1]`
- 标签: `[256, 256]`, 整数标签 (0=背景, 1,2,...=前景类别)

---

## 3. 运行Baseline

### 3.1 验证数据加载

```bash
# 运行数据sanity check
python train_one_gpu.py -i data/npy/CT_Abd
# 会生成 data_sanitycheck.png
```

### 3.2 训练Baseline模型

```bash
python train_one_gpu.py \
    -i data/npy/CT_Abd \
    -task_name MedSAM-Baseline \
    -model_type vit_b \
    -checkpoint work_dir/SAM/sam_vit_b_01ec64.pth \
    -num_epochs 100 \
    -batch_size 4 \
    -lr 0.0001 \
    --device cuda:0
```

### 3.3 监控训练 (可选)

```bash
# 使用wandb监控
python train_one_gpu.py \
    ... \
    -use_wandb True
```

---

## 4. 开发工作流

### 4.1 分支管理

```bash
# 查看当前分支
git branch

# 创建新功能分支
git checkout -b feature/balance-loss

# 开发完成后提交
git add .
git commit -m "feat(loss): implement Balance Loss"

# 推送到远程
git push origin feature/balance-loss
```

### 4.2 代码开发顺序

1. **第一阶段: Balance Loss**
   ```
   losses/__init__.py
   losses/balance_loss.py
   train_balance_loss.py (或修改 train_one_gpu.py)
   ```

2. **第二阶段: Attention模块**
   ```
   modules/__init__.py
   modules/attention_cross_block.py
   ```

3. **第三阶段: Few-Shot支持**
   ```
   datasets/__init__.py
   datasets/fewshot_dataset.py
   models/__init__.py
   models/medsam_fss.py
   train_fss.py
   ```

---

## 5. 测试验证

### 5.1 单元测试

```bash
# 测试Balance Loss
python -c "
import torch
from losses import BalanceLoss

# 创建测试数据
pred = torch.randn(2, 1, 256, 256)
target = torch.zeros(2, 1, 256, 256)
target[:, :, 100:150, 100:150] = 1

# 测试损失计算
loss_fn = BalanceLoss()
loss, loss_dict = loss_fn(pred, target)
print(f'Loss: {loss.item():.4f}')
print(f'Components: {loss_dict}')
print('Test passed!')
"
```

### 5.2 集成测试

```bash
# 使用Balance Loss训练几个epoch
python train_balance_loss.py \
    -i data/npy/CT_Abd \
    -task_name Test-BalanceLoss \
    -loss_type balance \
    -num_epochs 5 \
    -batch_size 2
```

---

## 6. 常用命令

### 训练命令

```bash
# Baseline训练
python train_one_gpu.py -i data/npy/CT_Abd -num_epochs 200

# Balance Loss训练
python train_balance_loss.py -i data/npy/CT_Abd -loss_type balance

# Few-Shot训练
python train_fss.py -i data/npy/CT_Abd -num_support 5 -use_attention True
```

### Git命令

```bash
# 查看状态
git status

# 查看修改
git diff

# 提交更改
git add .
git commit -m "message"

# 切换分支
git checkout branch_name

# 创建并切换分支
git checkout -b new_branch
```

### 实验管理

```bash
# 查看wandb运行
wandb runs

# 导出实验结果
python utils/export_results.py --run_id xxx

# 生成可视化
python utils/visualize.py --checkpoint work_dir/xxx/best.pth
```

---

## 7. 常见问题

### Q1: CUDA内存不足

```bash
# 方法1: 减小batch size
-batch_size 2

# 方法2: 使用混合精度训练
-use_amp

# 方法3: 梯度累积 (需要修改代码)
```

### Q2: 训练不收敛

- 检查学习率是否合适
- 检查数据预处理是否正确
- 检查损失函数数值范围

### Q3: 数据加载错误

```python
# 检查数据格式
import numpy as np
img = np.load('data/npy/CT_Abd/imgs/xxx.npy')
gt = np.load('data/npy/CT_Abd/gts/xxx.npy')
print(f'Image shape: {img.shape}, range: [{img.min()}, {img.max()}]')
print(f'GT shape: {gt.shape}, unique: {np.unique(gt)}')
```

---

## 8. 项目结构 (开发后)

```
MedSAM/
├── segment_anything/      # 原始SAM模块 (尽量不修改)
├── losses/                # [新] 损失函数
│   ├── __init__.py
│   └── balance_loss.py
├── modules/               # [新] 自定义模块
│   ├── __init__.py
│   └── attention_cross_block.py
├── datasets/              # [新] 数据集
│   ├── __init__.py
│   └── fewshot_dataset.py
├── models/                # [新] 模型定义
│   ├── __init__.py
│   └── medsam_fss.py
├── configs/               # [新] 配置文件
│   └── balance_loss.yaml
├── tests/                 # [新] 测试文件
│   ├── test_balance_loss.py
│   └── test_attention.py
├── docs/                  # [新] 文档
│   ├── PROJECT_PLAN.md
│   ├── CODE_IMPLEMENTATION.md
│   ├── EXPERIMENT_LOG.md
│   └── QUICKSTART.md
├── train_one_gpu.py       # 原始训练脚本
├── train_balance_loss.py  # [新] Balance Loss训练
└── train_fss.py           # [新] Few-Shot训练
```

---

**文档结束**
