# AMOS22 数据集解压和准备指南

## 当前状态
数据集文件已下载但未解压：
- `amos22.zip` - 需要解压

## 解压步骤

### 1. 解压AMOS22数据集
```bash
cd ~/chengang/zxw/MedSAM/data
unzip amos22.zip
```

### 2. 检查解压后的目录结构
```bash
ls -la amos22/
```

预期目录结构应该是：
```
amos22/
├── imagesTr/      # 训练图像
├── labelsTr/      # 训练标签
├── imagesVa/      # 验证图像（可能）
├── labelsVa/      # 验证标签（可能）
└── dataset.json   # 数据集描述文件
```

### 3. 如果目录结构不同，需要调整

**情况A：如果解压后是 `AMOS22/` 目录**
```bash
mv amos22 AMOS22
```

**情况B：如果解压后直接是 `imagesTr/` 等文件夹**
```bash
mkdir -p AMOS22
mv imagesTr labelsTr AMOS22/
```

### 4. 修改预处理脚本路径

根据实际解压后的目录结构，修改 `pre_AMOS22.py` 中的路径：
```python
# 当前设置
nii_path = "data/AMOS22/imagesTr"
gt_path = "data/AMOS22/labelsTr"

# 如果实际是 amos22（小写），改为：
nii_path = "data/amos22/imagesTr"
gt_path = "data/amos22/labelsTr"
```

### 5. 验证数据完整性
```bash
# 检查图像数量
ls data/AMOS22/imagesTr/*.nii.gz | wc -l  # 应该是600（500 CT + 100 MRI）

# 检查标签数量
ls data/AMOS22/labelsTr/*.nii.gz | wc -l  # 应该是600
```

## 快速执行命令

```bash
# 一键执行
cd ~/chengang/zxw/MedSAM/data
unzip amos22.zip
ls -la
```
