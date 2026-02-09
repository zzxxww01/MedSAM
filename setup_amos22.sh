#!/bin/bash
# AMOS22 数据集准备脚本
# 用途：解压数据集并验证完整性

set -e  # 遇到错误立即退出

echo "========================================="
echo "AMOS22 数据集准备"
echo "========================================="

# 切换到数据目录
cd ~/chengang/zxw/MedSAM/data

# 检查zip文件是否存在
if [ ! -f "amos22.zip" ]; then
    echo "错误: amos22.zip 不存在"
    exit 1
fi

# 解压数据集
echo "正在解压 amos22.zip..."
unzip -q amos22.zip

# 检查解压后的目录
echo ""
echo "检查解压后的目录结构..."
ls -la

# 尝试找到实际的目录名
if [ -d "AMOS22" ]; then
    DATA_DIR="AMOS22"
elif [ -d "amos22" ]; then
    DATA_DIR="amos22"
elif [ -d "amos" ]; then
    DATA_DIR="amos"
else
    echo "警告: 未找到标准目录名，请手动检查"
    ls -la
    exit 1
fi

echo ""
echo "找到数据目录: $DATA_DIR"

# 检查子目录
echo ""
echo "检查子目录..."
ls -la $DATA_DIR/

# 统计文件数量
if [ -d "$DATA_DIR/imagesTr" ]; then
    IMG_COUNT=$(ls $DATA_DIR/imagesTr/*.nii.gz 2>/dev/null | wc -l)
    echo "训练图像数量: $IMG_COUNT"
fi

if [ -d "$DATA_DIR/labelsTr" ]; then
    GT_COUNT=$(ls $DATA_DIR/labelsTr/*.nii.gz 2>/dev/null | wc -l)
    echo "训练标签数量: $GT_COUNT"
fi

echo ""
echo "========================================="
echo "数据集准备完成！"
echo "实际目录名: $DATA_DIR"
echo "========================================="
echo ""
echo "下一步: 修改 pre_AMOS22.py 中的路径为:"
echo "  nii_path = \"data/$DATA_DIR/imagesTr\""
echo "  gt_path = \"data/$DATA_DIR/labelsTr\""
