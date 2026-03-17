#!/bin/bash
# generate_all_figures.sh
# 一键运行所有图片生成脚本
#
# 用法: cd ~/chengang/zxw/MedSAM && bash scripts/generate_all_figures.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"  # cd to project root

echo "======================================"
echo " 生成全部论文图片"
echo "======================================"

# ── 1. 纯 matplotlib 图（无数据依赖） ──
echo ""
echo "=== 阶段 1: 纯 matplotlib 图 ==="

echo "[1/5] Fig.1 技术路线图..."
python scripts/generate_fig1_roadmap.py

echo "[2/5] Fig.2 Balance Loss 流程图..."
python scripts/generate_fig2_balance_loss.py

echo "[3/5] Fig.4 LG-Adapter 架构图..."
python scripts/generate_fig4_adapter.py

echo "[4/5] Fig.5 Adapter 管线位置图..."
python scripts/generate_fig5_pipeline.py

echo "[5/5] Fig.8 MedSAM 架构图 (可选)..."
python scripts/generate_fig8_medsam_arch.py

# ── 2. 训练曲线图（需要日志文件） ──
echo ""
echo "=== 阶段 2: 训练曲线图 ==="

LOG_A0="work_dir/baseline_train.log"
LOG_A3R3="work_dir/exp_logs/A3R3_train.log"

if [ -f "$LOG_A0" ] && [ -f "$LOG_A3R3" ]; then
    echo "[OK] 找到训练日志，生成曲线图..."
    python scripts/generate_fig3_training_curves.py \
        --log_a0 "$LOG_A0" \
        --log_a3r3 "$LOG_A3R3"
else
    echo "[SKIP] 训练日志不存在，跳过 Fig.3"
    [ ! -f "$LOG_A0" ] && echo "  缺少: $LOG_A0"
    [ ! -f "$LOG_A3R3" ] && echo "  缺少: $LOG_A3R3"
fi

# ── 3. 掩码对比图（需要预测 NPZ） ──
echo ""
echo "=== 阶段 3: 掩码对比图 ==="

PRED_DIR="work_dir/eval_predictions"
if [ -d "${PRED_DIR}/A0" ] && [ -d "${PRED_DIR}/C3" ]; then
    echo "[OK] 找到预测 NPZ，生成对比图..."
    python scripts/generate_fig6_mask_comparison.py
    python scripts/generate_fig7_boundary_detail.py
else
    echo "[SKIP] 预测 NPZ 不存在，跳过 Fig.6 和 Fig.7"
    echo "  请先运行: bash scripts/save_all_predictions.sh"
fi

# ── 4. 逐器官可视化图（直接使用论文表格数据） ──
echo ""
echo "=== 阶段 4: 逐器官可视化 ==="
python scripts/generate_fig10_per_organ_dsc.py

# ── 汇总 ──
echo ""
echo "======================================"
echo " 生成完毕！图片位于:"
echo "======================================"
ls -la thesis-medsam/figures/*.pdf 2>/dev/null || echo "  (暂无 PDF 文件)"
echo ""
echo "下一步: 将图片复制到 thesis-medsam/figures/ 并编译 LaTeX"
