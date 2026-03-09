#!/bin/bash
# save_all_predictions.sh
# 一键运行 A0/A3R3/C2/C3 评估并保存预测 NPZ
# 用于 Fig.6 和 Fig.7 的掩码对比图
#
# 用法: cd ~/chengang/zxw/MedSAM && bash scripts/save_all_predictions.sh

set -e

DATA_ROOT="data/npy/CT_Abd"
PRED_BASE="work_dir/eval_predictions"

echo "======================================"
echo " 保存全部实验预测 NPZ"
echo "======================================"

# ── A0 Baseline ──
echo ""
echo "[1/4] A0 Baseline..."
python eval_medsam_npz.py \
    --data_root "$DATA_ROOT" \
    --checkpoint "work_dir/MedSAM-Baseline-20260208-1953/medsam_model_best.pth" \
    --exp_name A0 \
    --out_csv "work_dir/eval_metrics/A0_case_metrics.csv" \
    --out_json "work_dir/eval_metrics/A0_summary.json" \
    --save_pred_dir "${PRED_BASE}/A0"

# ── A3R3 Balance Loss ──
echo ""
echo "[2/4] A3R3 Balance Loss..."
# 注意: A3R3 checkpoint 路径需要确认，以下为候选路径
A3R3_CKPT=$(find work_dir -name "medsam_model_best.pth" -path "*A3R3*" -o -name "medsam_model_best.pth" -path "*BalanceR3*" 2>/dev/null | head -1)
if [ -z "$A3R3_CKPT" ]; then
    echo "[WARN] A3R3 checkpoint not found, trying known paths..."
    A3R3_CKPT="work_dir/MedSAM-FLARE22-A3R3-Balance-*/medsam_model_best.pth"
    A3R3_CKPT=$(ls $A3R3_CKPT 2>/dev/null | head -1)
fi
if [ -n "$A3R3_CKPT" ]; then
    python eval_medsam_npz.py \
        --data_root "$DATA_ROOT" \
        --checkpoint "$A3R3_CKPT" \
        --exp_name A3R3 \
        --out_csv "work_dir/eval_metrics/A3R3_case_metrics.csv" \
        --out_json "work_dir/eval_metrics/A3R3_summary.json" \
        --save_pred_dir "${PRED_BASE}/A3R3"
else
    echo "[ERROR] A3R3 checkpoint not found! Please specify manually."
fi

# ── C2 LoRA ──
echo ""
echo "[3/4] C2 LoRA..."
C2_CKPT=$(find work_dir -name "medsam_model_best.pth" -path "*C2*" -o -name "medsam_model_best.pth" -path "*LoRA*" 2>/dev/null | head -1)
if [ -n "$C2_CKPT" ]; then
    python eval_medsam_npz.py \
        --data_root "$DATA_ROOT" \
        --checkpoint "$C2_CKPT" \
        --exp_name C2 \
        -use_lora true \
        -lora_rank 4 \
        --out_csv "work_dir/eval_metrics/C2_case_metrics.csv" \
        --out_json "work_dir/eval_metrics/C2_summary.json" \
        --save_pred_dir "${PRED_BASE}/C2"
else
    echo "[ERROR] C2 checkpoint not found! Please specify manually."
fi

# ── C3 LG-Adapter ──
echo ""
echo "[4/4] C3 LG-Adapter..."
C3_CKPT=$(find work_dir -name "medsam_model_best.pth" -path "*C3*" -o -name "medsam_model_best.pth" -path "*LG*" 2>/dev/null | head -1)
if [ -n "$C3_CKPT" ]; then
    python eval_medsam_npz.py \
        --data_root "$DATA_ROOT" \
        --checkpoint "$C3_CKPT" \
        --exp_name C3 \
        -use_lg_adapter true \
        --out_csv "work_dir/eval_metrics/C3_case_metrics.csv" \
        --out_json "work_dir/eval_metrics/C3_summary.json" \
        --save_pred_dir "${PRED_BASE}/C3"
else
    echo "[ERROR] C3 checkpoint not found! Please specify manually."
fi

echo ""
echo "======================================"
echo " 完成！预测 NPZ 保存在: ${PRED_BASE}/"
echo "======================================"
ls -la "${PRED_BASE}"/ 2>/dev/null
