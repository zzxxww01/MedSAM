#!/bin/bash
# eval_supplementary.sh
# 补充实验评估脚本
# 对 6 个补充实验依次运行 eval_medsam_npz.py，输出 CSV/JSON
#
# 用法: cd ~/chengang/zxw/MedSAM && bash scripts/eval_supplementary.sh

set -e

DATA_ROOT="data/npy/CT_Abd"
EVAL_DIR="work_dir/eval_metrics"
mkdir -p "${EVAL_DIR}"

echo "======================================"
echo " 补充实验评估"
echo "======================================"

find_best_ckpt() {
    local pattern="$1"
    local ckpt
    ckpt=$(find work_dir -name "medsam_model_best.pth" -path "*${pattern}*" 2>/dev/null | sort | tail -1)
    echo "$ckpt"
}

# ── 1. C2-r8 ──
echo ""
echo "[1/6] C2-r8 (LoRA rank=8)..."
CKPT=$(find_best_ckpt "C2-r8")
if [ -n "$CKPT" ]; then
    python eval_medsam_npz.py \
        --data_root "$DATA_ROOT" \
        --checkpoint "$CKPT" \
        --exp_name C2-r8 \
        -use_lora true -lora_rank 8 \
        --out_csv "${EVAL_DIR}/C2-r8_case_metrics.csv" \
        --out_json "${EVAL_DIR}/C2-r8_summary.json"
else
    echo "[WARN] C2-r8 checkpoint not found, skipping."
fi

# ── 2. C2-r16 ──
echo ""
echo "[2/6] C2-r16 (LoRA rank=16)..."
CKPT=$(find_best_ckpt "C2-r16")
if [ -n "$CKPT" ]; then
    python eval_medsam_npz.py \
        --data_root "$DATA_ROOT" \
        --checkpoint "$CKPT" \
        --exp_name C2-r16 \
        -use_lora true -lora_rank 16 \
        --out_csv "${EVAL_DIR}/C2-r16_case_metrics.csv" \
        --out_json "${EVAL_DIR}/C2-r16_summary.json"
else
    echo "[WARN] C2-r16 checkpoint not found, skipping."
fi

# ── 3. A0-focal ──
echo ""
echo "[3/6] A0-focal (Dice+Focal Loss)..."
CKPT=$(find_best_ckpt "A0-focal")
if [ -n "$CKPT" ]; then
    python eval_medsam_npz.py \
        --data_root "$DATA_ROOT" \
        --checkpoint "$CKPT" \
        --exp_name A0-focal \
        --out_csv "${EVAL_DIR}/A0-focal_case_metrics.csv" \
        --out_json "${EVAL_DIR}/A0-focal_summary.json"
else
    echo "[WARN] A0-focal checkpoint not found, skipping."
fi

# ── 4. A2-t0.5 ──
echo ""
echo "[4/6] A2-t0.5 (Intra-CBL tau=0.5)..."
CKPT=$(find_best_ckpt "A2-t0.5")
if [ -n "$CKPT" ]; then
    python eval_medsam_npz.py \
        --data_root "$DATA_ROOT" \
        --checkpoint "$CKPT" \
        --exp_name A2-t0.5 \
        --out_csv "${EVAL_DIR}/A2-t0.5_case_metrics.csv" \
        --out_json "${EVAL_DIR}/A2-t0.5_summary.json"
else
    echo "[WARN] A2-t0.5 checkpoint not found, skipping."
fi

# ── 5. A2-t0.7 ──
echo ""
echo "[5/6] A2-t0.7 (Intra-CBL tau=0.7)..."
CKPT=$(find_best_ckpt "A2-t0.7")
if [ -n "$CKPT" ]; then
    python eval_medsam_npz.py \
        --data_root "$DATA_ROOT" \
        --checkpoint "$CKPT" \
        --exp_name A2-t0.7 \
        --out_csv "${EVAL_DIR}/A2-t0.7_case_metrics.csv" \
        --out_json "${EVAL_DIR}/A2-t0.7_summary.json"
else
    echo "[WARN] A2-t0.7 checkpoint not found, skipping."
fi

# ── 6. A2-t0.8 ──
echo ""
echo "[6/6] A2-t0.8 (Intra-CBL tau=0.8)..."
CKPT=$(find_best_ckpt "A2-t0.8")
if [ -n "$CKPT" ]; then
    python eval_medsam_npz.py \
        --data_root "$DATA_ROOT" \
        --checkpoint "$CKPT" \
        --exp_name A2-t0.8 \
        --out_csv "${EVAL_DIR}/A2-t0.8_case_metrics.csv" \
        --out_json "${EVAL_DIR}/A2-t0.8_summary.json"
else
    echo "[WARN] A2-t0.8 checkpoint not found, skipping."
fi

echo ""
echo "======================================"
echo " 评估完成! 结果保存在: ${EVAL_DIR}/"
echo "======================================"
ls -la "${EVAL_DIR}"/ 2>/dev/null
