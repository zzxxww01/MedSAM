#!/bin/bash
# run_supplementary_experiments.sh
# 补充实验训练脚本 (单节点多 GPU DDP)
#
# 包含 6 个实验:
#   C2-r8   : LoRA rank=8
#   C2-r16  : LoRA rank=16
#   A0-focal: Dice+Focal Loss 基线
#   A2-t0.5 : Intra-CBL tau=0.5
#   A2-t0.7 : Intra-CBL tau=0.7
#   A2-t0.8 : Intra-CBL tau=0.8
#
# 用法: cd ~/chengang/zxw/MedSAM && bash scripts/run_supplementary_experiments.sh
# 单独运行某个实验: bash scripts/run_supplementary_experiments.sh <实验编号>
#   例: bash scripts/run_supplementary_experiments.sh 1   # 只运行 C2-r8

set -e

DATA_ROOT="data/npy/CT_Abd"
CHECKPOINT="work_dir/medsam_vit_b.pth"
COMMON_ARGS="-batch_size 8 -num_epochs 200 -lr 1e-4 -weight_decay 0.01"
NGPUS=$(python -c "import torch; print(torch.cuda.device_count())")

echo "======================================"
echo " 补充实验训练 (${NGPUS} GPUs)"
echo "======================================"

run_experiment() {
    local exp_name="$1"
    shift
    echo ""
    echo ">>> [${exp_name}] Starting..."
    torchrun --nproc_per_node="${NGPUS}" train_fss.py \
        -i "${DATA_ROOT}" \
        -checkpoint "${CHECKPOINT}" \
        -task_name "MedSAM-FLARE22-${exp_name}" \
        ${COMMON_ARGS} \
        "$@"
    echo ">>> [${exp_name}] Done."
}

# 允许只运行指定编号的实验
EXP_ID="${1:-all}"

# ── 1. C2-r8: LoRA rank=8 ──
if [ "$EXP_ID" = "all" ] || [ "$EXP_ID" = "1" ]; then
    run_experiment "C2-r8" \
        -use_lora true -lora_rank 8 \
        -loss_type balance --balance_alpha 0.5
fi

# ── 2. C2-r16: LoRA rank=16 ──
if [ "$EXP_ID" = "all" ] || [ "$EXP_ID" = "2" ]; then
    run_experiment "C2-r16" \
        -use_lora true -lora_rank 16 \
        -loss_type balance --balance_alpha 0.5
fi

# ── 3. A0-focal: Dice+Focal Loss baseline ──
if [ "$EXP_ID" = "all" ] || [ "$EXP_ID" = "3" ]; then
    run_experiment "A0-focal" \
        -loss_type focal --focal_gamma 2.0 --focal_alpha 0.25
fi

# ── 4. A2-t0.5: Intra-CBL tau=0.5 ──
if [ "$EXP_ID" = "all" ] || [ "$EXP_ID" = "4" ]; then
    run_experiment "A2-t0.5" \
        -loss_type intra_cbl --intra_threshold 0.5
fi

# ── 5. A2-t0.7: Intra-CBL tau=0.7 ──
if [ "$EXP_ID" = "all" ] || [ "$EXP_ID" = "5" ]; then
    run_experiment "A2-t0.7" \
        -loss_type intra_cbl --intra_threshold 0.7
fi

# ── 6. A2-t0.8: Intra-CBL tau=0.8 ──
if [ "$EXP_ID" = "all" ] || [ "$EXP_ID" = "6" ]; then
    run_experiment "A2-t0.8" \
        -loss_type intra_cbl --intra_threshold 0.8
fi

echo ""
echo "======================================"
echo " 全部训练完成!"
echo "======================================"
