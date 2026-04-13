#!/usr/bin/env python3
"""生成失败案例可视化图 (3×4: 3类失败 × GT/Baseline/BL/BL+MSL)

直接在服务器上运行，自动从 eval_predictions 中筛选代表性失败切片。

用法:
  cd ~/chengang/zxw/MedSAM
  python scripts/generate_failure_cases.py

前提: 已运行 bash scripts/save_all_predictions.sh 保存了 A0/A3R3/C3 的预测 NPZ。

输出: thesis-medsam/figures/failure_cases.pdf
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from skimage import measure

plt.rcParams["font.family"] = ["Times New Roman", "SimSun", "serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

# ═══════════════════════════════════════════
# 路径配置（根据你的服务器目录调整）
# ═══════════════════════════════════════════
DATA_ROOT = "data/npy/CT_Abd"                    # 原始数据（含 imgs）
PRED_DIRS = {
    "A0":   "work_dir/eval_predictions/A0",      # Baseline
    "A3R3": "work_dir/eval_predictions/A3R3",    # BL
    "C3":   "work_dir/eval_predictions/C3",      # BL+MSL
}
OUT_DIR = "thesis-medsam/figures"

# 器官 ID 映射
ORGAN_MAP = {
    1: "Liver", 2: "R.Kidney", 3: "Spleen", 4: "Pancreas",
    5: "Aorta", 6: "IVC", 7: "RAG", 8: "LAG",
    9: "Gallbladder", 10: "Esophagus", 11: "Stomach",
    12: "Duodenum", 13: "L.Kidney",
}

# 三类失败要找的目标器官
FAILURE_TARGETS = {
    "boundary":  [4],          # 胰腺 - 边界弱/灰度过渡
    "small":     [10, 7, 8],   # 食管/RAG/LAG - 小体积细长
    "adjacent":  [12],         # 十二指肠 - 与胰腺邻接粘连
}


def compute_slice_dice(pred, gt, label_id):
    """计算单个切片上某器官的 Dice"""
    p = (pred == label_id)
    g = (gt == label_id)
    inter = np.logical_and(p, g).sum()
    union = p.sum() + g.sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return 2 * inter / union


def find_failure_slice(data_files, pred_c3_files, target_labels, mode="worst"):
    """
    在 C3 (BL+MSL) 的预测中找到目标器官表现最差的切片。

    Returns: (case_idx, slice_idx, organ_id, dice_value)
    """
    best_failure = None  # (dice, case_idx, slice_idx, organ_id)

    for case_idx, (data_f, pred_f) in enumerate(zip(data_files, pred_c3_files)):
        pred_data = np.load(pred_f)
        segs = pred_data["segs"]
        gts = pred_data["gts"]

        for z in range(gts.shape[0]):
            gt_slice = gts[z]
            seg_slice = segs[z]

            for label_id in target_labels:
                gt_area = (gt_slice == label_id).sum()
                if gt_area < 50:  # 跳过太小的区域
                    continue

                dice = compute_slice_dice(seg_slice, gt_slice, label_id)

                # 寻找 Dice 较低但不为 0 的切片（有误差但不是完全漏掉）
                if 0.3 < dice < 0.92:
                    if best_failure is None or dice < best_failure[0]:
                        best_failure = (dice, case_idx, z, label_id)

    # 如果找不到 0.3-0.92 范围的，放宽条件
    if best_failure is None:
        for case_idx, (data_f, pred_f) in enumerate(zip(data_files, pred_c3_files)):
            pred_data = np.load(pred_f)
            segs = pred_data["segs"]
            gts = pred_data["gts"]

            for z in range(gts.shape[0]):
                gt_slice = gts[z]
                seg_slice = segs[z]

                for label_id in target_labels:
                    gt_area = (gt_slice == label_id).sum()
                    if gt_area < 30:
                        continue
                    dice = compute_slice_dice(seg_slice, gt_slice, label_id)
                    if dice < 0.98:
                        if best_failure is None or dice < best_failure[0]:
                            best_failure = (dice, case_idx, z, label_id)

    return best_failure


def get_organ_bbox(mask, label_id, pad=30):
    """获取器官区域的 bbox 用于放大显示"""
    ys, xs = np.where(mask == label_id)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x_min = max(0, xs.min() - pad)
    x_max = min(w, xs.max() + pad)
    y_min = max(0, ys.min() - pad)
    y_max = min(h, ys.max() + pad)
    return y_min, y_max, x_min, x_max


def overlay_contours(ax, ct_slice, gt_mask, pred_mask, organ_id, title="",
                     crop_bbox=None):
    """在 CT 切片上叠加 GT 轮廓(绿) 和预测轮廓(红)"""
    gt_binary = (gt_mask == organ_id).astype(float)
    pred_binary = (pred_mask == organ_id).astype(float)

    # 裁剪到器官区域
    if crop_bbox is not None:
        y1, y2, x1, x2 = crop_bbox
        ct_slice = ct_slice[y1:y2, x1:x2]
        gt_binary = gt_binary[y1:y2, x1:x2]
        pred_binary = pred_binary[y1:y2, x1:x2]

    ax.imshow(ct_slice, cmap="gray", vmin=0, vmax=1, aspect="equal")

    # GT 轮廓 - 绿色
    if gt_binary.max() > 0:
        contours = measure.find_contours(gt_binary, 0.5)
        for c in contours:
            ax.plot(c[:, 1], c[:, 0], color="#00FF00", linewidth=1.5, alpha=0.9)

    # 预测轮廓 - 红色
    if pred_binary.max() > 0:
        contours = measure.find_contours(pred_binary, 0.5)
        for c in contours:
            ax.plot(c[:, 1], c[:, 0], color="#FF3333", linewidth=1.5, alpha=0.9)

    # 标注差异区域：找 GT 有但预测没有的区域 (FN) 和反之 (FP)
    diff = np.logical_xor(gt_binary > 0, pred_binary > 0)
    if diff.sum() > 10:
        diff_ys, diff_xs = np.where(diff)
        # 在差异区域中心画黄色圆
        cy, cx = diff_ys.mean(), diff_xs.mean()
        radius = max(15, min(diff_ys.max() - diff_ys.min(),
                            diff_xs.max() - diff_xs.min()) / 2 + 5)
        circle = Circle((cx, cy), radius, fill=False,
                        edgecolor="yellow", linewidth=1.5, linestyle="--")
        ax.add_patch(circle)

    if title:
        ax.set_title(title, fontsize=9, pad=3)
    ax.axis("off")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 查找数据文件
    data_files = sorted(glob.glob(os.path.join(DATA_ROOT, "*.npz")))
    if not data_files:
        print(f"[ERROR] 原始数据目录为空: {DATA_ROOT}")
        sys.exit(1)

    # 查找预测文件
    pred_files = {}
    for exp_name, pred_dir in PRED_DIRS.items():
        files = sorted(glob.glob(os.path.join(pred_dir, "*.npz")))
        if not files:
            print(f"[ERROR] 预测目录为空: {pred_dir}")
            print("请先运行: bash scripts/save_all_predictions.sh")
            sys.exit(1)
        pred_files[exp_name] = files
        print(f"[OK] {exp_name}: {len(files)} 个预测文件")

    # 对齐文件列表（按文件名匹配）
    data_names = {os.path.basename(f): f for f in data_files}
    pred_names = {}
    for exp in ["A0", "A3R3", "C3"]:
        pred_names[exp] = {os.path.basename(f): f for f in pred_files[exp]}

    # 取交集
    common_names = sorted(
        set(data_names.keys()) &
        set(pred_names["A0"].keys()) &
        set(pred_names["A3R3"].keys()) &
        set(pred_names["C3"].keys())
    )
    print(f"\n共 {len(common_names)} 个可用病例")

    aligned_data = [data_names[n] for n in common_names]
    aligned_c3 = [pred_names["C3"][n] for n in common_names]

    # ════════════════════════════════════════
    # 自动筛选 3 类失败切片
    # ════════════════════════════════════════
    print("\n正在搜索失败案例...")
    failure_cases = []
    row_labels = [
        "(a) Weak boundary\n(Pancreas)",
        "(b) Small organ\n(Esophagus/Adrenal)",
        "(c) Adjacent organs\n(Duodenum)",
    ]

    for i, (fail_type, target_labels) in enumerate(FAILURE_TARGETS.items()):
        result = find_failure_slice(aligned_data, aligned_c3, target_labels)
        if result is None:
            print(f"[WARN] 未找到 {fail_type} 类型的失败切片，使用默认")
            result = (0.85, 0, 0, target_labels[0])

        dice, case_idx, slice_idx, organ_id = result
        case_name = common_names[case_idx]
        organ_name = ORGAN_MAP.get(organ_id, f"Label{organ_id}")
        print(f"  [{fail_type}] case={case_name}, slice={slice_idx}, "
              f"organ={organ_name}, C3_dice={dice:.4f}")
        failure_cases.append((case_idx, slice_idx, organ_id))

    # ════════════════════════════════════════
    # 绘图
    # ════════════════════════════════════════
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.5))
    col_titles = ["Ground Truth", "Baseline (A0)", "BL (A3R3)", "BL+MSL (C3)"]

    for row, (case_idx, slice_idx, organ_id) in enumerate(failure_cases):
        case_name = common_names[case_idx]

        # 加载原始 CT 图像
        orig_data = np.load(data_names[case_name])
        ct_vol = orig_data["imgs"]
        ct_slice = ct_vol[slice_idx]
        if ct_slice.ndim == 3:
            ct_slice = ct_slice[:, :, 0]  # 取第一通道
        # 归一化到 [0, 1]
        ct_min, ct_max = ct_slice.min(), ct_slice.max()
        if ct_max > ct_min:
            ct_slice = (ct_slice - ct_min) / (ct_max - ct_min)

        # 加载 GT
        gt_slice = orig_data["gts"][slice_idx]

        # 获取裁剪区域（聚焦目标器官）
        crop = get_organ_bbox(gt_slice, organ_id, pad=50)

        # 加载各实验的预测
        exp_keys = ["A0", "A3R3", "C3"]
        pred_slices = {}
        for exp in exp_keys:
            p_data = np.load(pred_names[exp][case_name])
            pred_slices[exp] = p_data["segs"][slice_idx]

        # 画 4 列
        # 列0: GT（绿色轮廓=GT，红色轮廓也=GT，即完全重合）
        overlay_contours(axes[row, 0], ct_slice, gt_slice, gt_slice,
                        organ_id, col_titles[0] if row == 0 else "", crop)

        # 列1-3: Baseline, BL, BL+MSL
        for col, (exp, title) in enumerate(
            zip(exp_keys, col_titles[1:]), start=1
        ):
            overlay_contours(axes[row, col], ct_slice, gt_slice,
                           pred_slices[exp], organ_id,
                           title if row == 0 else "", crop)

        # 行标签
        organ_name = ORGAN_MAP.get(organ_id, f"Label{organ_id}")
        axes[row, 0].set_ylabel(row_labels[row], fontsize=9,
                                rotation=0, labelpad=70, va="center")

    # 图例
    legend_elements = [
        Line2D([0], [0], color="#00FF00", linewidth=2, label="GT contour"),
        Line2D([0], [0], color="#FF3333", linewidth=2, label="Prediction contour"),
        Line2D([0], [0], color="yellow", linewidth=2, linestyle="--",
               label="Key difference region"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
              fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0.1, 0.03, 1, 1])
    out_path = os.path.join(OUT_DIR, "failure_cases.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n[OK] 已保存: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()

