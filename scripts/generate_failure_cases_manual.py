#!/usr/bin/env python3
"""生成失败案例可视化图 (3×4: 3类失败 × GT/Baseline/BL/BL+MSL) —— 手动指定版本

与 generate_failure_cases.py 的区别：
    - 本脚本不自动搜索最差切片，而是由用户在下方 MANUAL_CASES 中手动指定
      (case_name, slice_idx, organ_id, row_label) 四元组
    - 适合在预先浏览过预测结果、已确定想展示哪些切片的情况下使用
    - 提供 --list 模式，可快速打印各病例下目标器官的 Dice 分布，辅助挑选

用法:
  cd ~/chengang/zxw/MedSAM

  # 方式 1：先列出各器官的低 Dice 切片，辅助挑选
  python scripts/generate_failure_cases_manual.py --list

  # 方式 2：编辑下方 MANUAL_CASES 后运行生成
  python scripts/generate_failure_cases_manual.py

前提: 已运行 bash scripts/save_all_predictions.sh 保存了 A0/A3R3/C3 的预测 NPZ。

输出: thesis-medsam/figures/failure_cases.pdf
"""

import os
import sys
import glob
import argparse
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
DATA_ROOT = "data/npy/CT_Abd"                    # 原始数据（含 imgs, gts）
PRED_DIRS = {
    "A0":   "work_dir/eval_predictions/A0",      # Baseline
    "A3R3": "work_dir/eval_predictions/A3R3",    # BL
    "C3":   "work_dir/eval_predictions/C3",      # BL+MSL
}
OUT_DIR = "thesis-medsam/figures"
OUT_NAME = "failure_cases.pdf"

# 器官 ID 映射
ORGAN_MAP = {
    1: "Liver", 2: "R.Kidney", 3: "Spleen", 4: "Pancreas",
    5: "Aorta", 6: "IVC", 7: "RAG", 8: "LAG",
    9: "Gallbladder", 10: "Esophagus", 11: "Stomach",
    12: "Duodenum", 13: "L.Kidney",
}

# ═══════════════════════════════════════════
# 【手动指定失败案例】
#
# 每条记录格式：
#   {
#       "case_name": "FLARE22_Tr_0001.npz",  # 与 DATA_ROOT 下的文件名一致
#       "slice_idx": 42,                      # 切片索引（0-based）
#       "organ_id":  4,                       # 器官 ID（参见 ORGAN_MAP）
#       "row_label": "(a) Weak boundary\n(Pancreas)",  # 图中左侧行标签
#   }
#
# 提示：
#   1. 先用 `python scripts/generate_failure_cases_manual.py --list` 浏览
#      各病例下目标器官的 Dice 分布，记下感兴趣的 (case, slice, organ)
#   2. 将选中的三条失败案例填入下方 MANUAL_CASES
#   3. row_label 支持 \n 换行；第一个括号内标签建议与论文正文一致：
#      (a) 边界弱区域 / (b) 小体积细长器官 / (c) 器官邻接粘连
# ═══════════════════════════════════════════
MANUAL_CASES = [
    {
        "case_name": "FLARE22_Tr_0001.npz",   # TODO: 替换为实际 case 文件名
        "slice_idx": 42,                        # TODO: 替换为实际 slice 索引
        "organ_id":  4,                         # Pancreas
        "row_label": "(a) Weak boundary\n(Pancreas)",
    },
    {
        "case_name": "FLARE22_Tr_0005.npz",   # TODO: 替换
        "slice_idx": 88,                        # TODO: 替换
        "organ_id":  10,                        # Esophagus
        "row_label": "(b) Small organ\n(Esophagus)",
    },
    {
        "case_name": "FLARE22_Tr_0010.npz",   # TODO: 替换
        "slice_idx": 55,                        # TODO: 替换
        "organ_id":  12,                        # Duodenum
        "row_label": "(c) Adjacent organs\n(Duodenum)",
    },
]

# --list 模式下筛选阈值：只显示 Dice 低于此值的切片
LIST_DICE_UPPER = 0.92
LIST_DICE_LOWER = 0.30
LIST_MIN_AREA = 50  # 跳过过小区域

# 裁剪区域 padding（像素）
CROP_PAD = 50


def compute_slice_dice(pred, gt, label_id):
    """计算单个切片上某器官的 Dice"""
    p = (pred == label_id)
    g = (gt == label_id)
    inter = np.logical_and(p, g).sum()
    union = p.sum() + g.sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return 2 * inter / union


def get_organ_bbox(mask, label_id, pad=CROP_PAD):
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

    # 标注差异区域
    diff = np.logical_xor(gt_binary > 0, pred_binary > 0)
    if diff.sum() > 10:
        diff_ys, diff_xs = np.where(diff)
        cy, cx = diff_ys.mean(), diff_xs.mean()
        radius = max(15, min(diff_ys.max() - diff_ys.min(),
                             diff_xs.max() - diff_xs.min()) / 2 + 5)
        circle = Circle((cx, cy), radius, fill=False,
                        edgecolor="yellow", linewidth=1.5, linestyle="--")
        ax.add_patch(circle)

    if title:
        ax.set_title(title, fontsize=9, pad=3)
    ax.axis("off")


def load_file_index():
    """加载并对齐 data/A0/A3R3/C3 的文件列表，返回 common_names 与路径映射"""
    data_files = sorted(glob.glob(os.path.join(DATA_ROOT, "*.npz")))
    if not data_files:
        print(f"[ERROR] 原始数据目录为空: {DATA_ROOT}")
        sys.exit(1)

    pred_files = {}
    for exp_name, pred_dir in PRED_DIRS.items():
        files = sorted(glob.glob(os.path.join(pred_dir, "*.npz")))
        if not files:
            print(f"[ERROR] 预测目录为空: {pred_dir}")
            print("请先运行: bash scripts/save_all_predictions.sh")
            sys.exit(1)
        pred_files[exp_name] = files
        print(f"[OK] {exp_name}: {len(files)} 个预测文件")

    data_names = {os.path.basename(f): f for f in data_files}
    pred_names = {
        exp: {os.path.basename(f): f for f in pred_files[exp]}
        for exp in PRED_DIRS
    }

    common_names = sorted(
        set(data_names.keys())
        & set(pred_names["A0"].keys())
        & set(pred_names["A3R3"].keys())
        & set(pred_names["C3"].keys())
    )
    print(f"共 {len(common_names)} 个可用病例\n")
    return common_names, data_names, pred_names


def list_candidates(common_names, data_names, pred_names):
    """列出各病例下 C3 模型预测 Dice 较低的切片，供手动挑选参考"""
    print("═" * 72)
    print(" 候选失败切片列表（按 C3 Dice 升序，仅显示 "
          f"{LIST_DICE_LOWER} ≤ Dice < {LIST_DICE_UPPER} 的切片）")
    print("═" * 72)

    records = []
    for case_name in common_names:
        pred_c3 = np.load(pred_names["C3"][case_name])
        segs = pred_c3["segs"]
        gts = pred_c3["gts"]

        for z in range(gts.shape[0]):
            gt_slice = gts[z]
            seg_slice = segs[z]
            for label_id, organ_name in ORGAN_MAP.items():
                gt_area = (gt_slice == label_id).sum()
                if gt_area < LIST_MIN_AREA:
                    continue
                dice = compute_slice_dice(seg_slice, gt_slice, label_id)
                if LIST_DICE_LOWER <= dice < LIST_DICE_UPPER:
                    records.append({
                        "case": case_name,
                        "slice": z,
                        "organ_id": label_id,
                        "organ_name": organ_name,
                        "dice": dice,
                        "area": int(gt_area),
                    })

    records.sort(key=lambda r: r["dice"])

    # 按器官分组打印（方便挑选每类失败对应的切片）
    print(f"\n共找到 {len(records)} 个候选切片\n")
    print(f"{'器官':<14}{'Case':<30}{'Slice':>7}{'ID':>5}{'Dice':>9}{'Area':>8}")
    print("-" * 72)
    for r in records[:200]:  # 最多打印 200 条
        print(f"{r['organ_name']:<14}{r['case']:<30}"
              f"{r['slice']:>7}{r['organ_id']:>5}"
              f"{r['dice']:>9.4f}{r['area']:>8}")

    if len(records) > 200:
        print(f"\n[...] 另外 {len(records) - 200} 条未显示")

    print("\n" + "═" * 72)
    print(" 使用方法：")
    print("   1. 从上表中挑选 3 条有代表性的切片（对应三类失败情形）")
    print("   2. 将 (case_name, slice_idx, organ_id) 填入脚本顶部的 MANUAL_CASES")
    print("   3. 去掉 --list 参数再次运行脚本生成 failure_cases.pdf")
    print("═" * 72)


def validate_manual_cases(common_names, data_names, pred_names):
    """校验 MANUAL_CASES 中的配置是否有效，并打印各切片的三模型 Dice"""
    print("═" * 72)
    print(" 校验 MANUAL_CASES 配置")
    print("═" * 72)

    validated = []
    for i, case_cfg in enumerate(MANUAL_CASES):
        case_name = case_cfg["case_name"]
        slice_idx = case_cfg["slice_idx"]
        organ_id = case_cfg["organ_id"]
        row_label = case_cfg["row_label"]

        print(f"\n[{i+1}/{len(MANUAL_CASES)}] {row_label.replace(chr(10), ' ')}")

        # 校验 case_name 是否存在
        if case_name not in common_names:
            print(f"  [ERROR] case '{case_name}' 不在对齐后的病例列表中")
            print(f"          请确认该文件同时存在于 DATA_ROOT 及三个 PRED_DIRS 中")
            sys.exit(1)

        # 加载数据
        orig = np.load(data_names[case_name])
        gts = orig["gts"]

        if slice_idx < 0 or slice_idx >= gts.shape[0]:
            print(f"  [ERROR] slice_idx={slice_idx} 超出范围 "
                  f"[0, {gts.shape[0]-1}]")
            sys.exit(1)

        gt_slice = gts[slice_idx]
        gt_area = (gt_slice == organ_id).sum()
        if gt_area == 0:
            print(f"  [ERROR] 该切片上器官 {organ_id} "
                  f"({ORGAN_MAP.get(organ_id, '?')}) 不存在")
            sys.exit(1)

        # 打印三模型的 Dice 作为参考
        dices = {}
        for exp in ["A0", "A3R3", "C3"]:
            pred = np.load(pred_names[exp][case_name])["segs"][slice_idx]
            dices[exp] = compute_slice_dice(pred, gt_slice, organ_id)

        organ_name = ORGAN_MAP.get(organ_id, f"Label{organ_id}")
        print(f"  case={case_name}, slice={slice_idx}, "
              f"organ={organ_name}({organ_id}), gt_area={gt_area}")
        print(f"  Dice:  A0(Baseline)={dices['A0']:.4f}  "
              f"A3R3(BL)={dices['A3R3']:.4f}  "
              f"C3(BL+MSL)={dices['C3']:.4f}")
        if not (dices["A0"] <= dices["A3R3"] <= dices["C3"]):
            print(f"  [提示] 三模型 Dice 未呈单调递增，该切片不能直接体现"
                  f" A0→A3R3→C3 的渐进改善（仍可使用，仅作提示）")

        validated.append(case_cfg)

    print("\n[OK] 所有 MANUAL_CASES 校验通过\n")
    return validated


def render_figure(validated_cases, common_names, data_names, pred_names):
    """绘制 3×4 失败案例对比图"""
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(3, 4, figsize=(11, 8.5))
    col_titles = ["Ground Truth", "Baseline (A0)", "BL (A3R3)", "BL+MSL (C3)"]

    for row, case_cfg in enumerate(validated_cases):
        case_name = case_cfg["case_name"]
        slice_idx = case_cfg["slice_idx"]
        organ_id = case_cfg["organ_id"]
        row_label = case_cfg["row_label"]

        # 加载原始 CT 图像
        orig = np.load(data_names[case_name])
        ct_vol = orig["imgs"]
        ct_slice = ct_vol[slice_idx]
        if ct_slice.ndim == 3:
            ct_slice = ct_slice[:, :, 0]  # 取第一通道
        ct_min, ct_max = ct_slice.min(), ct_slice.max()
        if ct_max > ct_min:
            ct_slice = (ct_slice - ct_min) / (ct_max - ct_min)

        gt_slice = orig["gts"][slice_idx]
        crop = get_organ_bbox(gt_slice, organ_id, pad=CROP_PAD)

        pred_slices = {}
        for exp in ["A0", "A3R3", "C3"]:
            p_data = np.load(pred_names[exp][case_name])
            pred_slices[exp] = p_data["segs"][slice_idx]

        # 列 0: GT
        overlay_contours(axes[row, 0], ct_slice, gt_slice, gt_slice,
                         organ_id, col_titles[0] if row == 0 else "", crop)

        # 列 1-3: Baseline/BL/BL+MSL
        for col, (exp, title) in enumerate(
            zip(["A0", "A3R3", "C3"], col_titles[1:]), start=1
        ):
            overlay_contours(axes[row, col], ct_slice, gt_slice,
                             pred_slices[exp], organ_id,
                             title if row == 0 else "", crop)

        axes[row, 0].set_ylabel(row_label, fontsize=9,
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
    out_path = os.path.join(OUT_DIR, OUT_NAME)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] 已保存: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="手动挑选失败案例并生成 3×4 可视化对比图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--list", action="store_true",
        help="仅列出候选失败切片（不生成图片），辅助手动挑选"
    )
    args = parser.parse_args()

    common_names, data_names, pred_names = load_file_index()

    if args.list:
        list_candidates(common_names, data_names, pred_names)
        return

    validated = validate_manual_cases(common_names, data_names, pred_names)
    render_figure(validated, common_names, data_names, pred_names)


if __name__ == "__main__":
    main()

