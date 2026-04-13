#!/usr/bin/env python3
"""Fig.6 典型病例的预测掩码对比（GT / A0 / A3R3 / C2 / C3）

需要服务器上的评估预测 NPZ 文件。
脚本会自动在 work_dir/ 下搜索可能的预测文件。

输出: thesis-medsam/figures/mask_comparison.pdf
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams["font.family"] = ["Times New Roman", "SimSun", "serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "thesis-medsam", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# 搜索预测 NPZ 的候选路径模式
SEARCH_PATTERNS = [
    "work_dir/eval_predictions/{name}/*.npz",
    "work_dir/eval_metrics/{name}_predictions/*.npz",
    "work_dir/{name}_predictions/*.npz",
    "work_dir/predictions/{name}/*.npz",
]

EXPERIMENTS = ["A0", "A3R3", "C2", "C3"]
TITLES = ["真值标注", "Baseline", "Balance Loss", "LoRA", "MSL-Adapter"]

# 为 13 个器官类别定义颜色映射
ORGAN_COLORS = [
    [0, 0, 0],        # 0: background
    [255, 0, 0],      # 1: liver
    [0, 255, 0],      # 2: right kidney
    [0, 0, 255],      # 3: spleen
    [255, 255, 0],    # 4: pancreas
    [255, 0, 255],    # 5: aorta
    [0, 255, 255],    # 6: IVC
    [128, 0, 0],      # 7: RAG
    [0, 128, 0],      # 8: LAG
    [0, 0, 128],      # 9: gallbladder
    [128, 128, 0],    # 10: esophagus
    [128, 0, 128],    # 11: stomach
    [0, 128, 128],    # 12: duodenum
    [64, 64, 255],    # 13: left kidney
]


def find_predictions(name):
    """Search for prediction NPZ files for a given experiment name."""
    for pattern in SEARCH_PATTERNS:
        matches = glob.glob(pattern.format(name=name))
        if matches:
            return sorted(matches)
    return []


def load_npz(path):
    """Load prediction from NPZ file. Returns (image, mask) or None."""
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())
    # Try common key names
    img_key = next((k for k in keys if k in ("img", "image", "imgs")), None)
    mask_key = next((k for k in keys if k in ("mask", "pred", "segs", "gts")), None)
    if mask_key is None and len(keys) >= 1:
        mask_key = keys[0]
    img = data[img_key] if img_key else None
    mask = data[mask_key] if mask_key else None
    return img, mask


def to_2d(mask):
    """Ensure mask is 2D (H, W). If 3D, take the middle slice."""
    mask = np.squeeze(mask)
    if mask.ndim == 3:
        mask = mask[mask.shape[0] // 2]
    return mask


def colorize_mask(mask):
    """Convert label mask to RGB image."""
    mask = to_2d(mask)
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id, color in enumerate(ORGAN_COLORS):
        rgb[mask == label_id] = color
    return rgb


def main():
    parser = argparse.ArgumentParser(description="生成预测掩码对比图")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="选择样本索引 (默认第 0 个)")
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="Ground truth NPZ 目录（如不指定则自动搜索）")
    args = parser.parse_args()

    # 查找各实验的预测文件
    pred_files = {}
    missing = []
    for name in EXPERIMENTS:
        files = find_predictions(name)
        if files:
            pred_files[name] = files
            print(f"[OK] {name}: 找到 {len(files)} 个预测文件")
        else:
            missing.append(name)
            print(f"[WARN] {name}: 未找到预测文件")

    if missing:
        print(f"\n缺少以下实验的预测 NPZ: {', '.join(missing)}")
        print("请先运行: bash scripts/save_all_predictions.sh")
        print("或手动运行 eval_medsam_npz.py 并保存预测。")
        sys.exit(1)

    # 确定样本索引
    idx = args.sample_idx
    min_count = min(len(v) for v in pred_files.values())
    if idx >= min_count:
        print(f"[WARN] 索引 {idx} 超出范围，使用 0")
        idx = 0

    # 加载数据
    masks = {}
    img = None
    for name in EXPERIMENTS:
        fpath = pred_files[name][idx]
        loaded_img, loaded_mask = load_npz(fpath)
        if loaded_img is not None and img is None:
            img = loaded_img
        if loaded_mask is not None:
            masks[name] = loaded_mask

    # 尝试加载 GT
    gt_mask = None
    if args.gt_dir:
        gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.npz")))
        if gt_files and idx < len(gt_files):
            _, gt_mask = load_npz(gt_files[idx])

    # 绘图
    n_cols = 1 + len(masks)  # GT + experiments
    if gt_mask is None:
        n_cols = len(masks)

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    col = 0
    if gt_mask is not None:
        axes[col].imshow(colorize_mask(gt_mask))
        axes[col].set_title("真值标注", fontsize=11, fontweight="bold")
        axes[col].axis("off")
        col += 1

    for name in EXPERIMENTS:
        if name in masks:
            axes[col].imshow(colorize_mask(masks[name]))
            label = {
                "A0": "Baseline\nDSC=0.9407",
                "A3R3": "Balance Loss\nDSC=0.9543",
                "C2": "LoRA\nDSC=0.9312",
                "C3": "MSL-Adapter\nDSC=0.9571",
            }.get(name, name)
            axes[col].set_title(label, fontsize=10, fontweight="bold")
            axes[col].axis("off")
            col += 1

    fig.suptitle("预测掩码对比", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "mask_comparison.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()

