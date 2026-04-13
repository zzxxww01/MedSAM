#!/usr/bin/env python3
"""Fig.7 边界区域放大对比（展示 C3 高频特征增强效果）

需要服务器上的评估预测 NPZ 文件。

输出: thesis-medsam/figures/boundary_detail.pdf
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = ["Times New Roman", "SimSun", "serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "thesis-medsam", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

SEARCH_PATTERNS = [
    "work_dir/eval_predictions/{name}/*.npz",
    "work_dir/eval_metrics/{name}_predictions/*.npz",
    "work_dir/{name}_predictions/*.npz",
    "work_dir/predictions/{name}/*.npz",
]

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
    for pattern in SEARCH_PATTERNS:
        matches = glob.glob(pattern.format(name=name))
        if matches:
            return sorted(matches)
    return []


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())
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
    mask = to_2d(mask)
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id, color in enumerate(ORGAN_COLORS):
        rgb[mask == label_id] = color
    return rgb


def find_boundary_roi(mask, roi_size=128):
    """Find a region with interesting organ boundaries."""
    # Find boundary pixels (where adjacent pixels have different labels)
    mask_2d = to_2d(mask)
    h, w = mask_2d.shape

    # Simple boundary detection via gradient
    dy = np.abs(np.diff(mask_2d, axis=0))
    dx = np.abs(np.diff(mask_2d, axis=1))

    boundary = np.zeros(mask_2d.shape, dtype=np.int64)
    boundary[:-1, :] += (dy > 0).astype(np.int64)
    boundary[:, :-1] += (dx > 0).astype(np.int64)

    # Find densest boundary region
    best_score = 0
    best_y, best_x = h // 4, w // 4
    step = roi_size // 4

    for y in range(0, h - roi_size, step):
        for x in range(0, w - roi_size, step):
            score = boundary[y:y + roi_size, x:x + roi_size].sum()
            if score > best_score:
                best_score = score
                best_y, best_x = y, x

    return best_y, best_x, roi_size


def main():
    parser = argparse.ArgumentParser(description="生成边界放大对比图")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--roi_size", type=int, default=128,
                        help="放大区域大小（像素）")
    args = parser.parse_args()

    experiments = ["A0", "A3R3", "C3"]
    titles = ["Baseline", "Balance Loss", "MSL-Adapter"]

    pred_files = {}
    for name in experiments:
        files = find_predictions(name)
        if files:
            pred_files[name] = files
        else:
            print(f"[WARN] {name}: 未找到预测文件")

    if len(pred_files) < len(experiments):
        missing = [n for n in experiments if n not in pred_files]
        print(f"\n缺少以下实验的预测: {', '.join(missing)}")
        print("请先运行: bash scripts/save_all_predictions.sh")
        sys.exit(1)

    idx = min(args.sample_idx,
              min(len(v) for v in pred_files.values()) - 1)

    masks = {}
    for name in experiments:
        _, mask = load_npz(pred_files[name][idx])
        if mask is not None:
            masks[name] = to_2d(mask)

    # Find boundary ROI from A3R3 prediction
    ref_mask = masks.get("A3R3", list(masks.values())[0])
    ry, rx, rs = find_boundary_roi(ref_mask, args.roi_size)

    # Plot: top row = full mask with ROI box, bottom row = zoomed ROI
    n = len(experiments)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    for col, name in enumerate(experiments):
        mask_rgb = colorize_mask(masks[name])

        # Full view with ROI rectangle
        axes[0, col].imshow(mask_rgb)
        rect = Rectangle((rx, ry), rs, rs, linewidth=2,
                          edgecolor="yellow", facecolor="none", linestyle="--")
        axes[0, col].add_patch(rect)
        axes[0, col].set_title(titles[col], fontsize=11, fontweight="bold")
        axes[0, col].axis("off")

        # Zoomed view
        roi = mask_rgb[ry:ry + rs, rx:rx + rs]
        axes[1, col].imshow(roi, interpolation="nearest")
        axes[1, col].set_title(f"{titles[col]} (放大)", fontsize=10)
        axes[1, col].axis("off")

    fig.suptitle("边界区域放大对比",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "boundary_detail.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()

