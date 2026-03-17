#!/usr/bin/env python3
"""统计 FLARE22 各器官前景像素占比，并绘制分布图。

用途：支撑论文中“类别不平衡”动机，输出两个图：
  1. organ_pixel_ratio_boxplot.pdf   各器官前景像素占比箱线图
  2. organ_pixel_ratio_bar.pdf       各器官前景像素占比均值柱状图（带误差棒）

默认从 data_root 下读取 *.npz 文件。每个 NPZ 需包含 gts（或 mask）键。
像素占比定义为：某器官前景像素数 / 全体切片总像素数。

Usage:
  python scripts/generate_fig9_organ_pixel_ratio.py \
    --data_root work_dir/npz/CT_Abd \
    --split_file work_dir/test_split.txt

如果不提供 split_file，则统计 data_root 下全部 NPZ 文件。
"""

import argparse
import glob
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "thesis-medsam", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

ORGAN_LABELS = {
    1: "Liver",
    2: "R.Kidney",
    3: "Spleen",
    4: "Pancreas",
    5: "Aorta",
    6: "IVC",
    7: "RAG",
    8: "LAG",
    9: "Gallbladder",
    10: "Esophagus",
    11: "Stomach",
    12: "Duodenum",
    13: "L.Kidney",
}

BAR_COLORS = [
    "#C0392B", "#D35400", "#F39C12", "#27AE60", "#16A085", "#2980B9",
    "#8E44AD", "#9B59B6", "#2C3E50", "#7F8C8D", "#E67E22", "#1ABC9C", "#34495E"
]


def parse_args():
    parser = argparse.ArgumentParser(description="统计各器官前景像素占比分布")
    parser.add_argument("--data_root", type=str, required=True,
                        help="包含病例 npz 的目录")
    parser.add_argument("--split_file", type=str, default=None,
                        help="可选：指定病例列表文件，每行一个 npz 文件名")
    return parser.parse_args()


def load_case_paths(data_root: str, split_file: str = None) -> List[str]:
    if split_file and os.path.exists(split_file):
        with open(split_file, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        paths = []
        for name in names:
            path = name if os.path.isabs(name) else os.path.join(data_root, name)
            if not path.endswith(".npz"):
                path += ".npz"
            if os.path.exists(path):
                paths.append(path)
        return sorted(paths)
    return sorted(glob.glob(os.path.join(data_root, "*.npz")))


def load_gts(npz_path: str) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    if "gts" in data:
        return data["gts"]
    if "mask" in data:
        return data["mask"]
    raise KeyError(f"No 'gts' or 'mask' key found in {npz_path}")


def compute_ratios(case_paths: List[str]) -> Dict[int, List[float]]:
    ratios = {label_id: [] for label_id in ORGAN_LABELS}
    for path in case_paths:
        gts = np.squeeze(load_gts(path))
        total_pixels = float(gts.size)
        for label_id in ORGAN_LABELS:
            organ_pixels = float(np.sum(gts == label_id))
            ratios[label_id].append(organ_pixels / total_pixels)
    return ratios


def format_percent(x: float) -> str:
    return f"{x * 100:.2f}%"


def plot_boxplot(ratios: Dict[int, List[float]]):
    labels = [ORGAN_LABELS[i] for i in ORGAN_LABELS]
    data = [np.array(ratios[i], dtype=float) * 100.0 for i in ORGAN_LABELS]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], BAR_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)

    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Foreground Pixel Ratio (%)", fontsize=12)
    ax.set_title("Per-Organ Foreground Pixel Ratio Distribution", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "organ_pixel_ratio_boxplot.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")
    plt.close(fig)


def plot_bar(ratios: Dict[int, List[float]]):
    labels = [ORGAN_LABELS[i] for i in ORGAN_LABELS]
    means = [np.mean(ratios[i]) * 100.0 for i in ORGAN_LABELS]
    stds = [np.std(ratios[i]) * 100.0 for i in ORGAN_LABELS]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color=BAR_COLORS, alpha=0.8, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Foreground Pixel Ratio (%)", fontsize=12)
    ax.set_title("Mean Foreground Pixel Ratio per Organ", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for xi, m in zip(x, means):
        ax.text(xi, m + max(stds) * 0.08 + 0.01, f"{m:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "organ_pixel_ratio_bar.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")
    plt.close(fig)


def print_summary(ratios: Dict[int, List[float]]):
    print("\n[Summary] Foreground pixel ratios")
    for label_id, name in ORGAN_LABELS.items():
        arr = np.array(ratios[label_id], dtype=float)
        print(f"  {name:<12} mean={format_percent(arr.mean())}  std={format_percent(arr.std())}  min={format_percent(arr.min())}  max={format_percent(arr.max())}")


def main():
    args = parse_args()
    case_paths = load_case_paths(args.data_root, args.split_file)
    if not case_paths:
        raise FileNotFoundError(f"No NPZ files found in {args.data_root}")

    print(f"[data] cases: {len(case_paths)}")
    ratios = compute_ratios(case_paths)
    print_summary(ratios)
    plot_boxplot(ratios)
    plot_bar(ratios)


if __name__ == "__main__":
    main()
