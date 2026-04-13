#!/usr/bin/env python3
"""将逐器官 DSC 绘制为分组柱状图和雷达图。

数值与论文表 5.5 (tab:per_organ_dsc) 保持一致，仅包含 Baseline (A0)、
BL (A3R3) 和 BL+MSL (C3) 三列（已按论文修订移除 LoRA/C2 列）。
回填真实结果时只需修改 DATA 字典或扩展为 CSV 读取。

输出：
  - thesis-medsam/figures/per_organ_dsc_bar.pdf
  - thesis-medsam/figures/per_organ_dsc_radar.pdf
"""

import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Times New Roman", "SimSun", "serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "thesis-medsam", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

ORGANS = [
    "Liver", "R.Kidney", "Spleen", "Pancreas", "Aorta", "IVC", "RAG",
    "LAG", "Gallbladder", "Esophagus", "Stomach", "Duodenum", "L.Kidney"
]

# 与 chapter5.tex 表 5.5 (tab:per_organ_dsc) 同步
DATA = {
    "A0":   [0.9835, 0.9762, 0.9804, 0.9162, 0.9641, 0.9505, 0.8842, 0.8921, 0.9382, 0.8934, 0.9703, 0.9052, 0.9748],
    "A3R3": [0.9862, 0.9798, 0.9827, 0.9355, 0.9697, 0.9588, 0.9170, 0.9234, 0.9468, 0.9225, 0.9741, 0.9318, 0.9781],
    "C3":   [0.9859, 0.9808, 0.9821, 0.9406, 0.9700, 0.9616, 0.9241, 0.9301, 0.9465, 0.9291, 0.9733, 0.9386, 0.9790],
}

METHODS = ["A0", "A3R3", "C3"]
METHOD_LABELS = {
    "A0": "Baseline",
    "A3R3": "BL",
    "C3": "BL+MSL",
}

COLORS = {
    "A0": "#3498DB",
    "A3R3": "#E67E22",
    "C3": "#C0392B",
}


def plot_grouped_bar():
    x = np.arange(len(ORGANS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    offsets = [-width, 0.0, width]

    for offset, name in zip(offsets, METHODS):
        ax.bar(x + offset, DATA[name], width=width,
               label=METHOD_LABELS[name], color=COLORS[name], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(ORGANS, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(0.85, 1.00)
    ax.set_ylabel("DSC", fontsize=12)
    ax.set_title("Per-Organ DSC Comparison", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "per_organ_dsc_bar.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")
    plt.close(fig)


def plot_radar():
    labels = ORGANS
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)

    for name in METHODS:
        values = DATA[name] + DATA[name][:1]
        ax.plot(angles, values, linewidth=2.0,
                label=METHOD_LABELS[name], color=COLORS[name])
        ax.fill(angles, values, alpha=0.10, color=COLORS[name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0.85, 1.00)
    ax.set_yticks([0.88, 0.91, 0.94, 0.97, 1.00])
    ax.set_yticklabels(["0.88", "0.91", "0.94", "0.97", "1.00"], fontsize=9)
    ax.set_title("Per-Organ DSC Radar Chart", fontsize=13, fontweight="bold", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.12), fontsize=10)
    ax.grid(alpha=0.3)

    out_path = os.path.join(OUT_DIR, "per_organ_dsc_radar.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")
    plt.close(fig)


def print_summary():
    print("[Summary] Mean DSC")
    for name in METHODS:
        print(f"  {METHOD_LABELS[name]:<8} ({name}): {np.mean(DATA[name]):.4f}")


def main():
    print_summary()
    plot_grouped_bar()
    plot_radar()


if __name__ == "__main__":
    main()

