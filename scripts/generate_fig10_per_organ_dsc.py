#!/usr/bin/env python3
"""将逐器官 DSC 绘制为分组柱状图和雷达图。

默认使用论文当前表5.5中的数值；后续回填真实结果时，只需修改 DATA 字典或扩展为 CSV 读取。

输出：
  - thesis-medsam/figures/per_organ_dsc_bar.pdf
  - thesis-medsam/figures/per_organ_dsc_radar.pdf
"""

import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "thesis-medsam", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

ORGANS = [
    "Liver", "R.Kidney", "Spleen", "Pancreas", "Aorta", "IVC", "RAG",
    "LAG", "Gallbladder", "Esophagus", "Stomach", "Duodenum", "L.Kidney"
]

DATA = {
    "A0":    [0.9821, 0.9708, 0.9763, 0.8926, 0.9437, 0.9152, 0.8341, 0.8508, 0.9057, 0.8462, 0.9689, 0.8418, 0.9683],
    "A3R3":  [0.9856, 0.9789, 0.9813, 0.9375, 0.9648, 0.9571, 0.8893, 0.9026, 0.9357, 0.9018, 0.9752, 0.9013, 0.9764],
    "C2":    [0.9617, 0.9385, 0.9504, 0.8120, 0.8742, 0.8203, 0.7418, 0.7635, 0.8374, 0.7893, 0.9315, 0.7806, 0.9338],
    "C3":    [0.9849, 0.9801, 0.9807, 0.9462, 0.9685, 0.9593, 0.8978, 0.9147, 0.9402, 0.9253, 0.9741, 0.9224, 0.9788],
}

COLORS = {
    "A0": "#3498DB",
    "A3R3": "#E67E22",
    "C2": "#95A5A6",
    "C3": "#C0392B",
}


def plot_grouped_bar():
    x = np.arange(len(ORGANS))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    for offset, name in zip(offsets, ["A0", "A3R3", "C2", "C3"]):
        ax.bar(x + offset, DATA[name], width=width, label=name, color=COLORS[name], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(ORGANS, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(0.70, 1.00)
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

    for name in ["A0", "A3R3", "C2", "C3"]:
        values = DATA[name] + DATA[name][:1]
        ax.plot(angles, values, linewidth=2.0, label=name, color=COLORS[name])
        ax.fill(angles, values, alpha=0.08, color=COLORS[name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0.75, 1.00)
    ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_yticklabels(["0.80", "0.85", "0.90", "0.95", "1.00"], fontsize=9)
    ax.set_title("Per-Organ DSC Radar Chart", fontsize=13, fontweight="bold", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.12), fontsize=10)
    ax.grid(alpha=0.3)

    out_path = os.path.join(OUT_DIR, "per_organ_dsc_radar.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")
    plt.close(fig)


def print_summary():
    print("[Summary] Mean DSC")
    for name, vals in DATA.items():
        print(f"  {name:<5}: {np.mean(vals):.4f}")


def main():
    print_summary()
    plot_grouped_bar()
    plot_radar()


if __name__ == "__main__":
    main()
