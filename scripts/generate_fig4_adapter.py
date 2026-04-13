#!/usr/bin/env python3
"""Fig.4 MSL-Adapter Module Architecture

Pure matplotlib architecture diagram, no external data dependency.
Output: thesis-medsam/figures/lg_adapter_arch.pdf
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = ["Times New Roman", "SimSun", "serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "thesis-medsam", "figures")
os.makedirs(OUT_DIR, exist_ok=True)


def draw_box(ax, xy, w, h, text, color, fontsize=9, text_color="white"):
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.12",
        facecolor=color, edgecolor="black", linewidth=1.0,
        zorder=2,
    )
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color, zorder=3)
    return (cx, cy)


def draw_arrow(ax, start, end, color="black"):
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=12,
        color=color, linewidth=1.3,
        zorder=1,
    )
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 9.5)
    ax.axis("off")

    ax.text(4.75, 9.0, "MSL-Adapter 模块架构", ha="center",
            fontsize=14, fontweight="bold", color="#2C3E50")

    # Input feature
    draw_box(ax, (2.5, 7.8), 4.5, 0.6,
             "输入: x ∈ R^{Bx256x64x64}", "#34495E", 10)

    # Fork
    draw_arrow(ax, (4.75, 7.8), (2.2, 7.0), "#555")
    draw_arrow(ax, (4.75, 7.8), (7.3, 7.0), "#555")

    # Residual skip connection
    ax.annotate("", xy=(9.0, 2.8), xytext=(9.0, 8.1),
                arrowprops=dict(arrowstyle="-|>", color="#95A5A6",
                                linewidth=1.5, linestyle="--"))
    ax.text(9.3, 5.5, "残差\n跳连", fontsize=8, color="#95A5A6",
            ha="center", rotation=90)

    # Path 1: Standard DWConv 3x3
    draw_box(ax, (0.5, 6.2), 3.4, 0.6,
             "深度可分离卷积 3x3 (d=1)\ngroups=256", "#E74C3C", 9)
    draw_box(ax, (0.5, 5.2), 3.4, 0.5,
             "GELU", "#C0392B", 9)
    draw_arrow(ax, (2.2, 6.2), (2.2, 5.7), "#E74C3C")

    ax.text(2.2, 5.85, "f1", fontsize=8, ha="center", color="#E74C3C",
            bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="#E74C3C"))

    # Path 2: Dilated DWConv 3x3
    draw_box(ax, (5.6, 6.2), 3.4, 0.6,
             "空洞卷积 3x3 (d=2)\ngroups=256, dilation=2", "#3498DB", 9)
    draw_box(ax, (5.6, 5.2), 3.4, 0.5,
             "GELU", "#2980B9", 9)
    draw_arrow(ax, (7.3, 6.2), (7.3, 5.7), "#3498DB")

    ax.text(7.3, 5.85, "f2", fontsize=8, ha="center", color="#3498DB",
            bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="#3498DB"))

    # Receptive field annotation
    ax.text(2.2, 4.85, "有效感受野: 3x3", fontsize=7, ha="center",
            color="#E74C3C", fontstyle="italic")
    ax.text(7.3, 4.85, "有效感受野: 5x5", fontsize=7, ha="center",
            color="#3498DB", fontstyle="italic")

    # Concatenate
    draw_arrow(ax, (2.2, 5.2), (4.0, 4.2), "#555")
    draw_arrow(ax, (7.3, 5.2), (5.5, 4.2), "#555")

    draw_box(ax, (2.75, 3.6), 4.0, 0.5,
             "拼接: [f1 || f2] -> Bx512x64x64", "#8E44AD", 9)

    # Pointwise Conv 1x1
    draw_arrow(ax, (4.75, 3.6), (4.75, 3.0), "#555")
    draw_box(ax, (2.75, 2.4), 4.0, 0.5,
             "1x1卷积 (512->256) 跨尺度融合", "#2C3E50", 9)

    # Residual addition
    draw_arrow(ax, (4.75, 2.4), (4.75, 1.6), "#555")
    draw_arrow(ax, (9.0, 2.8), (5.65, 1.35), "#95A5A6")

    # Plus circle
    circle = plt.Circle((4.75, 1.3), 0.25, facecolor="#F39C12",
                         edgecolor="black", linewidth=1.0, zorder=2)
    ax.add_patch(circle)
    ax.text(4.75, 1.3, "+", ha="center", va="center",
            fontsize=14, fontweight="bold", color="white", zorder=3)

    # Output
    draw_arrow(ax, (4.75, 1.05), (4.75, 0.6), "#555")
    draw_box(ax, (2.5, 0.0), 4.5, 0.5,
             "输出: y = x + f_out ∈ R^{Bx256x64x64}", "#27AE60", 10)

    # Param count
    ax.text(4.75, -0.7, "总参数量: ~136K (占模型 0.15%)", ha="center",
            fontsize=9, fontstyle="italic", color="#7F8C8D")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "lg_adapter_arch.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()

