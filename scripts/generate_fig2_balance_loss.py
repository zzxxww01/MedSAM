#!/usr/bin/env python3
"""Fig.2 Balance Loss 组件与两阶段训练策略流程图

生成纯 matplotlib 流程图，无外部数据依赖。
输出: thesis-medsam/figures/balance_loss_flow.pdf
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["SimSun", "STSong", "Noto Serif CJK SC", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "thesis-medsam", "figures")
os.makedirs(OUT_DIR, exist_ok=True)


def draw_box(ax, xy, w, h, text, color, fontsize=9, text_color="white", alpha=1.0):
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.12",
        facecolor=color, edgecolor="black", linewidth=1.0,
        alpha=alpha, zorder=2,
    )
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color, zorder=3)
    return (cx, cy)


def draw_arrow(ax, start, end, color="black", style="-|>"):
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        mutation_scale=12,
        color=color, linewidth=1.3,
        zorder=1,
    )
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(-0.5, 13)
    ax.set_ylim(-0.5, 8.5)
    ax.axis("off")

    # ── 标题 ──
    ax.text(6.25, 8.0, "Balance Loss 组件与两阶段训练策略", ha="center",
            fontsize=14, fontweight="bold", color="#2C3E50")

    # ── 输入 ──
    draw_box(ax, (4.5, 6.8), 3.5, 0.7, "模型预测 P  vs  真实标签 Y", "#34495E", 10)

    # ── 分支：像素分类 ──
    draw_arrow(ax, (6.25, 6.8), (2.5, 6.0), "#555")
    draw_arrow(ax, (6.25, 6.8), (10.0, 6.0), "#555")

    ax.text(3.5, 6.35, "类别间维度", fontsize=9, ha="center", color="#E74C3C", fontweight="bold")
    ax.text(9.0, 6.35, "类别内维度", fontsize=9, ha="center", color="#3498DB", fontweight="bold")

    # ── 左支：Inter-CBL ──
    draw_box(ax, (0.3, 5.0), 4.4, 0.8,
             "Inter-CBL\n前景集合 F + 困难背景 H_B (neg_ratio=3.0)",
             "#E74C3C", 8.5)

    draw_box(ax, (0.3, 3.8), 4.4, 0.8,
             "L_inter = ½ (mean(ℓ_F) + mean(ℓ_HB))\n按前景数量对齐困难背景",
             "#C0392B", 8.5)

    draw_arrow(ax, (2.5, 5.0), (2.5, 4.6), "#E74C3C")

    # ── 右支：Intra-CBL ──
    draw_box(ax, (7.8, 5.0), 4.4, 0.8,
             "Intra-CBL\n阈值 τ=0.9 划分：易样本 E / 难样本 H",
             "#3498DB", 8.5)

    draw_box(ax, (7.8, 3.8), 4.4, 0.8,
             "L_intra = w_e·mean(ℓ_E) + w_h·mean(ℓ_H)\n(w_e=1.0, w_h=2.0)",
             "#2980B9", 8.5)

    draw_arrow(ax, (10.0, 5.0), (10.0, 4.6), "#3498DB")

    # ── Dice Loss ──
    draw_box(ax, (4.85, 3.8), 2.8, 0.8, "Dice Loss\nL_dice = 1 - Dice", "#27AE60", 9)

    # ── 汇合：Balance Loss ──
    draw_arrow(ax, (2.5, 3.8), (5.0, 2.6), "#E74C3C")
    draw_arrow(ax, (6.25, 3.8), (6.25, 2.6), "#27AE60")
    draw_arrow(ax, (10.0, 3.8), (7.5, 2.6), "#3498DB")

    draw_box(ax, (3.5, 1.8), 5.5, 0.7,
             "L_balance = α·L_inter + β·L_intra + γ·L_dice\n(α=0.5, β=1.0, γ=1.0)",
             "#8E44AD", 10)

    # ── 两阶段训练 ──
    draw_arrow(ax, (6.25, 1.8), (6.25, 1.3), "#8E44AD")

    # Stage 1
    draw_box(ax, (0.5, 0.1), 5.0, 0.8,
             "Stage 1 (Epoch 0~50)\nL_intra + L_dice  (不启用 Inter-CBL)",
             "#F39C12", 9, text_color="black")

    # Stage 2
    draw_box(ax, (7.0, 0.1), 5.5, 0.8,
             "Stage 2 (Epoch 50~200)\n完整 Balance Loss  (启用 Inter-CBL)",
             "#D35400", 9)

    draw_arrow(ax, (5.5, 0.5), (7.0, 0.5), "#555")
    ax.text(6.25, 0.5, "切换", ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="#555"))

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "balance_loss_flow.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
