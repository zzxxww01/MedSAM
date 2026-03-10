#!/usr/bin/env python3
"""Fig.2 Balance Loss Components & Two-Stage Training Strategy

Pure matplotlib flowchart, no external data dependency.
Output: thesis-medsam/figures/balance_loss_flow.pdf
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "serif"
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

    # Title
    ax.text(6.25, 8.0, "Balance Loss Components & Two-Stage Training", ha="center",
            fontsize=14, fontweight="bold", color="#2C3E50")

    # Input
    draw_box(ax, (4.5, 6.8), 3.5, 0.7, "Prediction P  vs  Ground Truth Y", "#34495E", 10)

    # Branches
    draw_arrow(ax, (6.25, 6.8), (2.5, 6.0), "#555")
    draw_arrow(ax, (6.25, 6.8), (10.0, 6.0), "#555")

    ax.text(3.5, 6.35, "Inter-Class", fontsize=9, ha="center", color="#E74C3C", fontweight="bold")
    ax.text(9.0, 6.35, "Intra-Class", fontsize=9, ha="center", color="#3498DB", fontweight="bold")

    # Left: Inter-CBL
    draw_box(ax, (0.3, 5.0), 4.4, 0.8,
             "Inter-CBL\nForeground F + Hard Background H_B (neg_ratio=3.0)",
             "#E74C3C", 8.5)

    draw_box(ax, (0.3, 3.8), 4.4, 0.8,
             "L_inter = 1/2 (mean(l_F) + mean(l_HB))\nAlign hard BG by foreground count",
             "#C0392B", 8.5)

    draw_arrow(ax, (2.5, 5.0), (2.5, 4.6), "#E74C3C")

    # Right: Intra-CBL
    draw_box(ax, (7.8, 5.0), 4.4, 0.8,
             "Intra-CBL\nThreshold t=0.9: Easy E / Hard H",
             "#3498DB", 8.5)

    draw_box(ax, (7.8, 3.8), 4.4, 0.8,
             "L_intra = w_e*mean(l_E) + w_h*mean(l_H)\n(w_e=1.0, w_h=2.0)",
             "#2980B9", 8.5)

    draw_arrow(ax, (10.0, 5.0), (10.0, 4.6), "#3498DB")

    # Dice Loss
    draw_box(ax, (4.85, 3.8), 2.8, 0.8, "Dice Loss\nL_dice = 1 - Dice", "#27AE60", 9)

    # Merge: Balance Loss
    draw_arrow(ax, (2.5, 3.8), (5.0, 2.6), "#E74C3C")
    draw_arrow(ax, (6.25, 3.8), (6.25, 2.6), "#27AE60")
    draw_arrow(ax, (10.0, 3.8), (7.5, 2.6), "#3498DB")

    draw_box(ax, (3.5, 1.8), 5.5, 0.7,
             "L_balance = a*L_inter + b*L_intra + g*L_dice\n(a=0.5, b=1.0, g=1.0)",
             "#8E44AD", 10)

    # Two-stage training
    draw_arrow(ax, (6.25, 1.8), (6.25, 1.3), "#8E44AD")

    # Stage 1
    draw_box(ax, (0.5, 0.1), 5.0, 0.8,
             "Stage 1 (Epoch 0~50)\nL_intra + L_dice  (Inter-CBL disabled)",
             "#F39C12", 9, text_color="black")

    # Stage 2
    draw_box(ax, (7.0, 0.1), 5.5, 0.8,
             "Stage 2 (Epoch 50~200)\nFull Balance Loss  (Inter-CBL enabled)",
             "#D35400", 9)

    draw_arrow(ax, (5.5, 0.5), (7.0, 0.5), "#555")
    ax.text(6.25, 0.5, "Switch", ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="#555"))

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "balance_loss_flow.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
