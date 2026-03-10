#!/usr/bin/env python3
"""Fig.1 Technical Roadmap (Three-Pillar Approach)

Pure matplotlib flowchart, no external data dependency.
Output: thesis-medsam/figures/tech_roadmap.pdf
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "thesis-medsam", "figures")
os.makedirs(OUT_DIR, exist_ok=True)


def draw_box(ax, xy, w, h, text, color, fontsize=10, text_color="white"):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor="black", linewidth=1.2,
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
        mutation_scale=14,
        color=color, linewidth=1.5,
        zorder=1,
    )
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-1, 7)
    ax.axis("off")

    # Top: MedSAM base
    c_base = draw_box(ax, (3.5, 5.8), 4.5, 0.8,
                       "MedSAM Base Model\n(ViT-Base + Mask Decoder)", "#2C3E50", 11)

    # Three-pillar title
    ax.text(5.75, 5.05, "Three-Pillar Technical Roadmap", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#2C3E50",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECF0F1",
                      edgecolor="#2C3E50", linewidth=1.5))

    # Pillar 1: Loss Layer
    c1 = draw_box(ax, (0.2, 3.2), 3.2, 1.2,
                  "Pillar 1: Loss Layer\n----------\nBalance Loss\n(Inter-CBL + Intra-CBL\n+ Dice + Two-Stage)",
                  "#E74C3C", 9)

    # Pillar 2: Ablation
    c2 = draw_box(ax, (4.15, 3.2), 3.2, 1.2,
                  "Pillar 2: Ablation\n----------\nLoRA (r=4) Frozen Backbone\n-> DSC 0.879\n-> Full FT is necessary",
                  "#3498DB", 9)

    # Pillar 3: Feature Layer
    c3 = draw_box(ax, (8.1, 3.2), 3.2, 1.2,
                  "Pillar 3: Feature Layer\n----------\nLG-Adapter\n(Dual-Path DWConv\n+ Dilated Conv + Residual)",
                  "#27AE60", 9)

    # Arrows: base -> three pillars
    draw_arrow(ax, (5.75, 5.8), (1.8, 4.4), "#E74C3C")
    draw_arrow(ax, (5.75, 5.8), (5.75, 4.4), "#3498DB")
    draw_arrow(ax, (5.75, 5.8), (9.7, 4.4), "#27AE60")

    # Bottom: ablation chain
    steps = [
        ("A0\nBaseline\nDSC 0.941", "#95A5A6", 0.3),
        ("A3R3\nBalance Loss\nDSC 0.960", "#E74C3C", 3.0),
        ("C2\nLoRA\nDSC 0.879", "#3498DB", 5.7),
        ("C3\nLG-Adapter\nDSC 0.962", "#27AE60", 8.4),
    ]
    for text, color, x in steps:
        draw_box(ax, (x, 0.8), 2.3, 1.0, text, color, 9)

    # Arrows: ablation chain
    draw_arrow(ax, (2.6, 1.3), (3.0, 1.3), "#555")
    draw_arrow(ax, (5.3, 1.3), (5.7, 1.3), "#555")
    draw_arrow(ax, (8.0, 1.3), (8.4, 1.3), "#555")

    ax.text(5.75, 0.3, "Ablation Path: A0 -> A3R3 -> C2 (counter-evidence) -> C3 (optimal)",
            ha="center", fontsize=10, fontstyle="italic", color="#7F8C8D")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "tech_roadmap.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
