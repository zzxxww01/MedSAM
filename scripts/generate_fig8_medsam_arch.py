#!/usr/bin/env python3
"""Fig.8 MedSAM Three-Stage Architecture (Optional)

Pure matplotlib architecture diagram, no external data dependency.
Output: thesis-medsam/figures/medsam_arch.pdf
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "thesis-medsam", "figures")
os.makedirs(OUT_DIR, exist_ok=True)


def draw_box(ax, xy, w, h, text, color, fontsize=9, text_color="white"):
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.12",
        facecolor=color, edgecolor="black", linewidth=1.2,
        zorder=2,
    )
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color, zorder=3)


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
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-0.5, 6)
    ax.axis("off")

    ax.text(7.0, 5.5, "MedSAM Three-Stage Architecture", ha="center",
            fontsize=15, fontweight="bold", color="#2C3E50")

    y_enc = 3.0
    y_prompt = 0.8

    # Input image
    draw_box(ax, (0.0, y_enc - 0.5), 2.0, 1.0,
             "Medical CT Slice\n1024x1024x3", "#95A5A6", 9)

    # Image Encoder
    draw_arrow(ax, (2.0, y_enc), (2.5, y_enc))
    draw_box(ax, (2.5, y_enc - 0.7), 3.5, 1.4,
             "Image Encoder\n----------\nViT-Base (12 Blocks)\nPatch 16x16\nMulti-Head Self-Attention\n-> 64x64x256",
             "#2C3E50", 8.5)

    # Feature map
    draw_arrow(ax, (6.0, y_enc), (6.5, y_enc))
    draw_box(ax, (6.5, y_enc - 0.35), 1.8, 0.7,
             "Image\nEmbedding\n64x64x256", "#8E44AD", 8)

    # Mask Decoder
    draw_arrow(ax, (8.3, y_enc), (8.8, y_enc))
    draw_box(ax, (8.8, y_enc - 0.7), 3.0, 1.4,
             "Mask Decoder\n----------\nBi-directional Cross-Attn\n(prompt<->image)\nMLP + Upsampling\n-> 1024x1024",
             "#3498DB", 8.5)

    # Output mask
    draw_arrow(ax, (11.8, y_enc), (12.3, y_enc))
    draw_box(ax, (12.3, y_enc - 0.5), 2.0, 1.0,
             "Segmentation\nMask\n1024x1024", "#27AE60", 9)

    # Prompt Encoder (bottom)
    draw_box(ax, (0.0, y_prompt - 0.35), 2.0, 0.7,
             "Box / Point\nPrompt", "#F39C12", 9, text_color="black")

    draw_arrow(ax, (2.0, y_prompt), (2.5, y_prompt), "#F39C12")
    draw_box(ax, (2.5, y_prompt - 0.5), 3.5, 1.0,
             "Prompt Encoder\n----------\nPosition Embedding\n+ Type Encoding\n(Frozen)",
             "#F39C12", 8.5, text_color="black")

    draw_arrow(ax, (6.0, y_prompt), (6.5, y_prompt), "#F39C12")
    draw_box(ax, (6.5, y_prompt - 0.35), 1.8, 0.7,
             "Prompt\nEmbedding", "#E67E22", 8)

    # Prompt Embedding -> Decoder
    draw_arrow(ax, (8.3, y_prompt), (10.3, y_enc - 0.7), "#F39C12")

    # Training annotations
    ax.text(4.25, y_enc + 1.1, "Full Fine-Tuning", ha="center",
            fontsize=9, color="#E74C3C", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#FADBD8",
                      edgecolor="#E74C3C"))
    ax.text(10.3, y_enc + 1.1, "Full Fine-Tuning", ha="center",
            fontsize=9, color="#E74C3C", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#FADBD8",
                      edgecolor="#E74C3C"))
    ax.text(4.25, y_prompt - 0.95, "Frozen", ha="center",
            fontsize=9, color="#7F8C8D", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#EAECEE",
                      edgecolor="#7F8C8D"))

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "medsam_arch.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
