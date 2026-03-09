#!/usr/bin/env python3
"""Fig.5 LG-Adapter 在 MedSAM 推理管线中的插入位置图

生成纯 matplotlib 管线图，无外部数据依赖。
输出: thesis-medsam/figures/adapter_pipeline.pdf
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["SimSun", "STSong", "Noto Serif CJK SC", "DejaVu Serif"]
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
    return (cx, cy)


def draw_arrow(ax, start, end, color="black", lw=1.5):
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=14,
        color=color, linewidth=lw,
        zorder=1,
    )
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-0.5, 5)
    ax.axis("off")

    ax.text(7.0, 4.5, "MedSAM + LG-Adapter 推理管线", ha="center",
            fontsize=14, fontweight="bold", color="#2C3E50")

    y_main = 2.5
    y_prompt = 0.8
    bh = 0.9
    bw = 2.5

    # ── CT 输入 ──
    draw_box(ax, (0.0, y_main - bh / 2), 1.8, bh,
             "CT 切片\n1024×1024", "#95A5A6", 9)

    # ── Image Encoder ──
    draw_arrow(ax, (1.8, y_main), (2.3, y_main), "#555")
    draw_box(ax, (2.3, y_main - bh / 2), bw, bh,
             "Image Encoder\n(ViT-Base, 12层)\n全参微调", "#2C3E50", 9)

    # ── LG-Adapter (高亮) ──
    draw_arrow(ax, (4.8, y_main), (5.5, y_main), "#E74C3C")
    draw_box(ax, (5.5, y_main - bh / 2), bw, bh,
             "LG-Adapter\n双路径 DWConv\n+ 残差连接", "#E74C3C", 10)

    # 虚线框高亮
    highlight = plt.Rectangle((5.35, y_main - bh / 2 - 0.15), bw + 0.3, bh + 0.3,
                               fill=False, edgecolor="#E74C3C",
                               linewidth=2.5, linestyle="--", zorder=1)
    ax.add_patch(highlight)
    ax.text(6.75, y_main + bh / 2 + 0.25, "本文新增模块", ha="center",
            fontsize=9, color="#E74C3C", fontweight="bold")

    # ── Mask Decoder ──
    draw_arrow(ax, (8.0, y_main), (8.7, y_main), "#555")
    draw_box(ax, (8.7, y_main - bh / 2), bw, bh,
             "Mask Decoder\n交叉注意力\n+ 上采样", "#3498DB", 9)

    # ── 输出 ──
    draw_arrow(ax, (11.2, y_main), (11.8, y_main), "#555")
    draw_box(ax, (11.8, y_main - bh / 2), 2.2, bh,
             "预测掩码\n1024×1024", "#27AE60", 9)

    # ── Prompt Encoder（下方） ──
    draw_box(ax, (0.0, y_prompt - 0.35), 1.8, 0.7,
             "Box 提示", "#F39C12", 9, text_color="black")
    draw_arrow(ax, (1.8, y_prompt), (2.3, y_prompt), "#F39C12")
    draw_box(ax, (2.3, y_prompt - 0.35), bw, 0.7,
             "Prompt Encoder\n(冻结)", "#F39C12", 9, text_color="black")

    # Prompt → Decoder
    draw_arrow(ax, (4.8, y_prompt), (9.95, y_main - bh / 2), "#F39C12")

    # ── 特征维度标注 ──
    ax.text(3.55, y_main - bh / 2 - 0.25, "64×64×256", fontsize=7,
            ha="center", color="#7F8C8D", fontstyle="italic")
    ax.text(6.75, y_main - bh / 2 - 0.25, "64×64×256", fontsize=7,
            ha="center", color="#7F8C8D", fontstyle="italic")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "adapter_pipeline.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
