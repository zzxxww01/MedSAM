"""
为 mask_comparison.pdf 和 boundary_detail.pdf 添加器官颜色图例
输出: figures/organ_legend.pdf (独立图例条, 可在LaTeX中通过 \includegraphics 放在图下方)

也可以直接修改 generate_fig6_mask_comparison.py, 在图底部加入图例。
两种方式都在下面提供。

用法: python scripts/generate_organ_legend.py
"""
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch

matplotlib.rcParams['font.family'] = 'Times New Roman'

# 13类器官的颜色映射 (与你的可视化脚本中的 colormap 保持一致)
# 如果你用的是不同的颜色方案, 请替换下面的颜色值
ORGAN_COLORS = {
    'Liver':       '#E41A1C',
    'R.Kidney':    '#377EB8',
    'Spleen':      '#4DAF4A',
    'Pancreas':    '#984EA3',
    'Aorta':       '#FF7F00',
    'IVC':         '#FFFF33',
    'RAG':         '#A65628',
    'LAG':         '#F781BF',
    'Gallbladder': '#999999',
    'Esophagus':   '#66C2A5',
    'Stomach':     '#FC8D62',
    'Duodenum':    '#8DA0CB',
    'L.Kidney':    '#E78AC3',
}


def generate_standalone_legend():
    """方式1: 生成独立的图例条 PDF, 在 LaTeX 中单独引入"""
    fig, ax = plt.subplots(figsize=(10, 0.6))
    ax.axis('off')

    legend_elements = [
        Patch(facecolor=color, edgecolor='black', linewidth=0.5, label=name)
        for name, color in ORGAN_COLORS.items()
    ]

    ax.legend(handles=legend_elements, loc='center', ncol=7,
             fontsize=7.5, frameon=False,
             handlelength=1.2, handletextpad=0.4,
             columnspacing=1.0)

    fig.savefig('thesis-medsam/figures/organ_legend.pdf',
                bbox_inches='tight', dpi=300, pad_inches=0.02)
    plt.close()
    print("Done: organ_legend.pdf")
    print()
    print("在 LaTeX 中使用:")
    print(r"""
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{figures/mask_comparison.pdf}
  \\[2pt]
  \includegraphics[width=0.85\textwidth]{figures/organ_legend.pdf}
  \caption{代表性切片预测掩码可视化}
  \label{fig:mask_comparison}
\end{figure}
""")


def show_how_to_add_to_existing_script():
    """方式2: 在已有的 generate_fig6 脚本中添加图例的代码片段"""
    print("=" * 60)
    print("方式2: 在 generate_fig6_mask_comparison.py 末尾添加以下代码:")
    print("=" * 60)
    print("""
# --- 在 fig.savefig() 之前添加 ---

from matplotlib.patches import Patch

organ_colors = {
    'Liver': '#E41A1C', 'R.Kidney': '#377EB8', 'Spleen': '#4DAF4A',
    'Pancreas': '#984EA3', 'Aorta': '#FF7F00', 'IVC': '#FFFF33',
    'RAG': '#A65628', 'LAG': '#F781BF', 'Gallbladder': '#999999',
    'Esophagus': '#66C2A5', 'Stomach': '#FC8D62', 'Duodenum': '#8DA0CB',
    'L.Kidney': '#E78AC3',
}

legend_elements = [
    Patch(facecolor=c, edgecolor='black', linewidth=0.5, label=name)
    for name, c in organ_colors.items()
]

fig.legend(handles=legend_elements, loc='lower center',
           ncol=7, fontsize=7, frameon=False,
           bbox_to_anchor=(0.5, -0.04))

plt.subplots_adjust(bottom=0.08)  # 给图例留空间
""")


if __name__ == '__main__':
    generate_standalone_legend()
    show_how_to_add_to_existing_script()
