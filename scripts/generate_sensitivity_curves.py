"""
生成 Balance Loss 超参数敏感性折线图（双纵轴）
输出: figures/sensitivity_alpha.pdf, figures/sensitivity_t1.pdf
用法: python scripts/generate_sensitivity_curves.py
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'

fig_dir = 'thesis-medsam/figures'

# ============ 图A: α 敏感性 ============
alphas = [0.3, 0.5, 0.7, 1.0]
dsc_a  = [0.9516, 0.9543, 0.9436, 0.9278]
hd95_a = [3.4285, 3.1847, 4.1835, 5.5417]

fig, ax1 = plt.subplots(figsize=(4.5, 3.2))
color_dsc, color_hd = '#2166AC', '#B2182B'

ln1 = ax1.plot(alphas, dsc_a, 'o-', color=color_dsc, linewidth=1.8,
               markersize=6, label='DSC', zorder=3)
ax1.set_xlabel(r'$\alpha$', fontsize=12)
ax1.set_ylabel('DSC ↑', color=color_dsc, fontsize=11)
ax1.tick_params(axis='y', labelcolor=color_dsc)
ax1.set_ylim(0.920, 0.960)
ax1.set_xticks(alphas)

ax2 = ax1.twinx()
ln2 = ax2.plot(alphas, hd95_a, 's--', color=color_hd, linewidth=1.8,
               markersize=6, label='HD95', zorder=3)
ax2.set_ylabel('HD95 (mm) ↓', color=color_hd, fontsize=11)
ax2.tick_params(axis='y', labelcolor=color_hd)
ax2.set_ylim(2.5, 6.0)

# 标注最优点
ax1.annotate(f'{dsc_a[1]:.4f}', (alphas[1], dsc_a[1]),
             textcoords="offset points", xytext=(8, 8),
             fontsize=8, color=color_dsc, fontweight='bold')
ax2.annotate(f'{hd95_a[1]:.4f}', (alphas[1], hd95_a[1]),
             textcoords="offset points", xytext=(8, -12),
             fontsize=8, color=color_hd, fontweight='bold')

# 最优点垂线
ax1.axvline(x=0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='lower left', fontsize=9, framealpha=0.9)

ax1.set_title(r'(a) Inter-CBL weight $\alpha$ sensitivity', fontsize=10, pad=8)
fig.tight_layout()
fig.savefig(f'{fig_dir}/sensitivity_alpha.pdf', bbox_inches='tight', dpi=300)
plt.close()

# ============ 图B: T1 敏感性 ============
t1s    = [0, 30, 50, 70, 100]
dsc_t  = [0.9463, 0.9501, 0.9543, 0.9530, 0.9512]
hd95_t = [3.7524, 3.4182, 3.1847, 3.2714, 3.3842]

fig, ax1 = plt.subplots(figsize=(4.5, 3.2))

ln1 = ax1.plot(t1s, dsc_t, 'o-', color=color_dsc, linewidth=1.8,
               markersize=6, label='DSC', zorder=3)
ax1.set_xlabel(r'$T_1$ (epoch)', fontsize=12)
ax1.set_ylabel('DSC ↑', color=color_dsc, fontsize=11)
ax1.tick_params(axis='y', labelcolor=color_dsc)
ax1.set_ylim(0.942, 0.958)
ax1.set_xticks(t1s)

ax2 = ax1.twinx()
ln2 = ax2.plot(t1s, hd95_t, 's--', color=color_hd, linewidth=1.8,
               markersize=6, label='HD95', zorder=3)
ax2.set_ylabel('HD95 (mm) ↓', color=color_hd, fontsize=11)
ax2.tick_params(axis='y', labelcolor=color_hd)
ax2.set_ylim(3.0, 3.9)

ax1.annotate(f'{dsc_t[2]:.4f}', (t1s[2], dsc_t[2]),
             textcoords="offset points", xytext=(8, 8),
             fontsize=8, color=color_dsc, fontweight='bold')
ax2.annotate(f'{hd95_t[2]:.4f}', (t1s[2], hd95_t[2]),
             textcoords="offset points", xytext=(8, -12),
             fontsize=8, color=color_hd, fontweight='bold')

ax1.axvline(x=50, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='lower right', fontsize=9, framealpha=0.9)

ax1.set_title(r'(b) Two-stage switchover $T_1$ sensitivity', fontsize=10, pad=8)
fig.tight_layout()
fig.savefig(f'{fig_dir}/sensitivity_t1.pdf', bbox_inches='tight', dpi=300)
plt.close()

print("Done: sensitivity_alpha.pdf, sensitivity_t1.pdf")
