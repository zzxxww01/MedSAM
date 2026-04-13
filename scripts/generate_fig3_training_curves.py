#!/usr/bin/env python3
"""Fig.3 A0 vs A3R3 Training Loss Curves

Requires server training log files.
Log format: Time: YYYYMMDD-HHMM, Epoch: N, Loss: X.XXXX

Usage:
  python scripts/generate_fig3_training_curves.py \
    --log_a0  work_dir/baseline_train.log \
    --log_a3r3 work_dir/exp_logs/A3R3_train.log

Output: thesis-medsam/figures/training_curves.pdf
"""

import argparse
import os
import re
import sys
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Times New Roman", "SimSun", "serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "thesis-medsam", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

LOG_PATTERN = re.compile(
    r"Epoch:\s*(\d+),\s*Loss:\s*([\d.]+)"
)


def parse_log(path):
    """Parse training log and return (epochs, losses)."""
    epochs, losses = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    if not epochs:
        print(f"[WARN] No valid entries found in {path}", file=sys.stderr)
    return epochs, losses


def main():
    parser = argparse.ArgumentParser(description="Generate training curve comparison")
    parser.add_argument("--log_a0", type=str,
                        default="work_dir/baseline_train.log")
    parser.add_argument("--log_a3r3", type=str,
                        default="work_dir/exp_logs/A3R3_train.log")
    args = parser.parse_args()

    if not os.path.exists(args.log_a0):
        print(f"[ERROR] A0 log not found: {args.log_a0}")
        sys.exit(1)
    if not os.path.exists(args.log_a3r3):
        print(f"[ERROR] A3R3 log not found: {args.log_a3r3}")
        sys.exit(1)

    ep_a0, loss_a0 = parse_log(args.log_a0)
    ep_a3r3, loss_a3r3 = parse_log(args.log_a3r3)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(ep_a0, loss_a0, color="#3498DB", linewidth=1.5, alpha=0.85,
            label="Baseline (Dice+CE)")
    ax.plot(ep_a3r3, loss_a3r3, color="#E74C3C", linewidth=1.5, alpha=0.85,
            label="Balance Loss (α=0.5)")

    # Mark two-stage switch point
    if len(ep_a3r3) > 50:
        ax.axvline(x=50, color="#F39C12", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(52, ax.get_ylim()[1] * 0.9, "两阶段切换\n(Epoch 50)",
                fontsize=8, color="#F39C12")

    ax.set_xlabel("训练轮次 (Epoch)", fontsize=12)
    ax.set_ylabel("训练损失", fontsize=12)
    ax.set_title("Baseline 与 Balance Loss 训练损失对比", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "training_curves.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()

