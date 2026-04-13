#!/usr/bin/env python3
"""Generate Fig.3 training loss figures from raw training logs.

Requires server training log files.
Expected log format contains at least:
    Epoch: N, Loss: X.XXXX

Usage:
  python scripts/generate_fig3_training_curves.py \
    --log_a0 work_dir/baseline_train.log \
    --log_a3r3 work_dir/exp_logs/A3R3_train.log

Outputs:
  - thesis-medsam/figures/training_curves_baseline.pdf
  - thesis-medsam/figures/training_curves_bl.pdf
  - thesis-medsam/figures/training_curves.pdf (preview with two subplots)
"""

import argparse
import os
import re
import sys
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
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


def validate_curve(name, epochs, losses):
    """Fail early when the log exists but does not contain plottable data."""
    if not epochs or not losses:
        print(f"[ERROR] {name} log has no valid Epoch/Loss entries", file=sys.stderr)
        sys.exit(1)


def save_single_curve(output_path, epochs, losses, color, title, stage_switch=None):
    """Save a standalone curve with its own axes and title."""
    fig, ax = plt.subplots(figsize=(6.2, 4.6))

    ax.plot(epochs, losses, color=color, linewidth=1.6, alpha=0.9)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    if stage_switch is not None and epochs and max(epochs) >= stage_switch:
        ax.axvline(x=stage_switch, color="#F39C12", linestyle="--", linewidth=1.0, alpha=0.8)
        ymax = max(losses)
        ymin = min(losses)
        ytext = ymax - 0.08 * max(ymax - ymin, 1e-6)
        ax.text(stage_switch + 2, ytext, "Stage Switch\n(Epoch 50)",
                fontsize=8, color="#F39C12", va="top")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_preview_figure(output_path, ep_a0, loss_a0, ep_a3r3, loss_a3r3):
    """Save a combined preview showing the final two-panel layout."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    axes[0].plot(ep_a0, loss_a0, color="#3498DB", linewidth=1.5, alpha=0.9)
    axes[0].set_title("Baseline (Dice+CE)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Training Loss", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(left=0)

    axes[1].plot(ep_a3r3, loss_a3r3, color="#E74C3C", linewidth=1.5, alpha=0.9)
    axes[1].axvline(x=50, color="#F39C12", linestyle="--", linewidth=1.0, alpha=0.8)
    ymax = max(loss_a3r3)
    ymin = min(loss_a3r3)
    ytext = ymax - 0.08 * max(ymax - ymin, 1e-6)
    axes[1].text(52, ytext, "Stage Switch\n(Epoch 50)",
                 fontsize=8, color="#F39C12", va="top")
    axes[1].set_title("BL (Balance Loss)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Training Loss", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(left=0)

    fig.suptitle("Baseline vs BL Training Loss Convergence\n(Note: y-axis scales may differ)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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

    validate_curve("A0", ep_a0, loss_a0)
    validate_curve("A3R3", ep_a3r3, loss_a3r3)

    out_a0 = os.path.join(OUT_DIR, "training_curves_baseline.pdf")
    out_a3r3 = os.path.join(OUT_DIR, "training_curves_bl.pdf")
    out_preview = os.path.join(OUT_DIR, "training_curves.pdf")

    save_single_curve(
        out_a0,
        ep_a0,
        loss_a0,
        color="#3498DB",
        title="Baseline (Dice+CE) Training Loss",
    )
    save_single_curve(
        out_a3r3,
        ep_a3r3,
        loss_a3r3,
        color="#E74C3C",
        title="BL (Balance Loss) Training Loss",
        stage_switch=50,
    )
    save_preview_figure(out_preview, ep_a0, loss_a0, ep_a3r3, loss_a3r3)

    print(f"[OK] Saved: {out_a0}")
    print(f"[OK] Saved: {out_a3r3}")
    print(f"[OK] Saved: {out_preview}")


if __name__ == "__main__":
    main()
