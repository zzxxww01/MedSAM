#!/usr/bin/env python3
"""One-shot Fig5.3 candidate generation with strict ordering and high gain.

Fixed policy:
  - Strictly increasing Dice: Baseline < BL < BL+MSL
  - Minimum total gain: BL+MSL - Baseline >= 0.2

Usage:
  python scripts/generate_fig5_3_strict_gain02.py

Optional:
  python scripts/generate_fig5_3_strict_gain02.py \
    --topk 12 \
    --output-csv thesis-medsam/figures/fig5_3_strict_gain02.csv \
    --render-dir thesis-medsam/figures/previews_strict_gain02
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Fig5.3 candidates with strict order and >=0.2 total gain"
    )
    parser.add_argument(
        "--config",
        default="thesis-medsam/figures/chapter5_manual_config.json",
        help="Path to chapter5 config JSON",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=12,
        help="Top-K candidates per organ",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=50,
        help="Minimum GT area per organ per slice",
    )
    parser.add_argument(
        "--output-csv",
        default="thesis-medsam/figures/fig5_3_strict_gain02.csv",
        help="Output csv path",
    )
    parser.add_argument(
        "--render-dir",
        default="thesis-medsam/figures/previews_strict_gain02",
        help="Output directory for rendered previews",
    )
    parser.add_argument(
        "--limit-renders",
        type=int,
        default=0,
        help="Optional cap for rendered previews (0 means render all)",
    )
    args = parser.parse_args()

    Path(args.render_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "scripts/select_fig5_3_ordered_candidates.py",
        "--config",
        args.config,
        "--organ-id",
        "4",
        "--organ-id",
        "10",
        "--organ-id",
        "7",
        "--organ-id",
        "8",
        "--topk",
        str(args.topk),
        "--min-area",
        str(args.min_area),
        "--strict",
        "--min-total-gain",
        "0.2",
        "--min-step-gain",
        "0.0",
        "--output-csv",
        args.output_csv,
        "--render-dir",
        args.render_dir,
    ]
    if args.limit_renders > 0:
        cmd.extend(["--limit-renders", str(args.limit_renders)])

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
