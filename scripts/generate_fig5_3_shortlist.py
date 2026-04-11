#!/usr/bin/env python3
"""Generate a small curated shortlist of chapter-5 preview figures.

This script avoids generating thousands of previews. It runs 12 targeted
`preview` renders (4 pancreas + 4 esophagus + 4 adjacency candidates) so the
user can quickly select Fig.5.3 rows.

Usage:
  python scripts/generate_fig5_3_shortlist.py \
    --config thesis-medsam/figures/chapter5_manual_config.json \
    --output-dir thesis-medsam/figures/previews_small
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


SHORTLIST = [
    # (a) Weak boundary: Pancreas
    ("CT_Abd_FLARE22_Tr_0030.npz", 19, "a_pancreas_0030_019.png"),
    ("CT_Abd_FLARE22_Tr_0025.npz", 19, "a_pancreas_0025_019.png"),
    ("CT_Abd_FLARE22_Tr_0007.npz", 55, "a_pancreas_0007_055.png"),
    ("CT_Abd_FLARE22_Tr_0035.npz", 27, "a_pancreas_0035_027.png"),
    # (b) Small thin organ: Esophagus
    ("CT_Abd_FLARE22_Tr_0031.npz", 95, "b_esophagus_0031_095.png"),
    ("CT_Abd_FLARE22_Tr_0031.npz", 96, "b_esophagus_0031_096.png"),
    ("CT_Abd_FLARE22_Tr_0006.npz", 83, "b_esophagus_0006_083.png"),
    ("CT_Abd_FLARE22_Tr_0026.npz", 47, "b_esophagus_0026_047.png"),
    # (c) Adjacent organs (fallback candidates, RAG/LAG)
    ("CT_Abd_FLARE22_Tr_0023.npz", 44, "c_lag_0023_044.png"),
    ("CT_Abd_FLARE22_Tr_0031.npz", 54, "c_rag_0031_054.png"),
    ("CT_Abd_FLARE22_Tr_0038.npz", 59, "c_rag_0038_059.png"),
    ("CT_Abd_FLARE22_Tr_0002.npz", 55, "c_rag_0002_055.png"),
]


def run_preview(config: str, case_name: str, slice_idx: int, output_path: str) -> None:
    cmd = [
        "python",
        "scripts/generate_chapter5_manual_figures.py",
        "preview",
        "--config",
        config,
        "--case-name",
        case_name,
        "--slice-idx",
        str(slice_idx),
        "--output",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 12 curated Fig.5.3 shortlist previews")
    parser.add_argument(
        "--config",
        default="thesis-medsam/figures/chapter5_manual_config.json",
        help="Path to chapter5 config JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="thesis-medsam/figures/previews_small",
        help="Directory for shortlist preview images",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(SHORTLIST)
    for idx, (case_name, slice_idx, out_name) in enumerate(SHORTLIST, start=1):
        out_path = str(out_dir / out_name)
        print(f"[{idx}/{total}] {case_name} slice={slice_idx} -> {out_path}")
        run_preview(args.config, case_name, slice_idx, out_path)

    print(f"[DONE] Generated {total} shortlist previews in: {out_dir}")


if __name__ == "__main__":
    main()
