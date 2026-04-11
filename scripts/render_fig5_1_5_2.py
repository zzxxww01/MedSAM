#!/usr/bin/env python3
"""Render thesis Fig5.1 and Fig5.2 with canonical output paths.

Usage:
  python scripts/render_fig5_1_5_2.py \
    --config thesis-medsam/figures/chapter5_manual_config.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render thesis Fig5.1 and Fig5.2")
    parser.add_argument(
        "--config",
        default="thesis-medsam/figures/chapter5_manual_config.json",
        help="Path to chapter5 manual figure config JSON",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}", file=sys.stderr)
        return 1

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    # Force canonical output file names used by thesis.
    cfg.setdefault("fig5_1", {})
    cfg.setdefault("fig5_2", {})
    cfg["fig5_1"]["output_path"] = "thesis-medsam/figures/mask_comparison.png"
    cfg["fig5_2"]["output_path"] = "thesis-medsam/figures/boundary_detail.png"

    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    base = [sys.executable, "scripts/generate_chapter5_manual_figures.py", "render", "--config", str(config_path)]
    run(base + ["--figure", "fig5_1"])
    run(base + ["--figure", "fig5_2"])

    print("[OK] Rendered:")
    print("  - thesis-medsam/figures/mask_comparison.png")
    print("  - thesis-medsam/figures/boundary_detail.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

