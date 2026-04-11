#!/usr/bin/env python3
"""Select Fig.5.3 candidates by ordered per-slice Dice relation.

Goal:
  Find slices where model performance follows a desired order, e.g.
  Baseline <= BL <= BL+MSL, then optionally render previews directly.

Example:
  python scripts/select_fig5_3_ordered_candidates.py \
    --config thesis-medsam/figures/chapter5_manual_config.json \
    --organ-id 4 --organ-id 10 --organ-id 7 --organ-id 8 \
    --topk 8 --min-area 80 --strict \
    --render-dir thesis-medsam/figures/previews_ordered
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
from typing import Any


def load_ch5_module() -> Any:
    mod_path = Path(__file__).resolve().parent / "generate_chapter5_manual_figures.py"
    spec = importlib.util.spec_from_file_location("ch5_manual", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from: {mod_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def satisfy_order(
    a0: float,
    a3r3: float,
    c3: float,
    strict: bool,
    eps: float,
) -> bool:
    if strict:
        return (a3r3 - a0) > eps and (c3 - a3r3) > eps
    return (a3r3 - a0) >= -eps and (c3 - a3r3) >= -eps


def sanitize_stem(case_name: str) -> str:
    return Path(case_name).stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Select ordered-Dice Fig5.3 candidates")
    parser.add_argument(
        "--config",
        default="thesis-medsam/figures/chapter5_manual_config.json",
        help="Path to chapter5 config JSON",
    )
    parser.add_argument(
        "--organ-id",
        type=int,
        action="append",
        required=True,
        help="Organ id(s), repeatable",
    )
    parser.add_argument("--topk", type=int, default=8, help="Top-K candidates per organ")
    parser.add_argument("--min-area", type=int, default=80, help="Minimum GT area per organ per slice")
    parser.add_argument("--strict", action="store_true", help="Require Baseline < BL < BL+MSL")
    parser.add_argument(
        "--min-total-gain",
        type=float,
        default=0.005,
        help="Minimum (BL+MSL - Baseline) Dice gain",
    )
    parser.add_argument(
        "--min-step-gain",
        type=float,
        default=0.0,
        help="Minimum gain for each step (BL-Baseline and BL+MSL-BL)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-9,
        help="Numeric tolerance for order checks",
    )
    parser.add_argument(
        "--output-csv",
        default="thesis-medsam/figures/fig5_3_ordered_candidates.csv",
        help="Output csv path",
    )
    parser.add_argument(
        "--render-dir",
        default="",
        help="If set, render preview PNGs for selected candidates into this dir",
    )
    parser.add_argument(
        "--limit-renders",
        type=int,
        default=0,
        help="Optional cap for total rendered previews (0 means render all selected)",
    )
    args = parser.parse_args()

    ch5 = load_ch5_module()
    config = ch5.load_json(args.config)
    index = ch5.build_case_index(config.get("data_root"), config["pred_dirs"])
    titles = config.get("titles", ch5.DEFAULT_TITLES)
    columns = ["GT", "A0", "A3R3", "C3"]

    records_by_organ: dict[int, list[dict[str, Any]]] = {oid: [] for oid in args.organ_id}

    for case_name in index.case_names:
        ref = ch5.get_reference_arrays(index, case_name)
        gt_vol = ch5.first_present(ref, ["gts", "gt", "mask"])
        if gt_vol is None:
            continue

        pred_a0 = ch5.first_present(ch5.get_prediction_arrays(index, case_name, "A0"), ["segs", "seg", "pred", "mask"])
        pred_a3r3 = ch5.first_present(ch5.get_prediction_arrays(index, case_name, "A3R3"), ["segs", "seg", "pred", "mask"])
        pred_c3 = ch5.first_present(ch5.get_prediction_arrays(index, case_name, "C3"), ["segs", "seg", "pred", "mask"])

        if pred_a0 is None or pred_a3r3 is None or pred_c3 is None:
            continue

        for slice_idx in range(gt_vol.shape[0]):
            gt_slice = ch5.get_volume_slice(gt_vol, slice_idx)
            a0_slice = ch5.get_volume_slice(pred_a0, slice_idx)
            a3r3_slice = ch5.get_volume_slice(pred_a3r3, slice_idx)
            c3_slice = ch5.get_volume_slice(pred_c3, slice_idx)
            if gt_slice is None or a0_slice is None or a3r3_slice is None or c3_slice is None:
                continue

            for organ_id in args.organ_id:
                area = int((gt_slice == organ_id).sum())
                if area < args.min_area:
                    continue

                d_a0 = ch5.compute_slice_dice(a0_slice, gt_slice, label_id=organ_id)
                d_a3r3 = ch5.compute_slice_dice(a3r3_slice, gt_slice, label_id=organ_id)
                d_c3 = ch5.compute_slice_dice(c3_slice, gt_slice, label_id=organ_id)

                if not satisfy_order(d_a0, d_a3r3, d_c3, strict=args.strict, eps=args.eps):
                    continue
                if (d_c3 - d_a0) < args.min_total_gain:
                    continue
                if (d_a3r3 - d_a0) < args.min_step_gain or (d_c3 - d_a3r3) < args.min_step_gain:
                    continue

                records_by_organ[organ_id].append(
                    {
                        "organ_id": organ_id,
                        "organ_name": ch5.ORGAN_MAP.get(organ_id, str(organ_id)),
                        "case_name": case_name,
                        "slice_idx": slice_idx,
                        "area": area,
                        "dice_a0": d_a0,
                        "dice_a3r3": d_a3r3,
                        "dice_c3": d_c3,
                        "gain_a3r3_over_a0": d_a3r3 - d_a0,
                        "gain_c3_over_a3r3": d_c3 - d_a3r3,
                        "gain_c3_over_a0": d_c3 - d_a0,
                    }
                )

    selected: list[dict[str, Any]] = []
    for organ_id, recs in records_by_organ.items():
        recs.sort(
            key=lambda r: (
                r["gain_c3_over_a0"],
                r["gain_c3_over_a3r3"],
                r["gain_a3r3_over_a0"],
                r["area"],
            ),
            reverse=True,
        )
        selected.extend(recs[: args.topk])

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "organ_id",
        "organ_name",
        "case_name",
        "slice_idx",
        "area",
        "dice_a0",
        "dice_a3r3",
        "dice_c3",
        "gain_a3r3_over_a0",
        "gain_c3_over_a3r3",
        "gain_c3_over_a0",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow(row)

    print(f"[OK] Selected {len(selected)} candidates, saved csv: {out_csv}")
    for organ_id in args.organ_id:
        count = len([r for r in selected if r["organ_id"] == organ_id])
        print(f"  organ {organ_id}: {count}")

    if args.render_dir:
        render_dir = Path(args.render_dir)
        render_dir.mkdir(parents=True, exist_ok=True)
        rendered = 0
        for rec in selected:
            if args.limit_renders > 0 and rendered >= args.limit_renders:
                break
            stem = sanitize_stem(rec["case_name"])
            out_name = (
                f"o{rec['organ_id']}_{rec['organ_name'].replace('.', '')}_"
                f"{stem}_slice{int(rec['slice_idx']):03d}.png"
            )
            selection = {
                "case_name": rec["case_name"],
                "slice_idx": int(rec["slice_idx"]),
                "columns": columns,
                "titles": titles,
                "show_slice_dice": True,
                "output_path": str(render_dir / out_name),
                "dpi": 500,
                "width_per_col": 4.8,
                "fig_height": 4.6,
            }
            ch5.render_mask_comparison(index, selection, titles)
            rendered += 1
        print(f"[OK] Rendered {rendered} previews to: {render_dir}")


if __name__ == "__main__":
    main()
