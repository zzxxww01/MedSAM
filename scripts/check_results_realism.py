#!/usr/bin/env python3
"""Audit thesis result summaries for overly polished or weakly supported patterns.

This script is intentionally conservative: it emits warnings, not corrections.
It is designed for review of existing result tables, not for generating new data.

Usage:
  python scripts/check_results_realism.py
  python scripts/check_results_realism.py --input thesis-medsam/data/results_update_template.json
  python scripts/check_results_realism.py --rules thesis-medsam/data/results_realism_template.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_INPUT = Path("thesis-medsam/data/results_update_template.json")
DEFAULT_RULES = Path("thesis-medsam/data/results_realism_template.json")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs)
    den_y = sum((y - my) ** 2 for y in ys)
    den = math.sqrt(den_x * den_y)
    return 0.0 if den == 0 else num / den


def affine_fit_residual(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float, float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0, 0.0, 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return my, 0.0, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    intercept = my - slope * mx
    max_abs_res = max(abs(y - (intercept + slope * x)) for x, y in zip(xs, ys))
    return intercept, slope, max_abs_res


def sign(x: float, eps: float = 1e-12) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def is_monotone(values: Sequence[float], direction: str) -> bool:
    if len(values) < 2:
        return True
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    if direction == "up":
        return all(d >= -1e-12 for d in diffs)
    if direction == "down":
        return all(d <= 1e-12 for d in diffs)
    raise ValueError(direction)


def unique_value_count(rows: Sequence[dict], key: str) -> int:
    return len({row[key] for row in rows if key in row})


def metric_range_warnings(rows: Sequence[dict], ranges: Dict[str, dict], section: str) -> List[str]:
    warnings: List[str] = []
    for idx, row in enumerate(rows):
        label = row.get("config") or row.get("route") or row.get("method") or row.get("neg_ratio") or row.get("w_h") or idx
        for key, spec in ranges.items():
            if key not in row:
                continue
            value = row[key]
            if isinstance(value, (int, float)):
                if value < spec["min"] or value > spec["max"]:
                    warnings.append(
                        f"{section}: {label} has {key}={value} outside [{spec['min']}, {spec['max']}]"
                    )
    return warnings


def correlation_warnings(rows: Sequence[dict], section: str, rules: dict) -> List[str]:
    warnings: List[str] = []
    if len(rows) < 4:
        return warnings

    hd95 = [row["HD95_mean"] for row in rows]
    asd = [row["ASD_mean"] for row in rows]
    dsc = [row["DSC_mean"] for row in rows]
    dsc_std = [row["DSC_std"] for row in rows]

    hd_asd_corr = corr(hd95, asd)
    if hd_asd_corr > rules["warn_if_hd95_asd_correlation_exceeds"]:
        _, _, max_abs_res = affine_fit_residual(hd95, asd)
        warnings.append(
            f"{section}: HD95_mean vs ASD_mean correlation is {hd_asd_corr:.6f}; "
            f"max affine residual={max_abs_res:.6f}. Check for over-smoothed boundary metrics."
        )

    dsc_std_corr = corr(dsc, dsc_std)
    if abs(dsc_std_corr) > rules["warn_if_dsc_std_correlation_magnitude_exceeds"]:
        warnings.append(
            f"{section}: DSC_mean vs DSC_std correlation magnitude is {abs(dsc_std_corr):.6f}. "
            "Std values may be too narratively aligned with means."
        )

    _, _, max_abs_res = affine_fit_residual(hd95, asd)
    if max_abs_res < rules["warn_if_two_metric_columns_are_nearly_affine_with_max_abs_residual_below"]:
        warnings.append(
            f"{section}: ASD_mean is nearly an affine transform of HD95_mean "
            f"(max residual {max_abs_res:.6f})."
        )

    if unique_value_count(rows, "DSC_std") < rules["warn_if_metric_std_unique_value_count_below"]:
        warnings.append(f"{section}: DSC_std has too few distinct values.")
    if unique_value_count(rows, "HD95_std") < rules["warn_if_metric_std_unique_value_count_below"]:
        warnings.append(f"{section}: HD95_std has too few distinct values.")
    if unique_value_count(rows, "ASD_std") < rules["warn_if_metric_std_unique_value_count_below"]:
        warnings.append(f"{section}: ASD_std has too few distinct values.")
    return warnings


def curve_warnings(rows: Sequence[dict], value_key: str, curve_name: str) -> List[str]:
    warnings: List[str] = []
    if len(rows) < 4:
        return warnings

    xs = [row[value_key] for row in rows]
    dsc = [row["DSC_mean"] for row in rows]
    hd95 = [row["HD95_mean"] for row in rows]
    asd = [row["ASD_mean"] for row in rows]

    dsc_peak = max(range(len(dsc)), key=dsc.__getitem__)
    hd95_valley = min(range(len(hd95)), key=hd95.__getitem__)
    asd_valley = min(range(len(asd)), key=asd.__getitem__)

    if dsc_peak == hd95_valley == asd_valley:
        left_dsc = is_monotone(dsc[: dsc_peak + 1], "up")
        right_dsc = is_monotone(dsc[dsc_peak:], "down")
        left_hd = is_monotone(hd95[: hd95_valley + 1], "down")
        right_hd = is_monotone(hd95[hd95_valley:], "up")
        left_asd = is_monotone(asd[: asd_valley + 1], "down")
        right_asd = is_monotone(asd[asd_valley:], "up")
        if left_dsc and right_dsc and left_hd and right_hd and left_asd and right_asd:
            warnings.append(
                f"{curve_name}: DSC peak and HD95/ASD valleys all occur at x={xs[dsc_peak]}, "
                "with fully monotone sides. Verify this is raw output rather than smoothed narrative data."
            )

    dsc_diffs = [dsc[i + 1] - dsc[i] for i in range(len(dsc) - 1)]
    hd95_diffs = [hd95[i + 1] - hd95[i] for i in range(len(hd95) - 1)]
    asd_diffs = [asd[i + 1] - asd[i] for i in range(len(asd) - 1)]
    same_sign_steps = 0
    for a, b, c in zip(dsc_diffs, hd95_diffs, asd_diffs):
        if sign(a) == -sign(b) == -sign(c) and sign(a) != 0:
            same_sign_steps += 1
    if same_sign_steps == len(dsc_diffs):
        warnings.append(
            f"{curve_name}: every step changes all three metrics in a perfectly aligned direction. "
            "Real nearby ablations often show at least one local metric disagreement."
        )
    return warnings


def collect_rows(data: dict) -> Dict[str, List[dict]]:
    return {
        "chapter3.neg_ratio_ablation": data["chapter3"]["neg_ratio_ablation"],
        "chapter3.wh_ablation": data["chapter3"]["wh_ablation"],
        "chapter3.unified_loss_table": data["chapter3"]["unified_loss_table"],
        "chapter4.route_selection_table": data["chapter4"]["route_selection_table"],
        "chapter5.main_result_rows": data["chapter5"]["main_result_rows"],
    }


def repeated_triplets(data: dict) -> List[str]:
    triplets: Dict[Tuple[float, float, float], List[str]] = {}

    for name, row in data["core_metrics"].items():
        trip = (row["DSC_mean"], row["HD95_mean"], row["ASD_mean"])
        triplets.setdefault(trip, []).append(f"core::{name}")

    for row in data["chapter3"]["unified_loss_table"]:
        trip = (row["DSC_mean"], row["HD95_mean"], row["ASD_mean"])
        triplets.setdefault(trip, []).append(f"unified::{row['config']}")

    for row in data["chapter4"]["route_selection_table"]:
        trip = (row["DSC_mean"], row["HD95_mean"], row["ASD_mean"])
        triplets.setdefault(trip, []).append(f"route::{row['route']}")

    for row in data["chapter5"]["main_result_rows"]:
        trip = (row["DSC_mean"], row["HD95_mean"], row["ASD_mean"])
        triplets.setdefault(trip, []).append(f"ch5::{row['method']}")

    messages: List[str] = []
    for trip, labels in triplets.items():
        if len(labels) > 1:
            messages.append(f"Repeated anchor triplet {trip}: {', '.join(labels)}")
    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit thesis metric summaries for realism warnings.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to result JSON.")
    parser.add_argument("--rules", default=str(DEFAULT_RULES), help="Path to realism rules JSON.")
    args = parser.parse_args()

    data = load_json(Path(args.input))
    rules = load_json(Path(args.rules))
    rows_by_section = collect_rows(data)
    global_rules = rules["global_constraints"]["table_level_checks"]
    metric_ranges = rules["global_constraints"]["metric_ranges"]

    warnings: List[str] = []

    for section, rows in rows_by_section.items():
        warnings.extend(metric_range_warnings(rows, metric_ranges, section))
        warnings.extend(correlation_warnings(rows, section, global_rules))

    unified_lookup = {row["config"]: row for row in data["chapter3"]["unified_loss_table"]}
    alpha_rows = [unified_lookup[name] for name in data["chapter3"]["figure_3_2_mapping"]["alpha_curve_configs"]]
    t1_rows = [unified_lookup[name] for name in data["chapter3"]["figure_3_2_mapping"]["t1_curve_configs"]]
    warnings.extend(curve_warnings(alpha_rows, "alpha", "chapter3.alpha_curve"))
    warnings.extend(curve_warnings(t1_rows, "T1", "chapter3.t1_curve"))

    repeated = repeated_triplets(data)

    print("Results Realism Audit")
    print("=====================")
    print(f"Input: {args.input}")
    print(f"Rules: {args.rules}")
    print()

    if repeated:
        print("Repeated Anchor Triplets")
        print("------------------------")
        for msg in repeated:
            print(f"- {msg}")
        print()

    if warnings:
        print("Warnings")
        print("--------")
        for msg in warnings:
            print(f"- {msg}")
        print()
        print(f"Summary: {len(warnings)} warning(s). Review raw case-level evidence before trusting these summaries.")
        return 1

    print("No warnings triggered by current soft rules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
