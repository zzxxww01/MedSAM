#!/usr/bin/env python3
"""Per-organ DSC analysis across experiments (A0 / A3R3 / C2 / C3).

Reads existing prediction NPZ files (saved by save_all_predictions.sh),
computes per-organ Dice Similarity Coefficient for each experiment,
and outputs:
  1. A CSV table with per-organ DSC for each experiment
  2. A LaTeX-formatted table ready for thesis insertion
  3. Wilcoxon signed-rank test p-values for key comparisons

Usage (on server):
  python scripts/analyze_per_organ_dsc.py

Requirements:
  - Prediction NPZ files in work_dir/eval_predictions/{A0,A3R3,C2,C3}/
  - Each NPZ contains keys: 'segs' (predicted), 'gts' (ground truth)
  - No GPU needed, runs in < 1 minute

Output:
  - thesis-medsam/tables/per_organ_dsc.csv
  - thesis-medsam/tables/per_organ_dsc.tex
  - Console summary
"""

import glob
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPERIMENTS = ["A0", "A3R3", "C2", "C3"]

SEARCH_PATTERNS = [
    "work_dir/eval_predictions/{name}/*.npz",
    "work_dir/eval_metrics/{name}_predictions/*.npz",
    "work_dir/{name}_predictions/*.npz",
    "work_dir/predictions/{name}/*.npz",
]

ORGAN_LABELS = {
    1: "Liver",
    2: "R.Kidney",
    3: "Spleen",
    4: "Pancreas",
    5: "Aorta",
    6: "IVC",
    7: "RAG",
    8: "LAG",
    9: "Gallbladder",
    10: "Esophagus",
    11: "Stomach",
    12: "Duodenum",
    13: "L.Kidney",
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "thesis-medsam", "tables")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def find_predictions(name):
    """Search for prediction NPZ files."""
    for pattern in SEARCH_PATTERNS:
        matches = glob.glob(pattern.format(name=name))
        if matches:
            return sorted(matches)
    return []


def compute_dice(pred, gt):
    """Compute Dice coefficient for binary masks."""
    intersection = np.sum(pred & gt)
    volume_sum = np.sum(pred) + np.sum(gt)
    if volume_sum == 0:
        return 1.0  # both empty -> perfect match
    return 2.0 * intersection / volume_sum


def to_3d(mask):
    """Ensure mask is at least 3D."""
    mask = np.squeeze(mask)
    if mask.ndim == 2:
        mask = mask[np.newaxis, ...]
    return mask


def compute_per_organ_dice(segs, gts):
    """Compute DSC for each of the 13 organs in a single case.

    Returns: dict {label_id: dice_score}
    """
    segs = to_3d(segs)
    gts = to_3d(gts)
    results = {}
    for label_id in ORGAN_LABELS:
        gt_bin = (gts == label_id)
        seg_bin = (segs == label_id)
        # Only compute if organ exists in GT
        if gt_bin.sum() > 0:
            results[label_id] = compute_dice(seg_bin, gt_bin)
        # If organ not in GT but predicted, that's a false positive
        elif seg_bin.sum() > 0:
            results[label_id] = 0.0
        # Both empty: skip (organ not present in this case)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Find prediction files
    pred_files = {}
    for name in EXPERIMENTS:
        files = find_predictions(name)
        if files:
            pred_files[name] = files
            print(f"[OK] {name}: found {len(files)} prediction files")
        else:
            print(f"[WARN] {name}: no prediction files found")

    if not pred_files:
        print("\nNo prediction files found. Run save_all_predictions.sh first.")
        sys.exit(1)

    # Compute per-organ DSC for each experiment
    # Structure: {exp_name: {label_id: [dice_per_case]}}
    all_results = {}
    for name in EXPERIMENTS:
        if name not in pred_files:
            continue

        organ_dices = {lid: [] for lid in ORGAN_LABELS}

        for fpath in pred_files[name]:
            data = np.load(fpath, allow_pickle=True)
            segs = data.get("segs", data.get("pred", None))
            gts = data.get("gts", data.get("mask", None))

            if segs is None or gts is None:
                keys = list(data.keys())
                print(f"  [WARN] {fpath}: keys={keys}, skipping")
                continue

            case_dice = compute_per_organ_dice(segs, gts)
            for lid, d in case_dice.items():
                organ_dices[lid].append(d)

        all_results[name] = organ_dices
        print(f"  {name}: computed DSC for {len(pred_files[name])} cases")

    # ---------------------------------------------------------------------------
    # Aggregate and display results
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("Per-Organ DSC Results (mean ± std)")
    print("=" * 80)

    header = f"{'Organ':<14}"
    for name in EXPERIMENTS:
        if name in all_results:
            header += f"  {name:>12}"
    print(header)
    print("-" * len(header))

    # Store means for CSV/LaTeX output
    mean_table = {}  # {label_id: {exp_name: (mean, std, n)}}

    for lid, organ_name in ORGAN_LABELS.items():
        row = f"{organ_name:<14}"
        mean_table[lid] = {}
        for name in EXPERIMENTS:
            if name not in all_results:
                continue
            dices = all_results[name][lid]
            if dices:
                m, s = np.mean(dices), np.std(dices)
                mean_table[lid][name] = (m, s, len(dices))
                row += f"  {m:.4f}±{s:.3f}"
            else:
                mean_table[lid][name] = (0.0, 0.0, 0)
                row += f"  {'N/A':>12}"
        print(row)

    # Overall mean
    print("-" * len(header))
    row = f"{'Mean':<14}"
    overall_means = {}
    for name in EXPERIMENTS:
        if name not in all_results:
            continue
        all_organ_means = []
        for lid in ORGAN_LABELS:
            m, s, n = mean_table[lid].get(name, (0.0, 0.0, 0))
            if n > 0:
                all_organ_means.append(m)
        overall = np.mean(all_organ_means) if all_organ_means else 0.0
        overall_means[name] = overall
        row += f"  {overall:>12.4f}"
    print(row)

    # ---------------------------------------------------------------------------
    # Statistical significance tests
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("Statistical Significance (Wilcoxon signed-rank test, per-case DSC)")
    print("=" * 80)

    comparisons = [("A0", "A3R3"), ("A3R3", "C3"), ("A0", "C3")]

    for exp_a, exp_b in comparisons:
        if exp_a not in all_results or exp_b not in all_results:
            continue

        # Collect per-case mean DSC for both experiments
        files_a = pred_files.get(exp_a, [])
        files_b = pred_files.get(exp_b, [])
        n_cases = min(len(files_a), len(files_b))

        dsc_a_list = []
        dsc_b_list = []

        for i in range(n_cases):
            data_a = np.load(files_a[i], allow_pickle=True)
            data_b = np.load(files_b[i], allow_pickle=True)

            segs_a = data_a.get("segs", data_a.get("pred"))
            gts_a = data_a.get("gts", data_a.get("mask"))
            segs_b = data_b.get("segs", data_b.get("pred"))
            gts_b = data_b.get("gts", data_b.get("mask"))

            if segs_a is None or gts_a is None or segs_b is None or gts_b is None:
                continue

            d_a = compute_per_organ_dice(segs_a, gts_a)
            d_b = compute_per_organ_dice(segs_b, gts_b)

            # Mean DSC across present organs
            common_labels = set(d_a.keys()) & set(d_b.keys())
            if common_labels:
                dsc_a_list.append(np.mean([d_a[l] for l in common_labels]))
                dsc_b_list.append(np.mean([d_b[l] for l in common_labels]))

        if len(dsc_a_list) >= 5:
            try:
                from scipy.stats import wilcoxon
                stat, p_value = wilcoxon(dsc_a_list, dsc_b_list)
                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
                print(f"  {exp_a} vs {exp_b}: p={p_value:.6f} ({sig}), n={len(dsc_a_list)}")
            except ImportError:
                print(f"  {exp_a} vs {exp_b}: scipy not available, skipping test")
                # Fallback: paired t-test with numpy
                dsc_a_arr = np.array(dsc_a_list)
                dsc_b_arr = np.array(dsc_b_list)
                diff = dsc_b_arr - dsc_a_arr
                t_stat = np.mean(diff) / (np.std(diff, ddof=1) / np.sqrt(len(diff)))
                print(f"  {exp_a} vs {exp_b}: t-stat={t_stat:.4f}, mean_diff={np.mean(diff):.4f}")
        else:
            print(f"  {exp_a} vs {exp_b}: insufficient data (n={len(dsc_a_list)})")

    # ---------------------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------------------

    csv_path = os.path.join(OUT_DIR, "per_organ_dsc.csv")
    with open(csv_path, "w") as f:
        exps_present = [e for e in EXPERIMENTS if e in all_results]
        f.write("Organ," + ",".join(exps_present) + "\n")
        for lid, organ_name in ORGAN_LABELS.items():
            row_vals = []
            for name in exps_present:
                m, s, n = mean_table[lid].get(name, (0.0, 0.0, 0))
                row_vals.append(f"{m:.4f}" if n > 0 else "N/A")
            f.write(f"{organ_name},{','.join(row_vals)}\n")
        # Overall
        row_vals = [f"{overall_means.get(name, 0):.4f}" for name in exps_present]
        f.write(f"Mean,{','.join(row_vals)}\n")
    print(f"\n[OK] CSV saved: {csv_path}")

    # ---------------------------------------------------------------------------
    # Save LaTeX table
    # ---------------------------------------------------------------------------

    tex_path = os.path.join(OUT_DIR, "per_organ_dsc.tex")
    exps_present = [e for e in EXPERIMENTS if e in all_results]
    n_cols = len(exps_present)

    with open(tex_path, "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("  \\centering\n")
        f.write("  \\caption{Per-organ DSC comparison across experiments "
                "(FLARE22, 40 cases)}\n")
        f.write("  \\label{tab:per_organ_dsc}\n")
        f.write("  \\begin{tabular}{l" + "c" * n_cols + "}\n")
        f.write("    \\hline\n")
        f.write("    Organ")
        for name in exps_present:
            f.write(f" & {name}")
        f.write(" \\\\\n")
        f.write("    \\hline\n")

        for lid, organ_name in ORGAN_LABELS.items():
            # Find best score for bolding
            scores = {}
            for name in exps_present:
                m, s, n = mean_table[lid].get(name, (0.0, 0.0, 0))
                if n > 0:
                    scores[name] = m
            best_name = max(scores, key=scores.get) if scores else None

            f.write(f"    {organ_name}")
            for name in exps_present:
                m, s, n = mean_table[lid].get(name, (0.0, 0.0, 0))
                if n > 0:
                    val = f"{m:.4f}"
                    if name == best_name:
                        f.write(f" & \\textbf{{{val}}}")
                    else:
                        f.write(f" & {val}")
                else:
                    f.write(" & N/A")
            f.write(" \\\\\n")

        f.write("    \\hline\n")
        f.write("    Mean")
        best_overall = max(overall_means, key=overall_means.get) if overall_means else None
        for name in exps_present:
            val = f"{overall_means.get(name, 0):.4f}"
            if name == best_overall:
                f.write(f" & \\textbf{{{val}}}")
            else:
                f.write(f" & {val}")
        f.write(" \\\\\n")
        f.write("    \\hline\n")
        f.write("  \\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"[OK] LaTeX table saved: {tex_path}")

    # ---------------------------------------------------------------------------
    # Highlight key findings
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("Key Findings")
    print("=" * 80)

    if "A0" in all_results and "A3R3" in all_results:
        print("\nBalance Loss (A3R3 vs A0) - Top 5 improved organs:")
        improvements = []
        for lid, organ_name in ORGAN_LABELS.items():
            m_a0 = mean_table[lid].get("A0", (0, 0, 0))[0]
            m_a3r3 = mean_table[lid].get("A3R3", (0, 0, 0))[0]
            if m_a0 > 0:
                improvements.append((organ_name, m_a3r3 - m_a0, m_a0, m_a3r3))
        improvements.sort(key=lambda x: x[1], reverse=True)
        for name, delta, before, after in improvements[:5]:
            print(f"  {name:<14}: {before:.4f} -> {after:.4f} (+{delta:.4f})")

    if "A3R3" in all_results and "C3" in all_results:
        print("\nLG-Adapter (C3 vs A3R3) - Top 5 improved organs:")
        improvements = []
        for lid, organ_name in ORGAN_LABELS.items():
            m_a3r3 = mean_table[lid].get("A3R3", (0, 0, 0))[0]
            m_c3 = mean_table[lid].get("C3", (0, 0, 0))[0]
            if m_a3r3 > 0:
                improvements.append((organ_name, m_c3 - m_a3r3, m_a3r3, m_c3))
        improvements.sort(key=lambda x: x[1], reverse=True)
        for name, delta, before, after in improvements[:5]:
            sign = "+" if delta >= 0 else ""
            print(f"  {name:<14}: {before:.4f} -> {after:.4f} ({sign}{delta:.4f})")


if __name__ == "__main__":
    main()
