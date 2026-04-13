#!/usr/bin/env python3
"""Apply thesis metric template to tex tables/fig scripts.

Usage:
  python scripts/apply_results_template.py \
      --template thesis-medsam/data/results_update_template.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")


def f4(x) -> str:
    return f"{float(x):.4f}"


def f3(x) -> str:
    return f"{float(x):.3f}"


def f2(x) -> str:
    return f"{float(x):.2f}"


def fmt_triplet_pm(row: dict) -> Tuple[str, str, str]:
    dsc = f"{f4(row['DSC_mean'])}$\\pm${f3(row['DSC_std'])}"
    hd95 = f"{f4(row['HD95_mean'])}$\\pm${f2(row['HD95_std'])}"
    asd = f"{f4(row['ASD_mean'])}$\\pm${f3(row['ASD_std'])}"
    return dsc, hd95, asd


def replace_line_by_prefix(text: str, prefix_regex: str, new_line: str) -> str:
    pat = re.compile(prefix_regex, re.M)
    if not pat.search(text):
        raise RuntimeError(f"Pattern not found: {prefix_regex}")
    return pat.sub(lambda _: new_line, text, count=1)


def update_sensitivity_script(script_text: str, tpl: dict) -> str:
    ul = {r["config"]: r for r in tpl["chapter3"]["unified_loss_table"]}
    alpha_cfgs = tpl["chapter3"]["figure_3_2_mapping"]["alpha_curve_configs"]
    t1_cfgs = tpl["chapter3"]["figure_3_2_mapping"]["t1_curve_configs"]

    alphas = [ul[c]["alpha"] for c in alpha_cfgs]
    dsc_a = [ul[c]["DSC_mean"] for c in alpha_cfgs]
    hd95_a = [ul[c]["HD95_mean"] for c in alpha_cfgs]

    t1s = [ul[c]["T1"] for c in t1_cfgs]
    dsc_t = [ul[c]["DSC_mean"] for c in t1_cfgs]
    hd95_t = [ul[c]["HD95_mean"] for c in t1_cfgs]

    def py_list(vals: List[float], nd: int = 4) -> str:
        if nd == 0:
            return "[" + ", ".join(str(int(v)) for v in vals) + "]"
        return "[" + ", ".join(f"{float(v):.{nd}f}" for v in vals) + "]"

    script_text = re.sub(r"(?m)^alphas\s*=.*$", f"alphas = {py_list(alphas, 1)}", script_text)
    script_text = re.sub(r"(?m)^dsc_a\s*=.*$", f"dsc_a  = {py_list(dsc_a, 4)}", script_text)
    script_text = re.sub(r"(?m)^hd95_a\s*=.*$", f"hd95_a = {py_list(hd95_a, 4)}", script_text)
    script_text = re.sub(r"(?m)^t1s\s*=.*$", f"t1s    = {py_list(t1s, 0)}", script_text)
    script_text = re.sub(r"(?m)^dsc_t\s*=.*$", f"dsc_t  = {py_list(dsc_t, 4)}", script_text)
    script_text = re.sub(r"(?m)^hd95_t\s*=.*$", f"hd95_t = {py_list(hd95_t, 4)}", script_text)
    return script_text


def _alpha_disp(v) -> str:
    if v is None:
        return "--"
    if isinstance(v, str):
        return v.replace("->", "$\\to$")
    return f"{float(v):.1f}"


def _t1_disp(v) -> str:
    if v is None:
        return "--"
    return str(int(v))


def _tau_disp(v) -> str:
    if v is None:
        return "--"
    return f"{float(v):.1f}"


def update_chapter3_tables(ch3: str, tpl: dict) -> str:
    # neg_ratio table
    for row in tpl["chapter3"]["neg_ratio_ablation"]:
        dsc, hd95, asd = fmt_triplet_pm(row)
        label = "3.0（默认）" if abs(float(row["neg_ratio"]) - 3.0) < 1e-9 else f"{float(row['neg_ratio']):.1f}"
        metric = f"\\textbf{{{dsc}}} & \\textbf{{{hd95}}} & \\textbf{{{asd}}}" if "默认" in label else f"{dsc} & {hd95} & {asd}"
        new_line = f"    {label} & {metric} \\\\"
        ch3 = replace_line_by_prefix(ch3, rf"^\s*{re.escape(label)}\s*&.*\\\\$", new_line)

    # w_h table
    for row in tpl["chapter3"]["wh_ablation"]:
        dsc, hd95, asd = fmt_triplet_pm(row)
        w = float(row["w_h"])
        label = "2.0（默认）" if abs(w - 2.0) < 1e-9 else ("1.0（等权）" if abs(w - 1.0) < 1e-9 else f"{w:.1f}")
        metric = f"\\textbf{{{dsc}}} & \\textbf{{{hd95}}} & \\textbf{{{asd}}}" if "默认" in label else f"{dsc} & {hd95} & {asd}"
        new_line = f"    {label} & {metric} \\\\"
        ch3 = replace_line_by_prefix(ch3, rf"^\s*{re.escape(label)}\s*&.*\\\\$", new_line)

    # unified table
    row_label = {
        "Baseline": "Baseline",
        "Baseline-Focal": "Baseline-Focal",
        "Baseline-BdLoss": "Baseline-BdLoss",
        "Baseline-ABL": "Baseline-ABL",
        "Baseline-OHEM": "Baseline-OHEM",
        "Inter-only": "Inter-only",
        "Intra-tau0.5": r"Intra-$\tau$0.5",
        "Intra-tau0.7": r"Intra-$\tau$0.7",
        "Intra-tau0.8": r"Intra-$\tau$0.8",
        "Intra-tau0.9": r"Intra-$\tau$0.9",
        "BL-alpha0.3": r"BL-$\alpha$0.3",
        "BL": "BL",
        "BL-alpha0.7": r"BL-$\alpha$0.7",
        "BL-alpha1.0": r"BL-$\alpha$1.0",
        "BL-T0": "BL-T0",
        "BL-T30": "BL-T30",
        "BL-T50": r"BL$^\dagger$",
        "BL-T70": "BL-T70",
        "BL-T100": "BL-T100",
        "BL-tau0.8": r"BL-$\tau$0.8",
        "BL-prog": "BL-prog",
        "BL-cv": "BL-cv",
    }

    for row in tpl["chapter3"]["unified_loss_table"]:
        cfg = row["config"]
        if cfg not in row_label:
            continue
        dsc, hd95, asd = fmt_triplet_pm(row)
        alpha_val = row.get("alpha")
        if alpha_val is None and row.get("alpha_schedule"):
            alpha_val = row.get("alpha_schedule")
        alpha = _alpha_disp(alpha_val)
        t1 = _t1_disp(row["T1"])
        tau = _tau_disp(row["tau"])
        label = row_label[cfg]
        pattern = re.compile(rf"^(\s*{re.escape(label)}\s*&.*\\\\)$", re.M)
        m = pattern.search(ch3)
        if not m:
            raise RuntimeError(f"Unified row not found for config: {cfg}")
        line = m.group(1)
        body = line[:-2].rstrip()  # remove trailing \\
        cols = [c.strip() for c in body.split("&")]
        if len(cols) < 8:
            raise RuntimeError(f"Unexpected column count for config {cfg}: {line}")
        new_metric = f"{dsc} & {hd95} & {asd}"
        if cfg in ("BL", "BL-T50"):
            new_metric = f"\\textbf{{{dsc}}} & \\textbf{{{hd95}}} & \\textbf{{{asd}}}"
        metric_cols = [c.strip() for c in new_metric.split("&")]
        cols[2] = alpha
        cols[3] = t1
        cols[4] = tau
        cols[5] = metric_cols[0]
        cols[6] = metric_cols[1]
        cols[7] = metric_cols[2]
        indent = re.match(r"^\s*", line).group(0)
        new_line = indent + " & ".join(cols) + r" \\"
        ch3 = pattern.sub(lambda _: new_line, ch3, count=1)

    return ch3


def update_chapter4_table(ch4: str, tpl: dict) -> str:
    route_map = {r["route"]: r for r in tpl["chapter4"]["route_selection_table"]}
    row_prefix = {
        "BL": r"BL\s*&\s*最优损失主干\s*&\s*0",
        "BL+CA": r"BL\+CA\s*&\s*BL \+ Cross-Attention 融合\s*&\s*\\\$\\sim\\\$263K\\\$\^\*\\\$",
        "BL+LoRA-r4": r"BL\+LoRA-r4\s*&\s*BL \+ LoRA（冻结主干）\s*&\s*147,456",
        "BL+LoRA-r8": r"BL\+LoRA-r8\s*&\s*BL \+ LoRA（冻结主干）\s*&\s*294,912",
        "BL+LoRA-r16": r"BL\+LoRA-r16\s*&\s*BL \+ LoRA（冻结主干）\s*&\s*589,824",
        "BL+MSL": r"BL\+MSL\s*&\s*BL \+ 局部适配器（主干全参更新）\s*&\s*136,448",
    }
    for route, rx in row_prefix.items():
        row = route_map[route]
        dsc, hd95, asd = fmt_triplet_pm(row)
        metrics = f"{dsc} & {hd95} & {asd}"
        if route == "BL+MSL":
            metrics = f"\\textbf{{{dsc}}} & \\textbf{{{hd95}}} & \\textbf{{{asd}}}"
        pattern = rf"^(\s*{rx}\s*&\s*).*(\\\\)$"
        ch4 = re.sub(pattern, lambda m: f"{m.group(1)}{metrics} {m.group(2)}", ch4, count=1, flags=re.M)
    return ch4


def update_chapter5_tables(ch5: str, tpl: dict) -> str:
    main = {r["method"]: r for r in tpl["chapter5"]["main_result_rows"]}
    unified = {r["config"]: r for r in tpl.get("chapter3", {}).get("unified_loss_table", [])}

    def _replace_main(label: str, key: str, bold: bool = False):
        row = main[key]
        dsc, hd95, asd = fmt_triplet_pm(row)
        metrics = f"{dsc} & {hd95} & {asd}"
        if bold:
            metrics = f"\\textbf{{{dsc}}} & \\textbf{{{hd95}}} & \\textbf{{{asd}}}"
        pattern = rf"^(\s*{re.escape(label)}\s*&\s*).*(\\\\)$"
        return re.sub(pattern, lambda m: f"{m.group(1)}{metrics} {m.group(2)}", ch5, count=1, flags=re.M)

    ch5 = _replace_main("Baseline（MedSAM 基线）", "Baseline")
    # Keep Baseline-Focal in chapter5 aligned with chapter3 unified table if available.
    if "Baseline-Focal" in unified:
        row = unified["Baseline-Focal"]
        dsc, hd95, asd = fmt_triplet_pm(row)
        pattern = r"^(\s*Baseline-Focal（Dice\+Focal）\s*&\s*).*(\\\\)$"
        ch5 = re.sub(pattern, lambda m: f"{m.group(1)}{dsc} & {hd95} & {asd} {m.group(2)}", ch5, count=1, flags=re.M)
    ch5 = _replace_main("BL（Balance Loss）", "BL")
    ch5 = _replace_main("BL+MSL（Balance Loss + 局部适配器）", "BL+MSL", bold=True)

    lit = {r["method"]: r for r in tpl["chapter5"]["literature_table_rows"]}
    for method in ("BL（Balance Loss）", "BL+MSL（本文最终方案）"):
        row = lit[method]
        dsc = f4(row["DSC_mean"])
        hd95 = f4(row["HD95_mean"])
        asd = f4(row["ASD_mean"])
        metrics = f"{dsc} & {hd95} & {asd}"
        if "BL+MSL" in method:
            metrics = f"\\textbf{{{dsc}}} & \\textbf{{{hd95}}} & \\textbf{{{asd}}}"
        pattern = rf"^(\s*{re.escape(method)}\s*&\s*[^&]*&\s*GT box\s*&\s*).*(\\\\)$"
        ch5 = re.sub(pattern, lambda m: f"{m.group(1)}{metrics} {m.group(2)}", ch5, count=1, flags=re.M)

    return ch5


def update_core_metric_occurrences(ch3: str, ch4: str, ch5: str, tpl: dict) -> Tuple[str, str, str]:
    # Keep this intentionally narrow: synchronize frequent cross-chapter literal mentions for BL/Baseline/BL+MSL.
    old_new = []
    # We read current anchor values from template's "current" defaults is not guaranteed;
    # users may directly edit to new numbers. Here only rewrite known old literals if present.
    # This pass still helps avoid obvious drift in prose.
    bl = tpl["core_metrics"]["BL"]
    baseline = tpl["core_metrics"]["Baseline"]
    bl_msl = tpl["core_metrics"]["BL_MSL"]

    old_new += [("0.9543", f4(bl["DSC_mean"])), ("3.1847", f4(bl["HD95_mean"])), ("0.3562", f4(bl["ASD_mean"]))]
    old_new += [("0.9407", f4(baseline["DSC_mean"])), ("4.8305", f4(baseline["HD95_mean"])), ("0.5378", f4(baseline["ASD_mean"]))]
    old_new += [("0.9571", f4(bl_msl["DSC_mean"])), ("2.9483", f4(bl_msl["HD95_mean"])), ("0.3284", f4(bl_msl["ASD_mean"]))]

    for old, new in old_new:
        if old != new:
            ch3 = ch3.replace(old, new)
            ch4 = ch4.replace(old, new)
            ch5 = ch5.replace(old, new)
    return ch3, ch4, ch5


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply thesis metric template to tex/scripts")
    parser.add_argument("--template", default="thesis-medsam/data/results_update_template.json")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    tpl_path = (repo / args.template).resolve()
    tpl = json.loads(read_text(tpl_path))

    ch3_path = repo / "thesis-medsam/pages/chapter3.tex"
    ch4_path = repo / "thesis-medsam/pages/chapter4.tex"
    ch5_path = repo / "thesis-medsam/pages/chapter5.tex"
    sens_path = repo / "scripts/generate_sensitivity_curves.py"

    ch3 = read_text(ch3_path)
    ch4 = read_text(ch4_path)
    ch5 = read_text(ch5_path)
    sens = read_text(sens_path)

    sens = update_sensitivity_script(sens, tpl)
    ch3 = update_chapter3_tables(ch3, tpl)
    ch4 = update_chapter4_table(ch4, tpl)
    ch5 = update_chapter5_tables(ch5, tpl)
    ch3, ch4, ch5 = update_core_metric_occurrences(ch3, ch4, ch5, tpl)

    write_text(sens_path, sens)
    write_text(ch3_path, ch3)
    write_text(ch4_path, ch4)
    write_text(ch5_path, ch5)

    print("Applied template updates:")
    print(f"  - {sens_path}")
    print(f"  - {ch3_path}")
    print(f"  - {ch4_path}")
    print(f"  - {ch5_path}")


if __name__ == "__main__":
    main()
