#!/usr/bin/env python3
"""统一生成论文第 5 章手工挑选可视化图。

目标:
    - 基于现存预测 NPZ 手工挑选合适的 slice
    - 正式生成图 5.1 / 5.2 / 5.3
    - 提供候选列表与单张预览，方便在远程服务器反复挑图

推荐工作流:
    1. 写模板:
       python scripts/generate_chapter5_manual_figures.py write-template \
         --output thesis-medsam/figures/chapter5_manual_config.json

    2. 浏览困难样例候选:
       python scripts/generate_chapter5_manual_figures.py list-failures \
         --config thesis-medsam/figures/chapter5_manual_config.json \
         --organ-id 4 --organ-id 10 --organ-id 12 --limit 80

    3. 单独预览某个 case/slice:
       python scripts/generate_chapter5_manual_figures.py preview \
         --config thesis-medsam/figures/chapter5_manual_config.json \
         --case-name FLARE22_Tr_0001.npz --slice-idx 42 \
         --output thesis-medsam/figures/preview_case_0001_slice_42.pdf

    4. 在配置文件中填好 fig5_1 / fig5_2 / fig5_3 的手工选择后正式导出:
       python scripts/generate_chapter5_manual_figures.py render \
         --config thesis-medsam/figures/chapter5_manual_config.json --figure all
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import logging
import os
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle


plt.rcParams["font.family"] = ["Times New Roman", "SimSun", "serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

ORGAN_COLORS = [
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [128, 0, 0],
    [0, 128, 0],
    [0, 0, 128],
    [128, 128, 0],
    [128, 0, 128],
    [0, 128, 128],
    [64, 64, 255],
]

ORGAN_MAP = {
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

DEFAULT_COLUMNS = ["GT", "A0", "A3R3", "C3"]
DEFAULT_TITLES = {
    "GT": "GT",
    "A0": "Baseline",
    "A3R3": "BL",
    "C3": "BL+MSL",
}


class CaseIndex:
    def __init__(self, data_root, pred_dirs, case_names, data_paths, pred_paths):
        self.data_root = data_root
        self.pred_dirs = pred_dirs
        self.case_names = case_names
        self.data_paths = data_paths
        self.pred_paths = pred_paths


def shorten_case_list(case_names: list[str], limit: int = 12) -> str:
    if not case_names:
        return "(no cases)"
    shown = case_names[:limit]
    suffix = "" if len(case_names) <= limit else f" ... ({len(case_names) - limit} more)"
    return ", ".join(shown) + suffix


def ensure_parent_dir(path_str: str) -> None:
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def load_json(path_str: str) -> dict:
    with open(path_str, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_template_config(output_path: str) -> None:
    template = {
        "data_root": "data/npy/CT_Abd",
        "pred_dirs": {
            "A0": "work_dir/eval_predictions/A0",
            "A3R3": "work_dir/eval_predictions/A3R3",
            "C3": "work_dir/eval_predictions/C3",
        },
        "titles": DEFAULT_TITLES,
        "fig5_1": {
            "case_name": "FLARE22_Tr_0001.npz",
            "slice_idx": 42,
            "columns": DEFAULT_COLUMNS,
            "show_slice_dice": True,
            "output_path": "thesis-medsam/figures/mask_comparison.pdf",
        },
        "fig5_2": {
            "case_name": "FLARE22_Tr_0001.npz",
            "slice_idx": 42,
            "columns": DEFAULT_COLUMNS,
            "roi": [160, 288, 160, 288],
            "roi_size": 128,
            "output_path": "thesis-medsam/figures/boundary_detail.pdf",
        },
        "fig5_3": {
            "columns": DEFAULT_COLUMNS,
            "output_path": "thesis-medsam/figures/failure_cases.pdf",
            "rows": [
                {
                    "case_name": "FLARE22_Tr_0001.npz",
                    "slice_idx": 42,
                    "organ_id": 4,
                    "row_label": "(a) Weak boundary\n(Pancreas)",
                },
                {
                    "case_name": "FLARE22_Tr_0005.npz",
                    "slice_idx": 88,
                    "organ_id": 10,
                    "row_label": "(b) Small organ\n(Esophagus)",
                },
                {
                    "case_name": "FLARE22_Tr_0010.npz",
                    "slice_idx": 55,
                    "organ_id": 12,
                    "row_label": "(c) Adjacent organs\n(Duodenum)",
                },
            ],
        },
    }
    ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(template, fh, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote template config: {output_path}")


def build_case_index(data_root: str | None, pred_dirs: dict[str, str]) -> CaseIndex:
    pred_paths = {}
    pred_case_sets = []
    for exp_name, pred_dir in pred_dirs.items():
        files = sorted(glob.glob(os.path.join(pred_dir, "*.npz")))
        if not files:
            raise FileNotFoundError(f"prediction directory is empty: {pred_dir}")
        pred_paths[exp_name] = {os.path.basename(path): path for path in files}
        pred_case_sets.append(set(pred_paths[exp_name].keys()))

    common_names = set.intersection(*pred_case_sets)

    data_paths = {}
    if data_root:
        data_files = sorted(glob.glob(os.path.join(data_root, "*.npz")))
        if data_files:
            data_paths = {os.path.basename(path): path for path in data_files}
            common_names &= set(data_paths.keys())

    case_names = sorted(common_names)
    if not case_names:
        raise RuntimeError("no common cases were found across data/prediction directories")

    return CaseIndex(data_root, pred_dirs, case_names, data_paths, pred_paths)


def resolve_case_name(index: CaseIndex, case_query: str) -> str:
    if case_query in index.case_names:
        return case_query

    query = case_query.lower()
    query_stem = Path(case_query).stem.lower()

    stem_matches = [name for name in index.case_names if Path(name).stem.lower() == query_stem]
    if len(stem_matches) == 1:
        return stem_matches[0]

    substring_matches = [name for name in index.case_names if query in name.lower()]
    if len(substring_matches) == 1:
        return substring_matches[0]

    if len(stem_matches) > 1 or len(substring_matches) > 1:
        ambiguous = stem_matches if len(stem_matches) > 1 else substring_matches
        raise KeyError(
            "case query is ambiguous: "
            f"{case_query}. Matches: {shorten_case_list(sorted(ambiguous), limit=20)}"
        )

    raise KeyError(
        "case not found: "
        f"{case_query}. Available cases include: {shorten_case_list(index.case_names, limit=20)}"
    )


def load_npz(path_str: str) -> dict[str, np.ndarray]:
    data = np.load(path_str, allow_pickle=True)
    return {key: data[key] for key in data.files}


def first_present(arrays: dict[str, np.ndarray], names: list[str]) -> np.ndarray | None:
    for name in names:
        if name in arrays:
            return arrays[name]
    return None


def get_reference_arrays(index: CaseIndex, case_name: str) -> dict[str, np.ndarray]:
    case_name = resolve_case_name(index, case_name)

    if case_name in index.data_paths:
        arrays = load_npz(index.data_paths[case_name])
        if first_present(arrays, ["imgs", "img", "image"]) is not None and first_present(
            arrays, ["gts", "gt", "mask"]
        ) is not None:
            return arrays

    for exp_name in ("A3R3", "A0", "C3"):
        if exp_name in index.pred_paths:
            arrays = load_npz(index.pred_paths[exp_name][case_name])
            if first_present(arrays, ["gts", "gt", "mask"]) is not None:
                return arrays

    raise RuntimeError(f"could not locate reference arrays for case: {case_name}")


def get_prediction_arrays(index: CaseIndex, case_name: str, exp_name: str) -> dict[str, np.ndarray]:
    case_name = resolve_case_name(index, case_name)
    return load_npz(index.pred_paths[exp_name][case_name])


def get_volume_slice(volume: np.ndarray | None, slice_idx: int) -> np.ndarray | None:
    if volume is None:
        return None
    arr = np.asarray(volume)
    if arr.ndim == 2:
        return arr
    if arr.ndim < 2:
        raise ValueError(f"unsupported array ndim: {arr.ndim}")
    if slice_idx < 0 or slice_idx >= arr.shape[0]:
        raise IndexError(f"slice_idx {slice_idx} out of range [0, {arr.shape[0] - 1}]")
    slice_arr = arr[slice_idx]
    if slice_arr.ndim == 3:
        slice_arr = slice_arr[..., 0]
    return np.asarray(slice_arr)


def normalize_ct(ct_slice: np.ndarray | None) -> np.ndarray | None:
    if ct_slice is None:
        return None
    arr = ct_slice.astype(np.float32)
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    return arr


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label_id, color in enumerate(ORGAN_COLORS):
        rgb[mask == label_id] = color
    return rgb


def compute_slice_dice(pred: np.ndarray, gt: np.ndarray, label_id: int | None = None) -> float:
    if label_id is not None:
        pred_bin = pred == label_id
        gt_bin = gt == label_id
    else:
        pred_bin = pred > 0
        gt_bin = gt > 0
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum()
    if union == 0:
        return 1.0
    return float(2 * inter / union)


def compute_diff_circle(gt_bin: np.ndarray, pred_bin: np.ndarray) -> tuple[float, float, float] | None:
    diff = np.logical_xor(gt_bin > 0, pred_bin > 0)
    if diff.sum() <= 10:
        return None
    ys, xs = np.where(diff)
    cy = float(ys.mean())
    cx = float(xs.mean())
    radius = max(8.0, float(max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 + 4))
    return cx, cy, radius


def get_organ_bbox(mask: np.ndarray, organ_id: int, pad: int = 40) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask == organ_id)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(h, int(ys.max()) + pad)
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(w, int(xs.max()) + pad)
    return y1, y2, x1, x2


def find_boundary_roi(mask: np.ndarray, roi_size: int = 128) -> tuple[int, int, int, int]:
    mask = np.asarray(mask)
    h, w = mask.shape
    if roi_size >= min(h, w):
        return 0, h, 0, w

    dy = np.abs(np.diff(mask, axis=0))
    dx = np.abs(np.diff(mask, axis=1))

    boundary = np.zeros(mask.shape, dtype=np.int32)
    boundary[:-1, :] += (dy > 0).astype(np.int32)
    boundary[:, :-1] += (dx > 0).astype(np.int32)

    best_score = -1
    best_y = 0
    best_x = 0
    step = max(1, roi_size // 4)

    for y in range(0, h - roi_size + 1, step):
        for x in range(0, w - roi_size + 1, step):
            score = int(boundary[y : y + roi_size, x : x + roi_size].sum())
            if score > best_score:
                best_score = score
                best_y = y
                best_x = x
    return best_y, best_y + roi_size, best_x, best_x + roi_size


def resolve_roi(selection: dict, fallback_mask: np.ndarray) -> tuple[int, int, int, int]:
    roi = selection.get("roi")
    if roi is not None:
        if len(roi) != 4:
            raise ValueError("roi must be [y1, y2, x1, x2]")
        y1, y2, x1, x2 = [int(v) for v in roi]
        return y1, y2, x1, x2

    roi_size = int(selection.get("roi_size", 128))
    return find_boundary_roi(fallback_mask, roi_size=roi_size)


def crop_image(image: np.ndarray | None, bbox: tuple[int, int, int, int] | None) -> np.ndarray | None:
    if image is None or bbox is None:
        return image
    y1, y2, x1, x2 = bbox
    return image[y1:y2, x1:x2]


def add_contours(ax, gt_bin: np.ndarray, pred_bin: np.ndarray) -> None:
    if gt_bin.max() > 0:
        ax.contour(gt_bin.astype(float), levels=[0.5], colors=["#00FF00"], linewidths=1.3)
    if pred_bin.max() > 0:
        ax.contour(pred_bin.astype(float), levels=[0.5], colors=["#FF3333"], linewidths=1.3)


def render_contour_panel(
    ax,
    ct_slice: np.ndarray | None,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    organ_id: int,
    title: str = "",
    crop_bbox: tuple[int, int, int, int] | None = None,
) -> None:
    ct_crop = crop_image(ct_slice, crop_bbox)
    gt_crop = crop_image(gt_mask, crop_bbox)
    pred_crop = crop_image(pred_mask, crop_bbox)
    gt_bin = (gt_crop == organ_id).astype(np.uint8)
    pred_bin = (pred_crop == organ_id).astype(np.uint8)

    if ct_crop is not None:
        ax.imshow(ct_crop, cmap="gray", vmin=0, vmax=1, aspect="equal")
        add_contours(ax, gt_bin, pred_bin)
    else:
        panel_mask = gt_crop if np.array_equal(gt_crop, pred_crop) else pred_crop
        ax.imshow(colorize_mask(panel_mask), interpolation="nearest")
        circle = compute_diff_circle(gt_bin, pred_bin)
        if circle is not None:
            cx, cy, radius = circle
            ax.add_patch(
                Circle((cx, cy), radius, fill=False, edgecolor="yellow", linewidth=1.5, linestyle="--")
            )

    circle = compute_diff_circle(gt_bin, pred_bin)
    if circle is not None:
        cx, cy, radius = circle
        ax.add_patch(
            Circle((cx, cy), radius, fill=False, edgecolor="yellow", linewidth=1.5, linestyle="--")
        )

    if title:
        ax.set_title(title, fontsize=10, pad=3)
    ax.axis("off")


def resolve_titles(selection: dict, global_titles: dict | None = None) -> dict:
    titles = dict(DEFAULT_TITLES)
    if global_titles:
        titles.update(global_titles)
    if "titles" in selection:
        titles.update(selection["titles"])
    return titles


def validate_case_and_slice(index: CaseIndex, case_name: str, slice_idx: int) -> None:
    case_name = resolve_case_name(index, case_name)
    ref = get_reference_arrays(index, case_name)
    gts = first_present(ref, ["gts", "gt", "mask"])
    if gts is None:
        raise RuntimeError(f"gts missing for case: {case_name}")
    if slice_idx < 0 or slice_idx >= gts.shape[0]:
        raise IndexError(f"slice_idx {slice_idx} out of range for case {case_name}")


def load_case_slices(index: CaseIndex, case_name: str, slice_idx: int) -> dict[str, np.ndarray | None]:
    case_name = resolve_case_name(index, case_name)
    validate_case_and_slice(index, case_name, slice_idx)
    ref_arrays = get_reference_arrays(index, case_name)
    imgs = first_present(ref_arrays, ["imgs", "img", "image"])
    gts = first_present(ref_arrays, ["gts", "gt", "mask"])
    ct_slice = normalize_ct(get_volume_slice(imgs, slice_idx))
    gt_slice = get_volume_slice(gts, slice_idx)

    slices = {"ct": ct_slice, "GT": gt_slice}
    for exp_name in index.pred_dirs:
        pred_arrays = get_prediction_arrays(index, case_name, exp_name)
        pred_volume = first_present(pred_arrays, ["segs", "seg", "pred", "mask"])
        if pred_volume is None:
            raise RuntimeError(f"prediction volume missing for {exp_name}/{case_name}")
        slices[exp_name] = get_volume_slice(pred_volume, slice_idx)
    return slices


def list_cases(index: CaseIndex, contains: str | None = None, limit: int = 100) -> list[str]:
    case_names = index.case_names
    if contains:
        needle = contains.lower()
        case_names = [name for name in case_names if needle in name.lower()]
    return case_names[:limit]


def print_cases(case_names: list[str]) -> None:
    if not case_names:
        print("[WARN] no cases matched the filter")
        return
    for idx, name in enumerate(case_names):
        print(f"{idx:4d}  {name}")


def make_archive(source_dir: str, archive_path: str) -> str:
    ensure_parent_dir(archive_path)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source_dir, arcname=Path(source_dir).name)
    return archive_path


def render_mask_comparison(index: CaseIndex, selection: dict, global_titles: dict | None = None) -> str:
    case_name = resolve_case_name(index, selection["case_name"])
    slice_idx = int(selection["slice_idx"])
    output_path = selection["output_path"]
    dpi = int(selection.get("dpi", 300))
    width_per_col = float(selection.get("width_per_col", 3.6))
    fig_height = float(selection.get("fig_height", 3.8))
    columns = selection.get("columns", DEFAULT_COLUMNS)
    titles = resolve_titles(selection, global_titles)
    show_slice_dice = bool(selection.get("show_slice_dice", False))

    slices = load_case_slices(index, case_name, slice_idx)
    gt_slice = slices["GT"]

    fig, axes = plt.subplots(1, len(columns), figsize=(width_per_col * len(columns), fig_height))
    if len(columns) == 1:
        axes = [axes]

    for ax, col_key in zip(axes, columns):
        mask = slices[col_key]
        title = titles.get(col_key, col_key)
        if col_key != "GT" and show_slice_dice:
            dice = compute_slice_dice(mask, gt_slice)
            title = f"{title}\nDice={dice:.3f}"
        ax.imshow(colorize_mask(mask), interpolation="nearest")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    fig.suptitle("Prediction Mask Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved fig5.1 candidate/final: {output_path}")
    return output_path


def render_boundary_detail(index: CaseIndex, selection: dict, global_titles: dict | None = None) -> str:
    case_name = resolve_case_name(index, selection["case_name"])
    slice_idx = int(selection["slice_idx"])
    output_path = selection["output_path"]
    dpi = int(selection.get("dpi", 300))
    width_per_col = float(selection.get("width_per_col", 3.6))
    fig_height = float(selection.get("fig_height", 7.2))
    columns = selection.get("columns", DEFAULT_COLUMNS)
    titles = resolve_titles(selection, global_titles)

    slices = load_case_slices(index, case_name, slice_idx)
    fallback_key = "A3R3" if "A3R3" in columns else columns[0]
    y1, y2, x1, x2 = resolve_roi(selection, slices[fallback_key])

    fig, axes = plt.subplots(2, len(columns), figsize=(width_per_col * len(columns), fig_height))
    if len(columns) == 1:
        axes = np.asarray([[axes[0]], [axes[1]]])

    for col_idx, col_key in enumerate(columns):
        mask = slices[col_key]
        mask_rgb = colorize_mask(mask)

        axes[0, col_idx].imshow(mask_rgb, interpolation="nearest")
        axes[0, col_idx].add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="yellow", facecolor="none", linestyle="--")
        )
        axes[0, col_idx].set_title(titles.get(col_key, col_key), fontsize=11, fontweight="bold")
        axes[0, col_idx].axis("off")

        axes[1, col_idx].imshow(mask_rgb[y1:y2, x1:x2], interpolation="nearest")
        axes[1, col_idx].set_title(f"{titles.get(col_key, col_key)} (zoom)", fontsize=10)
        axes[1, col_idx].axis("off")

    fig.suptitle("Boundary Detail Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved fig5.2 candidate/final: {output_path}")
    return output_path


def render_failure_cases(index: CaseIndex, selection: dict, global_titles: dict | None = None) -> str:
    output_path = selection["output_path"]
    dpi = int(selection.get("dpi", 300))
    rows = selection["rows"]
    columns = selection.get("columns", DEFAULT_COLUMNS)
    titles = resolve_titles(selection, global_titles)
    width_per_col = float(selection.get("width_per_col", 2.8))
    height_per_row = float(selection.get("height_per_row", 2.6))
    extra_height = float(selection.get("extra_height", 0.8))

    fig, axes = plt.subplots(
        len(rows),
        len(columns),
        figsize=(width_per_col * len(columns), height_per_row * len(rows) + extra_height),
    )
    if len(rows) == 1:
        axes = np.asarray([axes])
    if len(columns) == 1:
        axes = axes.reshape(len(rows), 1)

    for row_idx, row_cfg in enumerate(rows):
        case_name = resolve_case_name(index, row_cfg["case_name"])
        slice_idx = int(row_cfg["slice_idx"])
        organ_id = int(row_cfg["organ_id"])
        row_label = row_cfg["row_label"]

        slices = load_case_slices(index, case_name, slice_idx)
        crop_bbox = get_organ_bbox(slices["GT"], organ_id, pad=int(row_cfg.get("crop_pad", 50)))
        if crop_bbox is None:
            raise RuntimeError(
                f"organ {organ_id} not found on case={case_name}, slice={slice_idx}"
            )

        for col_idx, col_key in enumerate(columns):
            pred_mask = slices["GT"] if col_key == "GT" else slices[col_key]
            title = titles.get(col_key, col_key) if row_idx == 0 else ""
            render_contour_panel(
                axes[row_idx, col_idx],
                slices["ct"],
                slices["GT"],
                pred_mask,
                organ_id,
                title=title,
                crop_bbox=crop_bbox,
            )

        axes[row_idx, 0].set_ylabel(row_label, fontsize=9, rotation=0, labelpad=70, va="center")

    legend = [
        Line2D([0], [0], color="#00FF00", linewidth=2, label="GT contour"),
        Line2D([0], [0], color="#FF3333", linewidth=2, label="Prediction contour"),
        Line2D([0], [0], color="yellow", linewidth=2, linestyle="--", label="Key difference region"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0.08, 0.04, 1.0, 1.0])
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved fig5.3 candidate/final: {output_path}")
    return output_path


def list_failure_candidates(
    index: CaseIndex,
    organ_ids: list[int],
    dice_lower: float = 0.30,
    dice_upper: float = 0.92,
    min_area: int = 50,
    limit: int = 100,
    reference_experiment: str = "C3",
) -> list[dict]:
    records = []
    for case_name in index.case_names:
        ref = get_reference_arrays(index, case_name)
        gt_volume = first_present(ref, ["gts", "gt", "mask"])
        pred_arrays = get_prediction_arrays(index, case_name, reference_experiment)
        pred_volume = first_present(pred_arrays, ["segs", "seg", "pred", "mask"])

        for slice_idx in range(gt_volume.shape[0]):
            gt_slice = get_volume_slice(gt_volume, slice_idx)
            pred_slice = get_volume_slice(pred_volume, slice_idx)
            for organ_id in organ_ids:
                area = int((gt_slice == organ_id).sum())
                if area < min_area:
                    continue
                dice = compute_slice_dice(pred_slice, gt_slice, label_id=organ_id)
                if dice_lower <= dice < dice_upper:
                    records.append(
                        {
                            "case_name": case_name,
                            "slice_idx": slice_idx,
                            "organ_id": organ_id,
                            "organ_name": ORGAN_MAP.get(organ_id, str(organ_id)),
                            "dice": dice,
                            "area": area,
                        }
                    )

    records.sort(key=lambda item: item["dice"])
    return records[:limit]


def print_failure_candidates(records: list[dict]) -> None:
    if not records:
        print("[WARN] no candidates matched the filter")
        return
    print(f"{'Organ':<14}{'Case':<30}{'Slice':>8}{'ID':>6}{'Dice':>10}{'Area':>8}")
    print("-" * 80)
    for item in records:
        print(
            f"{item['organ_name']:<14}{item['case_name']:<30}"
            f"{item['slice_idx']:>8}{item['organ_id']:>6}"
            f"{item['dice']:>10.4f}{item['area']:>8}"
        )


def generate_preview_bundle(
    config: dict,
    output_root: str,
    slice_stride: int = 1,
    failure_limit: int = 200,
    organ_ids: list[int] | None = None,
    reference_experiment: str = "C3",
    preview_dpi: int = 500,
    mask_width_per_col: float = 4.8,
    failure_width_per_col: float = 4.0,
    failure_height_per_row: float = 3.8,
    archive: bool = False,
) -> dict:
    organ_ids = organ_ids or [4, 7, 8, 10, 12]
    index = build_case_index(config.get("data_root"), config["pred_dirs"])
    titles = config.get("titles", DEFAULT_TITLES)

    out_root = Path(output_root)
    mask_dir = out_root / "mask_previews"
    fail_dir = out_root / "failure_previews"
    mask_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)

    mask_count = 0
    for case_name in index.case_names:
        ref = get_reference_arrays(index, case_name)
        gt_volume = first_present(ref, ["gts", "gt", "mask"])
        if gt_volume is None:
            continue

        for slice_idx in range(0, gt_volume.shape[0], max(1, int(slice_stride))):
            gt_slice = get_volume_slice(gt_volume, slice_idx)
            if gt_slice is None or np.max(gt_slice) == 0:
                continue

            selection = {
                "case_name": case_name,
                "slice_idx": slice_idx,
                "output_path": str(mask_dir / f"{Path(case_name).stem}_slice{slice_idx:03d}.png"),
                "columns": DEFAULT_COLUMNS,
                "titles": titles,
                "show_slice_dice": True,
                "dpi": preview_dpi,
                "width_per_col": mask_width_per_col,
                "fig_height": 4.6,
            }
            with contextlib.redirect_stdout(io.StringIO()):
                render_mask_comparison(index, selection, titles)
            mask_count += 1

    records = list_failure_candidates(
        index,
        organ_ids=organ_ids,
        dice_lower=0.0,
        dice_upper=1.0,
        min_area=5,
        limit=failure_limit,
        reference_experiment=reference_experiment,
    )

    candidates_path = out_root / "failure_candidates.txt"
    with open(candidates_path, "w", encoding="utf-8") as fh:
        fh.write(f"{'rank':<6}{'organ':<14}{'case':<32}{'slice':>8}{'id':>6}{'dice':>10}{'area':>8}\n")
        fh.write("-" * 90 + "\n")
        for idx, item in enumerate(records):
            fh.write(
                f"{idx:<6}{item['organ_name']:<14}{item['case_name']:<32}"
                f"{item['slice_idx']:>8}{item['organ_id']:>6}"
                f"{item['dice']:>10.4f}{item['area']:>8}\n"
            )

    failure_count = 0
    for idx, item in enumerate(records):
        organ_name = item["organ_name"].replace(".", "")
        selection = {
            "output_path": str(
                fail_dir
                / (
                    f"{idx:03d}_{Path(item['case_name']).stem}_slice{item['slice_idx']:03d}_"
                    f"organ{item['organ_id']}_{organ_name}.png"
                )
            ),
            "columns": DEFAULT_COLUMNS,
            "titles": titles,
            "dpi": preview_dpi,
            "width_per_col": failure_width_per_col,
            "height_per_row": failure_height_per_row,
            "extra_height": 0.9,
            "rows": [
                {
                    "case_name": item["case_name"],
                    "slice_idx": item["slice_idx"],
                    "organ_id": item["organ_id"],
                    "row_label": f"{item['organ_name']}\nDice={item['dice']:.3f}",
                }
            ],
        }
        with contextlib.redirect_stdout(io.StringIO()):
            render_failure_cases(index, selection, titles)
        failure_count += 1

    archive_path = None
    if archive:
        archive_path = str(make_archive(str(out_root), str(out_root) + ".tar.gz"))

    return {
        "output_root": str(out_root),
        "mask_count": mask_count,
        "failure_count": failure_count,
        "candidates_path": str(candidates_path),
        "archive_path": archive_path,
    }


def render_from_config(config: dict, figure_key: str) -> None:
    index = build_case_index(config.get("data_root"), config["pred_dirs"])
    global_titles = config.get("titles", DEFAULT_TITLES)

    renderers = {
        "fig5_1": render_mask_comparison,
        "fig5_2": render_boundary_detail,
        "fig5_3": render_failure_cases,
    }

    if figure_key == "all":
        for key in ("fig5_1", "fig5_2", "fig5_3"):
            renderers[key](index, config[key], global_titles)
        return

    renderers[figure_key](index, config[figure_key], global_titles)


def preview_case(index: CaseIndex, case_name: str, slice_idx: int, output_path: str) -> None:
    case_name = resolve_case_name(index, case_name)
    render_mask_comparison(
        index,
        {
            "case_name": case_name,
            "slice_idx": slice_idx,
            "output_path": output_path,
            "columns": DEFAULT_COLUMNS,
            "titles": DEFAULT_TITLES,
            "show_slice_dice": True,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual generator for thesis chapter-5 figures")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tpl = subparsers.add_parser("write-template", help="write a template JSON config")
    tpl.add_argument("--output", required=True, help="path to the JSON template")

    lst = subparsers.add_parser("list-failures", help="list low-dice candidate slices")
    lst.add_argument("--config", required=True, help="path to config JSON")
    lst.add_argument("--organ-id", type=int, action="append", required=True, help="organ id, repeatable")
    lst.add_argument("--dice-lower", type=float, default=0.30)
    lst.add_argument("--dice-upper", type=float, default=0.92)
    lst.add_argument("--min-area", type=int, default=50)
    lst.add_argument("--limit", type=int, default=100)
    lst.add_argument("--reference-experiment", default="C3")

    lcs = subparsers.add_parser("list-cases", help="list available case names")
    lcs.add_argument("--config", required=True, help="path to config JSON")
    lcs.add_argument("--contains", default=None, help="optional substring filter")
    lcs.add_argument("--limit", type=int, default=100)

    bnd = subparsers.add_parser("bundle-previews", help="generate all preview images and optional tar.gz bundle")
    bnd.add_argument("--config", required=True, help="path to config JSON")
    bnd.add_argument("--output-root", required=True, help="directory to store generated previews")
    bnd.add_argument("--slice-stride", type=int, default=1, help="sample every Nth slice for mask previews")
    bnd.add_argument("--failure-limit", type=int, default=200, help="max failure preview rows to render")
    bnd.add_argument("--organ-id", type=int, action="append", dest="organ_ids", help="organ ids for failure search")
    bnd.add_argument("--reference-experiment", default="C3", help="which experiment to rank failure candidates by")
    bnd.add_argument("--preview-dpi", type=int, default=500, help="dpi for preview images")
    bnd.add_argument("--mask-width-per-col", type=float, default=4.8, help="mask preview figure width per column")
    bnd.add_argument("--failure-width-per-col", type=float, default=4.0, help="failure preview figure width per column")
    bnd.add_argument("--failure-height-per-row", type=float, default=3.8, help="failure preview figure height per row")
    bnd.add_argument("--archive", action="store_true", help="also create output-root.tar.gz")

    prv = subparsers.add_parser("preview", help="render a one-off preview for a chosen case/slice")
    prv.add_argument("--config", required=True, help="path to config JSON")
    prv.add_argument("--case-name", required=True)
    prv.add_argument("--slice-idx", required=True, type=int)
    prv.add_argument("--output", required=True)

    rnd = subparsers.add_parser("render", help="render fig5.1/5.2/5.3 from config")
    rnd.add_argument("--config", required=True, help="path to config JSON")
    rnd.add_argument(
        "--figure",
        choices=["all", "fig5_1", "fig5_2", "fig5_3"],
        default="all",
        help="which figure to render",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "write-template":
        write_template_config(args.output)
        return

    config = load_json(args.config)
    index = build_case_index(config.get("data_root"), config["pred_dirs"])

    if args.command == "list-failures":
        records = list_failure_candidates(
            index,
            organ_ids=args.organ_id,
            dice_lower=args.dice_lower,
            dice_upper=args.dice_upper,
            min_area=args.min_area,
            limit=args.limit,
            reference_experiment=args.reference_experiment,
        )
        print_failure_candidates(records)
        return

    if args.command == "list-cases":
        print_cases(list_cases(index, contains=args.contains, limit=args.limit))
        return

    if args.command == "bundle-previews":
        info = generate_preview_bundle(
            config,
            output_root=args.output_root,
            slice_stride=args.slice_stride,
            failure_limit=args.failure_limit,
            organ_ids=args.organ_ids,
            reference_experiment=args.reference_experiment,
            preview_dpi=args.preview_dpi,
            mask_width_per_col=args.mask_width_per_col,
            failure_width_per_col=args.failure_width_per_col,
            failure_height_per_row=args.failure_height_per_row,
            archive=args.archive,
        )
        print(f"[OK] output_root = {info['output_root']}")
        print(f"[OK] mask_previews = {info['mask_count']}")
        print(f"[OK] failure_previews = {info['failure_count']}")
        print(f"[OK] failure_candidates = {info['candidates_path']}")
        if info["archive_path"]:
            print(f"[OK] archive = {info['archive_path']}")
        return

    if args.command == "preview":
        preview_case(index, args.case_name, args.slice_idx, args.output)
        return

    if args.command == "render":
        render_from_config(config, args.figure)
        return


if __name__ == "__main__":
    main()

