#!/usr/bin/env python3
"""Prompt Robustness Evaluation: GT Box Perturbation Experiment

Evaluates model performance under varying levels of box prompt perturbation
to assess robustness. For each perturbation level, the GT bounding box is
expanded (or contracted for negative values) by a fixed number of pixels
on each side.

Perturbation levels tested: 0, 5, 10, 20, 30, 50 pixels

Usage:
  python scripts/eval_prompt_robustness.py \
    --data_root data/npy/CT_Abd \
    --checkpoints \
      Baseline:work_dir/baseline/medsam_model_best.pth \
      BL:work_dir/A3R3/medsam_model_best.pth \
      BL+MSL:work_dir/C3/medsam_model_best.pth \
    --out_dir work_dir/prompt_robustness

Output:
  - CSV per (model, shift) combination
  - Summary JSON with all results
  - Robustness curve plot (PDF)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from os.path import basename, join
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from skimage import transform
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from segment_anything import sam_model_registry
from models.medsam_fss import MedSAMFSS, apply_lora_to_image_encoder, LocalGlobalAdapter
from utils.SurfaceDice import (
    compute_average_surface_distance,
    compute_dice_coefficient,
    compute_robust_hausdorff,
    compute_surface_distances,
)

# Default perturbation levels (pixels)
DEFAULT_SHIFTS = [0, 5, 10, 20, 30, 50]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate prompt robustness under GT box perturbation."
    )
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Directory with *.npz test volumes."
    )
    parser.add_argument(
        "--checkpoints", type=str, nargs="+", required=True,
        help=(
            "Model checkpoints in format NAME:PATH. "
            "Example: Baseline:work_dir/baseline/best.pth BL:work_dir/BL/best.pth"
        ),
    )
    parser.add_argument(
        "--shifts", type=int, nargs="+", default=DEFAULT_SHIFTS,
        help=f"Perturbation levels in pixels (default: {DEFAULT_SHIFTS})."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Inference device."
    )
    parser.add_argument(
        "--sam_base_checkpoint", type=str, default="",
        help="SAM base checkpoint path."
    )
    parser.add_argument(
        "--out_dir", type=str, default="work_dir/prompt_robustness",
        help="Output directory for results."
    )
    # Architecture flags (applied per checkpoint via config)
    parser.add_argument(
        "--model_configs", type=str, nargs="*", default=[],
        help=(
            "Per-model architecture config in format NAME:lora=BOOL:lora_rank=INT:lg=BOOL. "
            "Example: BL+LoRA:lora=true:lora_rank=4:lg=false "
            "BL+MSL:lora=false:lora_rank=4:lg=true"
        ),
    )
    return parser.parse_args()


def resolve_sam_base_checkpoint(user_path: str = "") -> str:
    candidates = []
    if user_path:
        candidates.append(user_path)
    candidates.extend([
        "work_dir/SAM/sam_vit_b_01ec64.pth",
        "work_dir/medsam_vit_b.pth",
        "sam_vit_b_01ec64.pth",
    ])
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "No SAM base checkpoint found. Provide --sam_base_checkpoint."
    )


def load_checkpoint_state(ckpt_path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint
    clean_state = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        clean_state[key] = value
    return clean_state


def load_model(
    ckpt_path: str,
    sam_base_ckpt: str,
    device: torch.device,
    use_lora: bool = False,
    lora_rank: int = 4,
    use_lg_adapter: bool = False,
) -> torch.nn.Module:
    state_dict = load_checkpoint_state(ckpt_path)
    base_model = sam_model_registry["vit_b"](checkpoint=sam_base_ckpt)
    model = MedSAMFSS(
        image_encoder=base_model.image_encoder,
        mask_decoder=base_model.mask_decoder,
        prompt_encoder=base_model.prompt_encoder,
    )
    if use_lora:
        apply_lora_to_image_encoder(model.image_encoder, rank=lora_rank)
    if use_lg_adapter:
        model.local_global_adapter = LocalGlobalAdapter(embed_dim=256)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def compute_image_embedding(
    model: torch.nn.Module, image_2d: np.ndarray, device: torch.device
) -> Tuple[torch.Tensor, int, int]:
    if image_2d.ndim == 2:
        image_3c = np.repeat(image_2d[:, :, None], 3, axis=-1)
    else:
        image_3c = image_2d
    height, width = image_3c.shape[:2]
    image_1024 = transform.resize(
        image_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.float32)
    image_1024 = (image_1024 - image_1024.min()) / np.clip(
        image_1024.max() - image_1024.min(), a_min=1e-8, a_max=None
    )
    image_t = (
        torch.tensor(image_1024, dtype=torch.float32, device=device)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    embedding = model.image_encoder(image_t)
    if hasattr(model, "local_global_adapter") and model.local_global_adapter is not None:
        embedding = model.local_global_adapter(embedding)
    return embedding, height, width


@torch.no_grad()
def infer_single_label_with_box(
    model: torch.nn.Module,
    embedding: torch.Tensor,
    box_xyxy: np.ndarray,
    out_h: int,
    out_w: int,
    device: torch.device,
) -> np.ndarray:
    box_1024 = (
        box_xyxy / np.array([out_w, out_h, out_w, out_h], dtype=np.float32) * 1024.0
    )
    box_t = torch.as_tensor(box_1024, dtype=torch.float32, device=device).view(1, 1, 4)
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=None, boxes=box_t, masks=None
    )
    low_res_logits, _ = model.mask_decoder(
        image_embeddings=embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    low_res_pred = torch.sigmoid(low_res_logits)
    pred = (
        F.interpolate(
            low_res_pred, size=(out_h, out_w), mode="bilinear", align_corners=False
        )[0, 0]
        .cpu()
        .numpy()
    )
    return (pred > 0.5).astype(np.uint8)


def infer_volume_with_shift(
    model: torch.nn.Module,
    imgs: np.ndarray,
    gts: np.ndarray,
    device: torch.device,
    bbox_shift: int,
) -> np.ndarray:
    """Infer volume with fixed bbox expansion/contraction."""
    segs = np.zeros_like(gts, dtype=np.uint8)
    for z in range(imgs.shape[0]):
        gt_2d = gts[z]
        label_ids = np.unique(gt_2d)
        label_ids = label_ids[label_ids > 0]
        if len(label_ids) == 0:
            continue

        embedding, h, w = compute_image_embedding(model, imgs[z], device)

        for label_id in label_ids:
            gt_mask = (gt_2d == label_id).astype(np.uint8)
            ys, xs = np.where(gt_mask > 0)
            if len(xs) == 0:
                continue
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            # Apply perturbation (positive = expand, negative = contract)
            x_min = max(0, x_min - bbox_shift)
            x_max = min(w - 1, x_max + bbox_shift)
            y_min = max(0, y_min - bbox_shift)
            y_max = min(h - 1, y_max + bbox_shift)

            # Ensure valid box
            if x_max <= x_min or y_max <= y_min:
                continue

            box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
            pred_mask = infer_single_label_with_box(model, embedding, box, h, w, device)
            segs[z, pred_mask > 0] = np.uint8(label_id)
    return segs


def spacing_to_zyx(spacing: np.ndarray) -> np.ndarray:
    spacing = np.asarray(spacing).reshape(-1).astype(float)
    if spacing.size >= 3:
        return np.array([spacing[2], spacing[1], spacing[0]], dtype=float)
    return np.array([1.0, 1.0, 1.0], dtype=float)


def compute_case_metrics(
    segs: np.ndarray, gts: np.ndarray, spacing_zyx: np.ndarray
) -> Tuple[float, float, float, int]:
    dices, hd95s, asds = [], [], []
    labels = np.unique(gts)
    labels = labels[labels > 0]
    for label_id in labels:
        gt_bin = gts == label_id
        seg_bin = segs == label_id
        gt_n = int(gt_bin.sum())
        seg_n = int(seg_bin.sum())
        if gt_n == 0 and seg_n == 0:
            continue
        if gt_n == 0 or seg_n == 0:
            dice, hd95, asd = 0.0, 500.0, 500.0
        else:
            dice = float(compute_dice_coefficient(gt_bin, seg_bin))
            surf_dist = compute_surface_distances(gt_bin, seg_bin, spacing_zyx)
            hd95 = float(compute_robust_hausdorff(surf_dist, 95))
            asd_gt, asd_pred = compute_average_surface_distance(surf_dist)
            asd = float((asd_gt + asd_pred) / 2.0)
        dices.append(dice)
        hd95s.append(hd95)
        asds.append(asd)
    if len(dices) == 0:
        return 0.0, 500.0, 500.0, 0
    return float(np.mean(dices)), float(np.mean(hd95s)), float(np.mean(asds)), len(dices)


def parse_model_config(config_str: str) -> dict:
    """Parse NAME:key=val:key=val format."""
    parts = config_str.split(":")
    name = parts[0]
    cfg = {"name": name, "use_lora": False, "lora_rank": 4, "use_lg_adapter": False}
    for part in parts[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            if k == "lora":
                cfg["use_lora"] = v.lower() in ("true", "1", "yes")
            elif k == "lora_rank":
                cfg["lora_rank"] = int(v)
            elif k == "lg":
                cfg["use_lg_adapter"] = v.lower() in ("true", "1", "yes")
    return cfg


def plot_robustness_curves(all_results: dict, shifts: list, out_path: str):
    """Generate robustness curve plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plot generation.")
        return

    plt.rcParams["font.family"] = ["Times New Roman", "SimSun", "serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False

    colors = {"Baseline": "#95A5A6", "BL": "#E74C3C", "BL+MSL": "#27AE60"}
    markers = {"Baseline": "o", "BL": "s", "BL+MSL": "D"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [("DSC", "dice_mean", True), ("HD95 (mm)", "hd95_mean", False),
               ("ASD (mm)", "asd_mean", False)]

    for ax, (label, key, higher_better) in zip(axes, metrics):
        for model_name in all_results:
            vals = [all_results[model_name][s][key] for s in shifts]
            color = colors.get(model_name, "#333333")
            marker = markers.get(model_name, "^")
            ax.plot(shifts, vals, marker=marker, color=color, linewidth=2,
                    markersize=8, label=model_name)
        ax.set_xlabel("Box Prompt 扰动幅度 (像素)", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f"{label} vs 提示扰动", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(shifts)

    fig.suptitle("推理阶段 GT Box Prompt 鲁棒性评估", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved plot: {out_path}")
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    sam_base_ckpt = resolve_sam_base_checkpoint(args.sam_base_checkpoint)

    # Parse checkpoints
    model_specs = []
    for ckpt_str in args.checkpoints:
        if ":" not in ckpt_str:
            raise ValueError(f"Checkpoint must be in NAME:PATH format, got: {ckpt_str}")
        name, path = ckpt_str.split(":", 1)
        model_specs.append({"name": name, "path": path})

    # Parse model configs
    model_configs = {}
    for cfg_str in args.model_configs:
        cfg = parse_model_config(cfg_str)
        model_configs[cfg["name"]] = cfg

    # Load test data
    npz_files = sorted(glob.glob(join(args.data_root, "*.npz")))
    print(f"[data] {len(npz_files)} test volumes found")

    # Run evaluation
    all_results = {}  # {model_name: {shift: {dice_mean, hd95_mean, asd_mean, ...}}}

    for spec in model_specs:
        model_name = spec["name"]
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"Checkpoint: {spec['path']}")
        print(f"{'='*60}")

        cfg = model_configs.get(model_name, {})
        model = load_model(
            ckpt_path=spec["path"],
            sam_base_ckpt=sam_base_ckpt,
            device=device,
            use_lora=cfg.get("use_lora", False),
            lora_rank=cfg.get("lora_rank", 4),
            use_lg_adapter=cfg.get("use_lg_adapter", False),
        )

        all_results[model_name] = {}

        for shift in args.shifts:
            print(f"\n  [shift={shift}px] Evaluating...")
            rows = []
            for npz_path in tqdm(npz_files, desc=f"{model_name} shift={shift}"):
                npz = np.load(npz_path, allow_pickle=True)
                imgs = npz["imgs"]
                gts = npz["gts"]
                spacing = npz["spacing"] if "spacing" in npz else np.array([1.0, 1.0, 1.0])

                segs = infer_volume_with_shift(model, imgs, gts, device, shift)
                spacing_zyx = spacing_to_zyx(spacing)
                dice, hd95, asd, n_labels = compute_case_metrics(segs, gts, spacing_zyx)
                rows.append({
                    "case": basename(npz_path),
                    "dice": dice, "hd95": hd95, "asd": asd,
                    "n_labels": n_labels,
                })

            # Save per-case CSV
            csv_path = join(args.out_dir, f"{model_name}_shift{shift}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["case", "dice", "hd95", "asd", "n_labels"])
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

            dice_arr = np.array([r["dice"] for r in rows])
            hd95_arr = np.array([r["hd95"] for r in rows])
            asd_arr = np.array([r["asd"] for r in rows])
            summary = {
                "dice_mean": float(np.mean(dice_arr)),
                "dice_std": float(np.std(dice_arr)),
                "hd95_mean": float(np.mean(hd95_arr)),
                "hd95_std": float(np.std(hd95_arr)),
                "asd_mean": float(np.mean(asd_arr)),
                "asd_std": float(np.std(asd_arr)),
                "n_cases": len(rows),
            }
            all_results[model_name][shift] = summary
            print(f"  [shift={shift}px] DSC={summary['dice_mean']:.4f}±{summary['dice_std']:.4f} "
                  f"HD95={summary['hd95_mean']:.4f} ASD={summary['asd_mean']:.4f}")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Save full results JSON
    json_path = join(args.out_dir, "prompt_robustness_results.json")
    # Convert int keys to str for JSON
    json_results = {}
    for model_name, shifts_dict in all_results.items():
        json_results[model_name] = {str(k): v for k, v in shifts_dict.items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    print(f"\n[output] Full results: {json_path}")

    # Generate robustness curve plot
    plot_path = join(args.out_dir, "prompt_robustness_curves.pdf")
    plot_robustness_curves(all_results, args.shifts, plot_path)

    # Print summary table
    print(f"\n{'='*80}")
    print("Prompt Robustness Summary")
    print(f"{'='*80}")
    header = f"{'Model':<12}" + "".join(f"{'shift='+str(s)+'px':>16}" for s in args.shifts)
    print(header)
    print("-" * len(header))
    for model_name in all_results:
        line = f"{model_name:<12}"
        for s in args.shifts:
            d = all_results[model_name][s]["dice_mean"]
            line += f"{d:>16.4f}"
        print(line)


if __name__ == "__main__":
    main()

