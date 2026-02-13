#!/usr/bin/env python
"""Evaluate MedSAM checkpoints on NPZ volumes and report DSC/HD95/ASD.

Expected NPZ format:
  - imgs: (Z, H, W) or image-like slices
  - gts: (Z, H, W), integer labels
  - spacing: optional, usually (x, y, z)

Example:
  python eval_medsam_npz.py \
    --data_root data/npy/CT_Abd \
    --checkpoint work_dir/xxx/medsam_model_best.pth \
    --exp_name A1 \
    --out_csv work_dir/eval_metrics/A1_case_metrics.csv \
    --out_json work_dir/eval_metrics/A1_summary.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from os.path import basename
from os.path import join
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from skimage import transform
from tqdm import tqdm

from segment_anything import sam_model_registry
from utils.SurfaceDice import compute_average_surface_distance
from utils.SurfaceDice import compute_dice_coefficient
from utils.SurfaceDice import compute_robust_hausdorff
from utils.SurfaceDice import compute_surface_distances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MedSAM checkpoints on NPZ volumes."
    )
    parser.add_argument("--data_root", type=str, required=True, help="Directory with *.npz volumes.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to medsam_model_best.pth.")
    parser.add_argument(
        "--sam_base_checkpoint",
        type=str,
        default="",
        help=(
            "Optional SAM base checkpoint for model init. "
            "If empty, script will auto search common paths."
        ),
    )
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name (A1/A2/A3...).")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device, e.g. cuda:0.")
    parser.add_argument("--bbox_shift", type=int, default=20, help="BBox expansion in pixels.")
    parser.add_argument("--max_cases", type=int, default=0, help="Use first N cases only (0 = all).")
    parser.add_argument("--out_csv", type=str, required=True, help="Per-case metrics csv path.")
    parser.add_argument("--out_json", type=str, required=True, help="Summary json path.")
    parser.add_argument(
        "--save_pred_dir",
        type=str,
        default="",
        help="Optional directory to save predicted NPZ files (segs/gts/spacing).",
    )
    return parser.parse_args()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_checkpoint_state(ckpt_path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint

    clean_state = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        clean_state[key] = value
    return clean_state


def resolve_sam_base_checkpoint(user_path: str = "") -> str:
    candidates: List[str] = []
    if user_path:
        candidates.append(user_path)
    candidates.extend(
        [
            "work_dir/SAM/sam_vit_b_01ec64.pth",
            "work_dir/medsam_vit_b.pth",
            "sam_vit_b_01ec64.pth",
        ]
    )
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "No SAM base checkpoint found. Provide --sam_base_checkpoint, "
        "or place checkpoint at one of: "
        "work_dir/SAM/sam_vit_b_01ec64.pth, work_dir/medsam_vit_b.pth, sam_vit_b_01ec64.pth"
    )


def load_medsam_model(ckpt_path: str, sam_base_ckpt: str, device: torch.device) -> torch.nn.Module:
    state_dict = load_checkpoint_state(ckpt_path)
    model = sam_model_registry["vit_b"](checkpoint=sam_base_ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[load] base checkpoint: {sam_base_ckpt}")
    print(f"[load] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
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
    box_1024 = box_xyxy / np.array([out_w, out_h, out_w, out_h], dtype=np.float32) * 1024.0
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
    pred = F.interpolate(
        low_res_pred, size=(out_h, out_w), mode="bilinear", align_corners=False
    )[0, 0].cpu().numpy()
    return (pred > 0.5).astype(np.uint8)


def infer_volume(
    model: torch.nn.Module,
    imgs: np.ndarray,
    gts: np.ndarray,
    device: torch.device,
    bbox_shift: int,
) -> np.ndarray:
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

            x_min = max(0, x_min - bbox_shift)
            x_max = min(w - 1, x_max + bbox_shift)
            y_min = max(0, y_min - bbox_shift)
            y_max = min(h - 1, y_max + bbox_shift)
            box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

            pred_mask = infer_single_label_with_box(model, embedding, box, h, w, device)
            segs[z, pred_mask > 0] = np.uint8(label_id)
    return segs


def spacing_to_zyx(spacing: np.ndarray) -> np.ndarray:
    spacing = np.asarray(spacing).reshape(-1).astype(float)
    if spacing.size >= 3:
        # SITK spacing is usually (x, y, z), while mask axes are (z, y, x).
        return np.array([spacing[2], spacing[1], spacing[0]], dtype=float)
    return np.array([1.0, 1.0, 1.0], dtype=float)


def compute_case_metrics(
    segs: np.ndarray, gts: np.ndarray, spacing_zyx: np.ndarray
) -> Tuple[float, float, float, int]:
    dices: List[float] = []
    hd95s: List[float] = []
    asds: List[float] = []

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
            dice = 0.0
            hd95 = 500.0
            asd = 500.0
        else:
            dice = float(compute_dice_coefficient(gt_bin, seg_bin))
            surface_distances = compute_surface_distances(gt_bin, seg_bin, spacing_zyx)
            hd95 = float(compute_robust_hausdorff(surface_distances, 95))
            asd_gt, asd_pred = compute_average_surface_distance(surface_distances)
            asd = float((asd_gt + asd_pred) / 2.0)

        dices.append(dice)
        hd95s.append(hd95)
        asds.append(asd)

    if len(dices) == 0:
        return 0.0, 500.0, 500.0, 0
    return float(np.mean(dices)), float(np.mean(hd95s)), float(np.mean(asds)), len(dices)


def dump_case_csv(rows: Iterable[Dict[str, object]], out_csv: str) -> None:
    ensure_parent_dir(out_csv)
    fieldnames = ["case", "dice", "hd95", "asd", "n_labels"]
    with open(out_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    ensure_parent_dir(args.out_csv)
    ensure_parent_dir(args.out_json)
    if args.save_pred_dir:
        os.makedirs(args.save_pred_dir, exist_ok=True)

    device = torch.device(args.device)
    sam_base_ckpt = resolve_sam_base_checkpoint(args.sam_base_checkpoint)
    model = load_medsam_model(args.checkpoint, sam_base_ckpt, device)

    npz_files = sorted(glob.glob(join(args.data_root, "*.npz")))
    if args.max_cases > 0:
        npz_files = npz_files[: args.max_cases]
    print(f"[data] cases: {len(npz_files)}")

    rows: List[Dict[str, object]] = []
    for npz_path in tqdm(npz_files, desc=args.exp_name):
        npz = np.load(npz_path, allow_pickle=True)
        imgs = npz["imgs"]
        gts = npz["gts"]
        spacing = npz["spacing"] if "spacing" in npz else np.array([1.0, 1.0, 1.0])

        segs = infer_volume(model, imgs, gts, device, args.bbox_shift)

        if args.save_pred_dir:
            pred_path = join(args.save_pred_dir, basename(npz_path))
            np.savez_compressed(pred_path, segs=segs, gts=gts, spacing=spacing)

        spacing_zyx = spacing_to_zyx(spacing)
        dice, hd95, asd, n_labels = compute_case_metrics(segs, gts, spacing_zyx)
        rows.append(
            {
                "case": basename(npz_path),
                "dice": dice,
                "hd95": hd95,
                "asd": asd,
                "n_labels": n_labels,
            }
        )

    dump_case_csv(rows, args.out_csv)

    dice_arr = np.array([float(row["dice"]) for row in rows], dtype=float)
    hd95_arr = np.array([float(row["hd95"]) for row in rows], dtype=float)
    asd_arr = np.array([float(row["asd"]) for row in rows], dtype=float)
    summary = {
        "exp_name": args.exp_name,
        "num_cases": int(len(rows)),
        "dice_mean": float(np.mean(dice_arr)) if len(rows) else 0.0,
        "dice_std": float(np.std(dice_arr)) if len(rows) else 0.0,
        "hd95_mean": float(np.mean(hd95_arr)) if len(rows) else 0.0,
        "hd95_std": float(np.std(hd95_arr)) if len(rows) else 0.0,
        "asd_mean": float(np.mean(asd_arr)) if len(rows) else 0.0,
        "asd_std": float(np.std(asd_arr)) if len(rows) else 0.0,
        "checkpoint": args.checkpoint,
        "data_root": args.data_root,
        "bbox_shift": int(args.bbox_shift),
        "device": args.device,
    }

    with open(args.out_json, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\n[summary]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[output] csv: {args.out_csv}")
    print(f"[output] json: {args.out_json}")


if __name__ == "__main__":
    main()
