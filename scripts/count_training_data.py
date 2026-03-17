#!/usr/bin/env python
"""Count training data statistics for thesis writing.

Run on server:
    cd ~/chengang/zxw/MedSAM && python scripts/count_training_data.py

Outputs exact numbers for:
  - FLARE22 (CT_Abd): number of 3D cases, 2D slices
  - AMOS22 (CT_AMOS): number of 3D cases, 2D slices
  - Combined totals
"""

import glob
import os

def count_npy_dataset(data_root, name):
    imgs_dir = os.path.join(data_root, "imgs")
    gts_dir = os.path.join(data_root, "gts")

    if not os.path.isdir(imgs_dir):
        print(f"[{name}] Directory not found: {imgs_dir}")
        return

    img_files = sorted(glob.glob(os.path.join(imgs_dir, "**/*.npy"), recursive=True))
    gt_files = sorted(glob.glob(os.path.join(gts_dir, "**/*.npy"), recursive=True))
    npz_files = sorted(glob.glob(os.path.join(data_root, "*.npz")))

    # Extract unique case names (everything before the last -NNN.npy)
    case_names = set()
    for f in img_files:
        base = os.path.basename(f)
        # e.g. CT_Abd_FLARE_0001-042.npy -> CT_Abd_FLARE_0001
        parts = base.rsplit("-", 1)
        if len(parts) == 2:
            case_names.add(parts[0])
        else:
            case_names.add(base.replace(".npy", ""))

    print(f"\n{'='*50}")
    print(f"  Dataset: {name}")
    print(f"  Path:    {data_root}")
    print(f"{'='*50}")
    print(f"  3D NPZ volumes:    {len(npz_files)}")
    print(f"  2D image slices:   {len(img_files)}")
    print(f"  2D GT slices:      {len(gt_files)}")
    print(f"  Unique 3D cases:   {len(case_names)}")
    if case_names:
        print(f"  Case name samples: {sorted(case_names)[:3]} ...")
    return len(img_files)


if __name__ == "__main__":
    total = 0

    # FLARE22
    n = count_npy_dataset("data/npy/CT_Abd", "FLARE22 (CT_Abd)")
    if n:
        total += n

    # AMOS22
    n = count_npy_dataset("data/npy/CT_AMOS", "AMOS22 (CT_AMOS)")
    if n:
        total += n

    print(f"\n{'='*50}")
    print(f"  COMBINED TOTAL 2D slices: {total}")
    print(f"{'='*50}")
