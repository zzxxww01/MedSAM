# -*- coding: utf-8 -*-
"""
统计预处理后的前景切片数量
在服务器上运行: python scripts/count_slices.py

会自动统计 data/npy/CT_Abd 和 data/npy/CT_AMOS 目录下的切片数
"""
import os
import glob

def count_slices(npy_dir):
    """统计某个 npy 目录下的前景切片数量和病例数"""
    gt_path = os.path.join(npy_dir, "gts")
    if not os.path.exists(gt_path):
        print(f"目录不存在: {gt_path}")
        return

    files = sorted(glob.glob(os.path.join(gt_path, "*.npy")))
    total = len(files)

    # 按病例名分组（去掉最后的 -XXX.npy 后缀）
    cases = set()
    for f in files:
        name = os.path.basename(f)
        # 格式: CT_Abd_FLARE22_Tr_XXXX-YYY.npy 或 CT_AMOS_amos_XXXX-YYY.npy
        case_name = name.rsplit("-", 1)[0]
        cases.add(case_name)

    print(f"\n目录: {npy_dir}")
    print(f"  病例数: {len(cases)}")
    print(f"  总切片数: {total}")
    print(f"  平均每病例切片数: {total / len(cases):.1f}" if cases else "")

    # 按病例统计详细信息
    case_counts = {}
    for f in files:
        name = os.path.basename(f)
        case_name = name.rsplit("-", 1)[0]
        case_counts[case_name] = case_counts.get(case_name, 0) + 1

    counts = sorted(case_counts.values())
    print(f"  切片数范围: {min(counts)} ~ {max(counts)}")
    print(f"  中位数: {counts[len(counts)//2]}")

    return total, len(cases)


if __name__ == "__main__":
    print("=" * 50)
    print("前景切片统计")
    print("=" * 50)

    for dataset_dir in ["data/npy/CT_Abd", "data/npy/CT_AMOS"]:
        count_slices(dataset_dir)

    print("\n请将以上数字告诉我，以便更新论文中的数据集描述。")
