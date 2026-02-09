# -*- coding: utf-8 -*-
"""
AMOS22 数据集预处理脚本
基于 pre_CT_MR.py 修改，适配 AMOS22 数据集结构

AMOS22 器官标签 (15个):
1: 脾脏, 2: 右肾, 3: 左肾, 4: 胆囊, 5: 食道
6: 肝脏, 7: 胃, 8: 主动脉, 9: 下腔静脉, 10: 胰腺
11: 右肾上腺, 12: 左肾上腺, 13: 十二指肠, 14: 膀胱, 15: 前列腺/子宫
"""

import numpy as np
import SimpleITK as sitk
import os
from os.path import join
from skimage import transform
from tqdm import tqdm
import cc3d

# ============ 配置参数 ============
modality = "CT"
anatomy = "AMOS"
prefix = modality + "_" + anatomy + "_"

# AMOS22 数据集路径
nii_path = "data/AMOS22/imagesTr"
gt_path = "data/AMOS22/labelsTr"
npy_path = "data/npy/CT_AMOS"

# 创建输出目录
os.makedirs(join(npy_path, "gts"), exist_ok=True)
os.makedirs(join(npy_path, "imgs"), exist_ok=True)

# 预处理参数
image_size = 1024
voxel_num_thre2d = 100
voxel_num_thre3d = 1000

# CT窗宽窗位 (腹部)
WINDOW_LEVEL = 40
WINDOW_WIDTH = 400

# 要移除的标签ID (可选)
remove_label_ids = []  # AMOS22保留所有标签
tumor_id = None

# ============ 获取文件列表 ============
# AMOS22 文件命名: amos_0001.nii.gz
names = sorted([f for f in os.listdir(gt_path) if f.endswith('.nii.gz')])
print(f"原始文件数量: {len(names)}")

# 过滤只保留CT图像 (AMOS22中CT编号为0001-0500)
ct_names = []
for name in names:
    # 提取编号
    num = int(name.replace('amos_', '').replace('.nii.gz', ''))
    if num <= 500:  # CT图像编号1-500
        img_path_check = join(nii_path, name)
        if os.path.exists(img_path_check):
            ct_names.append(name)

names = ct_names
print(f"CT图像数量: {len(names)}")

# ============ 主处理循环 ============
for name in tqdm(names):
    gt_sitk = sitk.ReadImage(join(gt_path, name))
    gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))

    # 移除指定标签
    for remove_label_id in remove_label_ids:
        gt_data_ori[gt_data_ori == remove_label_id] = 0

    # 处理肿瘤实例分割 (如果需要)
    if tumor_id is not None:
        tumor_bw = np.uint8(gt_data_ori == tumor_id)
        gt_data_ori[tumor_bw > 0] = 0
        tumor_inst, _ = cc3d.connected_components(
            tumor_bw, connectivity=26, return_N=True
        )
        gt_data_ori[tumor_inst > 0] = (
            tumor_inst[tumor_inst > 0] + np.max(gt_data_ori) + 1
        )

    # 3D去除小目标
    gt_data_ori = cc3d.dust(
        gt_data_ori, threshold=voxel_num_thre3d, connectivity=26, in_place=True
    )

    # 2D去除小目标
    for slice_i in range(gt_data_ori.shape[0]):
        gt_i = gt_data_ori[slice_i, :, :]
        gt_data_ori[slice_i, :, :] = cc3d.dust(
            gt_i, threshold=voxel_num_thre2d, connectivity=8, in_place=True
        )

    # 找到非空切片
    z_index, _, _ = np.where(gt_data_ori > 0)
    z_index = np.unique(z_index)

    if len(z_index) == 0:
        print(f"跳过 {name}: 无有效标签")
        continue

    # 裁剪ROI
    gt_roi = gt_data_ori[z_index, :, :]

    # 读取图像
    img_sitk = sitk.ReadImage(join(nii_path, name))
    image_data = sitk.GetArrayFromImage(img_sitk)

    # CT窗宽窗位预处理
    lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
    upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
    image_data_pre = (
        (image_data_pre - np.min(image_data_pre))
        / (np.max(image_data_pre) - np.min(image_data_pre))
        * 255.0
    )
    image_data_pre = np.uint8(image_data_pre)
    img_roi = image_data_pre[z_index, :, :]

    # 获取文件名前缀
    case_name = name.replace('.nii.gz', '')

    # 保存每个切片为npy文件
    for i in range(img_roi.shape[0]):
        img_i = img_roi[i, :, :]
        gt_i = gt_roi[i, :, :]

        # 转换为3通道RGB
        img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)

        # 缩放到1024x1024
        resize_img = transform.resize(
            img_3c,
            (image_size, image_size),
            order=3,
            preserve_range=True,
            mode="constant",
            anti_aliasing=True,
        )

        # 归一化到[0, 1]
        resize_img_01 = (resize_img - resize_img.min()) / np.clip(
            resize_img.max() - resize_img.min(), a_min=1e-8, a_max=None
        )

        # 缩放标签 (最近邻插值)
        resize_gt = transform.resize(
            gt_i,
            (image_size, image_size),
            order=0,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )
        resize_gt = np.uint8(resize_gt)

        # 保存文件
        slice_name = f"{prefix}{case_name}-{str(i).zfill(3)}.npy"
        np.save(join(npy_path, "imgs", slice_name), resize_img_01)
        np.save(join(npy_path, "gts", slice_name), resize_gt)

print(f"预处理完成! 输出目录: {npy_path}")
