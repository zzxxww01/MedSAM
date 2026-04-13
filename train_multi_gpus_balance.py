# -*- coding: utf-8 -*-
"""
MedSAM 训练脚本 - Balance Loss 版本
基于 train_multi_gpus.py 修改，集成 Balance Loss

支持的损失类型:
- baseline: Dice + CE (原始)
- inter_cbl: Inter-CBL + Dice
- intra_cbl: Intra-CBL + Dice
- balance: 完整 Balance Loss
"""

import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob

# 导入 Balance Loss 模块
from losses import InterClassBalanceLoss, IntraClassBalanceLoss, BalanceLoss

# 设置随机种子
torch.manual_seed(2023)
torch.cuda.empty_cache()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error: "
            + img_name
            + " vs "
            + os.path.basename(self.gt_path_files[index])
        )
        label_ids = np.unique(gt)[1:]
        gt2D = np.uint8(gt == random.choice(label_ids.tolist()))

        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )


# 参数解析器
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--tr_npy_path", type=str, default="data/npy/CT_Abd",
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument("-checkpoint", type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth")
parser.add_argument("--load_pretrain", type=bool, default=True)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
parser.add_argument("-num_epochs", type=int, default=1000)
parser.add_argument("-batch_size", type=int, default=8)
parser.add_argument("-num_workers", type=int, default=8)
parser.add_argument("-weight_decay", type=float, default=0.01)
parser.add_argument("-lr", type=float, default=0.0001, metavar="LR")
parser.add_argument("-use_wandb", type=bool, default=False)
parser.add_argument("-use_amp", action="store_true", default=False)

# 分布式训练参数
parser.add_argument("--world_size", type=int, help="world size")
parser.add_argument("--node_rank", type=int, default=0)
parser.add_argument("--bucket_cap_mb", type=int, default=25)
parser.add_argument("--grad_acc_steps", type=int, default=1)
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--init_method", type=str, default="env://")

# Balance Loss 参数
parser.add_argument(
    "-loss_type", type=str, default="baseline",
    choices=["baseline", "inter_cbl", "intra_cbl", "balance"],
    help="损失函数类型"
)
parser.add_argument("-stage1_epochs", type=int, default=50, help="Balance Loss第一阶段epoch数")
parser.add_argument("--inter_neg_ratio", "-inter_neg_ratio", type=float, default=3.0, help="Inter-CBL负样本比例")
parser.add_argument(
    "--intra_threshold",
    "-intra_threshold",
    type=float,
    default=0.9,
    help="Intra-CBL困难样本阈值",
)
parser.add_argument(
    "--intra_hard_weight",
    "-intra_hard_weight",
    type=float,
    default=2.0,
    help="Intra-CBL困难样本权重",
)
parser.add_argument("--balance_alpha", "-balance_alpha", type=float, default=1.0, help="BalanceLoss中Inter-CBL权重")
parser.add_argument("--balance_beta", "-balance_beta", type=float, default=1.0, help="BalanceLoss中Intra-CBL权重")
parser.add_argument("--balance_gamma", "-balance_gamma", type=float, default=1.0, help="BalanceLoss中Dice权重")
parser.add_argument(
    "--balance_neg_ratio",
    "-balance_neg_ratio",
    type=float,
    default=3.0,
    help="BalanceLoss中Inter-CBL负样本比例",
)
parser.add_argument(
    "--balance_hard_threshold",
    "-balance_hard_threshold",
    type=float,
    default=0.9,
    help="BalanceLoss中Intra-CBL困难样本阈值",
)
parser.add_argument(
    "--balance_hard_weight",
    "-balance_hard_weight",
    type=float,
    default=2.0,
    help="BalanceLoss中Intra-CBL困难样本权重",
)

args = parser.parse_args()

if args.use_wandb:
    import wandb
    wandb.login()
    wandb.init(
        project=args.task_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "model_type": args.model_type,
            "loss_type": args.loss_type,
            "stage1_epochs": args.stage1_epochs,
            "inter_neg_ratio": args.inter_neg_ratio,
            "intra_threshold": args.intra_threshold,
            "intra_hard_weight": args.intra_hard_weight,
            "balance_alpha": args.balance_alpha,
            "balance_beta": args.balance_beta,
            "balance_gamma": args.balance_gamma,
            "balance_neg_ratio": args.balance_neg_ratio,
            "balance_hard_threshold": args.balance_hard_threshold,
            "balance_hard_weight": args.balance_hard_weight,
        },
    )

run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)


class MedSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # 冻结prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=box_torch, masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


def main():
    ngpus_per_node = torch.cuda.device_count()
    print("Spawning processes")
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    node_rank = int(args.node_rank)
    rank = node_rank * ngpus_per_node + gpu
    world_size = args.world_size
    print(f"[Rank {rank}]: Use GPU: {gpu} for training")
    is_main_host = rank == 0
    if is_main_host:
        os.makedirs(model_save_path, exist_ok=True)
        shutil.copyfile(
            __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
        )
    torch.cuda.set_device(gpu)
    torch.distributed.init_process_group(
        backend="nccl", init_method=args.init_method, rank=rank, world_size=world_size
    )

    # 加载模型
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder
    ).cuda()

    # DDP包装
    medsam_model = nn.parallel.DistributedDataParallel(
        medsam_model,
        device_ids=[gpu],
        output_device=gpu,
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
        bucket_cap_mb=args.bucket_cap_mb,
    )
    medsam_model.train()

    print(f"Total parameters: {sum(p.numel() for p in medsam_model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in medsam_model.parameters() if p.requires_grad)}")

    # 优化器
    img_mask_encdec_params = list(
        medsam_model.module.image_encoder.parameters()
    ) + list(medsam_model.module.mask_decoder.parameters())
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )

    # 损失函数设置
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    # 根据loss_type选择损失函数
    if args.loss_type == "inter_cbl":
        custom_loss = InterClassBalanceLoss(neg_ratio=args.inter_neg_ratio)
        print(f"Using Inter-CBL + Dice Loss (neg_ratio={args.inter_neg_ratio})")
    elif args.loss_type == "intra_cbl":
        custom_loss = IntraClassBalanceLoss(
            hard_threshold=args.intra_threshold, hard_weight=args.intra_hard_weight
        )
        print(
            f"Using Intra-CBL + Dice Loss (threshold={args.intra_threshold}, hard_weight={args.intra_hard_weight})"
        )
    elif args.loss_type == "balance":
        custom_loss = BalanceLoss(
            alpha=args.balance_alpha,
            beta=args.balance_beta,
            gamma=args.balance_gamma,
            stage1_epochs=args.stage1_epochs,
            neg_ratio=args.balance_neg_ratio,
            hard_threshold=args.balance_hard_threshold,
            hard_weight=args.balance_hard_weight,
        )
        print(
            "Using Balance Loss (full): "
            f"alpha={args.balance_alpha}, beta={args.balance_beta}, gamma={args.balance_gamma}, "
            f"stage1_epochs={args.stage1_epochs}, neg_ratio={args.balance_neg_ratio}, "
            f"hard_threshold={args.balance_hard_threshold}, hard_weight={args.balance_hard_weight}"
        )
    else:
        custom_loss = None
        print("Using Baseline (Dice + CE Loss)")

    # 数据加载
    num_epochs = args.num_epochs
    losses = []
    best_loss = 1e10
    train_dataset = NpyDataset(args.tr_npy_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    print(f"Number of training samples: {len(train_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    # 恢复训练
    start_epoch = 0
    if args.resume is not None and os.path.isfile(args.resume):
        print(f"[Rank {rank}] Loading checkpoint '{args.resume}'")
        loc = f"cuda:{gpu}"
        checkpoint = torch.load(args.resume, map_location=loc)
        start_epoch = checkpoint["epoch"] + 1
        medsam_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"[Rank {rank}] Loaded checkpoint (epoch {checkpoint['epoch']})")
        torch.distributed.barrier()

    # AMP设置
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print(f"[Rank {rank}] Using AMP for training")

    # 训练循环
    iter_num = 0
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        train_dataloader.sampler.set_epoch(epoch)

        for step, (image, gt2D, boxes, _) in enumerate(
            tqdm(train_dataloader, desc=f"[Rank {rank}] Epoch {epoch}")
        ):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.cuda(), gt2D.cuda()

            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = medsam_model(image, boxes_np)
                    # 根据损失类型计算损失
                    if args.loss_type == "balance":
                        loss = custom_loss(medsam_pred, gt2D, epoch=epoch)
                    elif args.loss_type in ["inter_cbl", "intra_cbl"]:
                        loss = custom_loss(medsam_pred, gt2D) + seg_loss(medsam_pred, gt2D)
                    else:
                        loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                medsam_pred = medsam_model(image, boxes_np)
                # 根据损失类型计算损失
                if args.loss_type == "balance":
                    loss = custom_loss(medsam_pred, gt2D, epoch=epoch)
                elif args.loss_type in ["inter_cbl", "intra_cbl"]:
                    loss = custom_loss(medsam_pred, gt2D) + seg_loss(medsam_pred, gt2D)
                else:
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # 定期保存checkpoint
            if step > 10 and step % 100 == 0 and is_main_host:
                checkpoint = {
                    "model": medsam_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }
                torch.save(checkpoint, join(model_save_path, "medsam_model_latest_step.pth"))

            epoch_loss += loss.item()
            iter_num += 1

        # Epoch结束处理
        epoch_loss /= max(1, len(train_dataloader))
        losses.append(epoch_loss)

        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})

        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}')

        # 保存模型
        if is_main_host:
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))

        torch.distributed.barrier()

        # 绘制损失曲线
        plt.plot(losses)
        plt.title(f"{args.loss_type} Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, args.task_name + "_train_loss.png"))
        plt.close()


if __name__ == "__main__":
    main()
