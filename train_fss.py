# -*- coding: utf-8 -*-
"""DDP training script for MedSAM-FSS with optional attention fusion."""

import argparse
import glob
import os
import random
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import monai
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from losses import BalanceLoss, InterClassBalanceLoss, IntraClassBalanceLoss
from models import MedSAMFSS
from segment_anything import sam_model_registry

join = os.path.join

torch.manual_seed(2023)
torch.cuda.empty_cache()


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(glob.glob(join(self.gt_path, "**/*.npy"), recursive=True))
        self.gt_path_files = [
            p for p in self.gt_path_files if os.path.isfile(join(self.img_path, os.path.basename(p)))
        ]
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        gt_file = self.gt_path_files[index]
        img_name = os.path.basename(gt_file)
        img_1024 = np.load(join(self.img_path, img_name), "r", allow_pickle=True)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0, "image should be normalized to [0,1]"

        gt = np.load(gt_file, "r", allow_pickle=True)
        label_ids = np.unique(gt)[1:]
        gt2d = np.uint8(gt == random.choice(label_ids.tolist()))
        assert np.max(gt2d) == 1 and np.min(gt2d) == 0, "ground truth should be binary"

        y_indices, x_indices = np.where(gt2d > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        height, width = gt2d.shape

        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(width, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(height, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])

        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2d[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--tr_npy_path", type=str, default="data/npy/CT_Abd")
    parser.add_argument("-task_name", type=str, default="MedSAM-FSS")
    parser.add_argument("-model_type", type=str, default="vit_b")
    parser.add_argument("-checkpoint", type=str, default="work_dir/medsam_vit_b.pth")
    parser.add_argument("-work_dir", type=str, default="./work_dir")
    parser.add_argument("-num_epochs", type=int, default=200)
    parser.add_argument("-batch_size", type=int, default=8)
    parser.add_argument("-num_workers", type=int, default=8)
    parser.add_argument("-weight_decay", type=float, default=0.01)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("-use_wandb", type=str2bool, default=False)
    parser.add_argument("-use_amp", action="store_true", default=False)

    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--bucket_cap_mb", type=int, default=25)
    parser.add_argument("--grad_acc_steps", type=int, default=1)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--init_method", type=str, default="env://")

    parser.add_argument(
        "-loss_type",
        type=str,
        default="baseline",
        choices=["baseline", "dicece", "inter_cbl", "intra_cbl", "balance"],
    )
    parser.add_argument("-stage1_epochs", type=int, default=50)
    parser.add_argument("--inter_neg_ratio", type=float, default=3.0)
    parser.add_argument("--intra_threshold", type=float, default=0.9)
    parser.add_argument("--intra_hard_weight", type=float, default=2.0)
    parser.add_argument("--balance_alpha", type=float, default=1.0)
    parser.add_argument("--balance_beta", type=float, default=1.0)
    parser.add_argument("--balance_gamma", type=float, default=1.0)
    parser.add_argument("--balance_neg_ratio", type=float, default=3.0)
    parser.add_argument("--balance_hard_threshold", type=float, default=0.9)
    parser.add_argument("--balance_hard_weight", type=float, default=2.0)

    parser.add_argument("-use_attention", type=str2bool, default=False)
    parser.add_argument("-attention_embed_dim", type=int, default=256)
    parser.add_argument("-attention_num_heads", type=int, default=8)
    parser.add_argument("-num_support", type=int, default=1)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.world_size is None:
        args.world_size = torch.cuda.device_count()

    args.run_id = datetime.now().strftime("%Y%m%d-%H%M")
    args.model_save_path = join(args.work_dir, args.task_name + "-" + args.run_id)

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
                "use_attention": args.use_attention,
                "attention_embed_dim": args.attention_embed_dim,
                "attention_num_heads": args.attention_num_heads,
            },
        )

    ngpus_per_node = torch.cuda.device_count()
    print("Spawning processes")
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    node_rank = int(args.node_rank)
    rank = node_rank * ngpus_per_node + gpu
    world_size = int(args.world_size)
    is_main_host = rank == 0

    print(f"[Rank {rank}] Use GPU: {gpu}")
    if is_main_host:
        os.makedirs(args.model_save_path, exist_ok=True)
        shutil.copyfile(__file__, join(args.model_save_path, args.run_id + "_" + os.path.basename(__file__)))

    torch.cuda.set_device(gpu)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.init_method,
        rank=rank,
        world_size=world_size,
    )

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAMFSS(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        use_support_attention=args.use_attention,
        attention_embed_dim=args.attention_embed_dim,
        attention_num_heads=args.attention_num_heads,
    ).cuda()

    medsam_model = nn.parallel.DistributedDataParallel(
        medsam_model,
        device_ids=[gpu],
        output_device=gpu,
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
        bucket_cap_mb=args.bucket_cap_mb,
    )
    medsam_model.train()

    if is_main_host:
        print(
            f"Attention enabled={args.use_attention}, "
            f"embed_dim={args.attention_embed_dim}, heads={args.attention_num_heads}"
        )
        print(f"Total parameters: {sum(p.numel() for p in medsam_model.parameters())}")
        print(f"Trainable parameters: {sum(p.numel() for p in medsam_model.parameters() if p.requires_grad)}")

    trainable_params = list(medsam_model.module.image_encoder.parameters()) + list(
        medsam_model.module.mask_decoder.parameters()
    )
    if args.use_attention and medsam_model.module.attention_fusion is not None:
        trainable_params += list(medsam_model.module.attention_fusion.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    loss_type = "baseline" if args.loss_type == "dicece" else args.loss_type
    if loss_type == "inter_cbl":
        custom_loss = InterClassBalanceLoss(neg_ratio=args.inter_neg_ratio)
        print(f"[Rank {rank}] Using Inter-CBL + Dice (neg_ratio={args.inter_neg_ratio})")
    elif loss_type == "intra_cbl":
        custom_loss = IntraClassBalanceLoss(
            hard_threshold=args.intra_threshold,
            hard_weight=args.intra_hard_weight,
        )
        print(
            f"[Rank {rank}] Using Intra-CBL + Dice (threshold={args.intra_threshold}, "
            f"hard_weight={args.intra_hard_weight})"
        )
    elif loss_type == "balance":
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
            f"[Rank {rank}] Using Balance Loss: alpha={args.balance_alpha}, beta={args.balance_beta}, "
            f"gamma={args.balance_gamma}, stage1_epochs={args.stage1_epochs}, "
            f"neg_ratio={args.balance_neg_ratio}, hard_threshold={args.balance_hard_threshold}, "
            f"hard_weight={args.balance_hard_weight}"
        )
    else:
        custom_loss = None
        print(f"[Rank {rank}] Using Baseline Dice + BCE")

    train_dataset = NpyDataset(args.tr_npy_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"[Rank {rank}] Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=f"cuda:{gpu}")
        medsam_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        torch.distributed.barrier()

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    losses = []
    best_loss = float("inf")

    for epoch in range(start_epoch, args.num_epochs):
        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0

        for step, (image, gt2d, boxes, _) in enumerate(
            tqdm(train_dataloader, desc=f"[Rank {rank}] Epoch {epoch}")
        ):
            optimizer.zero_grad(set_to_none=True)
            boxes_np = boxes.detach().cpu().numpy()
            image = image.cuda(non_blocking=True)
            gt2d = gt2d.cuda(non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.use_amp):
                pred = medsam_model(image, boxes_np)
                if loss_type == "balance":
                    loss = custom_loss(pred, gt2d, epoch=epoch)
                elif loss_type in {"inter_cbl", "intra_cbl"}:
                    loss = custom_loss(pred, gt2d) + seg_loss(pred, gt2d)
                else:
                    loss = seg_loss(pred, gt2d) + ce_loss(pred, gt2d.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if step > 10 and step % 100 == 0 and is_main_host:
                checkpoint = {"model": medsam_model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
                torch.save(checkpoint, join(args.model_save_path, "medsam_model_latest_step.pth"))

        epoch_loss /= max(1, len(train_dataloader))
        losses.append(epoch_loss)
        print(f"Time: {datetime.now().strftime('%Y%m%d-%H%M')}, Epoch: {epoch}, Loss: {epoch_loss}")

        if args.use_wandb:
            import wandb

            if rank == 0:
                wandb.log({"epoch_loss": epoch_loss, "epoch": epoch})

        if is_main_host:
            checkpoint = {"model": medsam_model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
            torch.save(checkpoint, join(args.model_save_path, "medsam_model_latest.pth"))
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(checkpoint, join(args.model_save_path, "medsam_model_best.pth"))

            plt.plot(losses)
            plt.title(f"{args.task_name} ({loss_type}) Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(join(args.model_save_path, args.task_name + "_train_loss.png"))
            plt.close()

        torch.distributed.barrier()

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
