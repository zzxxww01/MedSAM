# -*- coding: utf-8 -*-
"""
Balance Loss 实现
包含:
- InterClassBalanceLoss: 类别间平衡 (前景vs背景)
- IntraClassBalanceLoss: 类别内平衡 (简单vs困难样本)
- BalanceLoss: 完整损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InterClassBalanceLoss(nn.Module):
    """
    类别间平衡损失 (Inter-Class Balance Loss)
    通过挖掘困难背景样本，使前景和背景的损失贡献相等
    """

    def __init__(self, neg_ratio=3.0):
        """
        Args:
            neg_ratio: 负样本与正样本的比例
        """
        super().__init__()
        self.neg_ratio = neg_ratio

    def forward(self, pred, target):
        """
        Args:
            pred: 预测logits (B, 1, H, W)
            target: 真实标签 (B, 1, H, W)
        """
        pred = pred.view(-1)
        target = target.view(-1).float()

        # 分离前景和背景
        pos_mask = target > 0.5
        neg_mask = target <= 0.5

        num_pos = pos_mask.sum().item()
        num_neg = neg_mask.sum().item()

        if num_pos == 0:
            return F.binary_cross_entropy_with_logits(pred, target)

        # 计算前景损失
        pos_loss = F.binary_cross_entropy_with_logits(
            pred[pos_mask], target[pos_mask], reduction='sum'
        )

        # 挖掘困难背景样本
        neg_pred = pred[neg_mask]
        neg_target = target[neg_mask]

        # 选择困难负样本数量
        num_hard_neg = min(int(num_pos * self.neg_ratio), num_neg)

        if num_hard_neg > 0:
            # 按预测概率排序，选择最难的负样本
            neg_probs = torch.sigmoid(neg_pred)
            _, hard_indices = neg_probs.topk(num_hard_neg)

            neg_loss = F.binary_cross_entropy_with_logits(
                neg_pred[hard_indices], neg_target[hard_indices], reduction='sum'
            )
        else:
            neg_loss = torch.tensor(0.0, device=pred.device)

        # 平衡前景和背景损失
        total_loss = (pos_loss + neg_loss) / (num_pos + num_hard_neg + 1e-8)
        return total_loss


class IntraClassBalanceLoss(nn.Module):
    """
    类别内平衡损失 (Intra-Class Balance Loss)
    根据预测置信度划分难易样本，困难样本权重更高
    """

    def __init__(self, hard_threshold=0.5, hard_weight=2.0):
        """
        Args:
            hard_threshold: 困难样本阈值
            hard_weight: 困难样本权重
        """
        super().__init__()
        self.hard_threshold = hard_threshold
        self.hard_weight = hard_weight

    def forward(self, pred, target):
        """
        Args:
            pred: 预测logits (B, 1, H, W)
            target: 真实标签 (B, 1, H, W)
        """
        pred_flat = pred.view(-1)
        target_flat = target.view(-1).float()

        # 计算预测概率
        pred_prob = torch.sigmoid(pred_flat)

        # 计算每个像素的置信度误差
        confidence_error = torch.abs(pred_prob - target_flat)

        # 划分难易样本
        hard_mask = confidence_error > self.hard_threshold
        easy_mask = ~hard_mask

        # 计算权重
        weights = torch.ones_like(pred_flat)
        weights[hard_mask] = self.hard_weight

        # 加权BCE损失
        loss = F.binary_cross_entropy_with_logits(
            pred_flat, target_flat, weight=weights, reduction='mean'
        )
        return loss


class BalanceLoss(nn.Module):
    """
    完整Balance Loss
    loss = α×Inter-CBL + β×Intra-CBL + γ×Dice

    两阶段训练策略:
    - Stage 1 (Epoch 0-50): Intra-CBL + Dice (稳定启动)
    - Stage 2 (Epoch 50+): Inter-CBL + Intra-CBL + Dice (完整损失)
    """

    def __init__(
        self,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        stage1_epochs=50,
        neg_ratio=3.0,
        hard_threshold=0.5,
        hard_weight=2.0,
    ):
        """
        Args:
            alpha: Inter-CBL权重
            beta: Intra-CBL权重
            gamma: Dice Loss权重
            stage1_epochs: 第一阶段epoch数
            neg_ratio: 负样本比例
            hard_threshold: 困难样本阈值
            hard_weight: 困难样本权重
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.stage1_epochs = stage1_epochs

        self.inter_cbl = InterClassBalanceLoss(neg_ratio=neg_ratio)
        self.intra_cbl = IntraClassBalanceLoss(
            hard_threshold=hard_threshold, hard_weight=hard_weight
        )

    def dice_loss(self, pred, target, smooth=1e-5):
        """Dice Loss"""
        pred_prob = torch.sigmoid(pred)
        pred_flat = pred_prob.view(-1)
        target_flat = target.view(-1).float()

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + smooth) / (
            pred_flat.sum() + target_flat.sum() + smooth
        )
        return 1 - dice

    def forward(self, pred, target, epoch=0):
        """
        Args:
            pred: 预测logits (B, 1, H, W)
            target: 真实标签 (B, 1, H, W)
            epoch: 当前epoch数
        """
        # Dice Loss
        dice = self.dice_loss(pred, target)

        # Intra-CBL (始终启用)
        intra = self.intra_cbl(pred, target)

        # 两阶段策略
        if epoch < self.stage1_epochs:
            # Stage 1: Intra-CBL + Dice
            loss = self.beta * intra + self.gamma * dice
        else:
            # Stage 2: Inter-CBL + Intra-CBL + Dice
            inter = self.inter_cbl(pred, target)
            loss = self.alpha * inter + self.beta * intra + self.gamma * dice

        return loss
