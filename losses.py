import torch
import torch.nn as nn

class BalanceCrossEntropyLoss(nn.Module):
    """
    Balanced cross-entropy loss to handle class imbalance by selecting a subset of negative samples.
    Shape:
        - Input: :math:`(N, 1, H, W)`
        - GT: :math:`(N, 1, H, W)`, same shape as the input
        - Mask: :math:`(N, H, W)`, same spatial shape as the input
        - Output: scalar.
    """
    def __init__(self, negative_ratio=3.0, eps=1e-6):
        """
        Initialize the loss module.

        Args:
            negative_ratio (float): Ratio of negative to positive samples to use in loss computation.
            eps (float): Small value to prevent division by zero.
        """
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, return_origin=False):
        """
        Compute the balanced cross-entropy loss.

        Args:
            pred (Tensor): Predicted probabilities, shape :math:`(N, 1, H, W)`.
            gt (Tensor): Ground truth binary labels, shape :math:`(N, 1, H, W)`.
            mask (Tensor): Mask indicating positive regions, shape :math:`(N, H, W)`.
            return_origin (bool): If True, return the original loss along with the balanced loss.

        Returns:
            Tensor: The computed loss. If return_origin is True, returns a tuple (balanced_loss, original_loss).
        """
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(positive_count * self.negative_ratio))
        loss = nn.functional.binary_cross_entropy(pred, gt, reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        # negative_loss, _ = torch.topk(negative_loss.view(-1).contiguous(), negative_count)
        negative_loss, _ = negative_loss.view(-1).topk(negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)

        if return_origin:
            return balance_loss, loss
        return balance_loss


class DiceLoss(nn.Module):
    """
    Dice loss for measuring similarity between predicted and ground truth heatmaps, useful in segmentation.
    """
    def __init__(self, eps=1e-6):
        """
        Initialize the loss module.

        Args:
            eps (float): Small value to prevent division by zero.
        """
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt, mask, weights=None):
        """
        Compute the Dice loss.

        Args:
            pred (Tensor): Predicted probabilities, shape (N, 1, H, W) or (N, H, W).
            gt (Tensor): Ground truth binary labels, shape (N, 1, H, W).
            mask (Tensor): Mask indicating regions to consider, shape (N, H, W).
            weights (Tensor, optional): Weights for each pixel, shape (N, H, W).

        Returns:
            Tensor: The computed Dice loss.
        """
        return self._compute(pred, gt, mask, weights)

    def _compute(self, pred, gt, mask, weights):
        """
        Internal method to calculate the Dice loss.

        Args:
            pred (Tensor): Predicted probabilities, shape (N, H, W).
            gt (Tensor): Ground truth binary labels, shape (N, H, W).
            mask (Tensor): Mask indicating regions to consider, shape (N, H, W).
            weights (Tensor, optional): Weights for each pixel, shape (N, H, W).

        Returns:
            Tensor: The computed Dice loss.
        """
        if pred.dim() == 4:
            pred = pred[:, 0, :, :]
            gt = gt[:, 0, :, :]
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        intersection = (pred * gt * mask).sum()

        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class MaskL1Loss(nn.Module):
    """
    Masked L1 loss, computes the L1 loss only on regions specified by the mask.
    """
    def __init__(self, eps=1e-6):
        """
        Initialize the loss module.

        Args:
            eps (float): Small value to prevent division by zero.
        """
        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt, mask):
        """
        Compute the masked L1 loss.

        Args:
            pred (Tensor): Predicted values, shape (N, C, H, W).
            gt (Tensor): Ground truth values, shape (N, C, H, W).
            mask (Tensor): Mask indicating regions to consider, shape (N, H, W).

        Returns:
            Tensor: The computed masked L1 loss.
        """
        loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        return loss


class DBLoss(nn.Module):
    """
    Differentiable Binarization (DB) loss combining balanced cross-entropy, Dice, and masked L1 losses.
    """
    def __init__(self, alpha=1.0, beta=10, ohem_ratio=3, reduction='mean', eps=1e-6):
        """
        Initialize the DB loss module.

        Args:
            alpha (float): Weight for the shrink map loss.
            beta (float): Weight for the threshold map loss.
            ohem_ratio (float): Ratio for Online Hard Example Mining in balanced cross-entropy.
            reduction (str): Reduction method, 'mean' or 'sum' (currently not used in computation).
            eps (float): Small value to prevent division by zero.
        """
        super().__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio)
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction

    def forward(self, pred, batch):
        """
        Compute the DB loss.

        Args:
            pred (Tensor): Predictions, shape (N, C, H, W), where C=3 during training, C=2 during inference.
            batch (dict): Dictionary with ground truth tensors:
                - 'shrink_map': shape (N, 1, H, W)
                - 'shrink_mask': shape (N, H, W)
                - 'threshold_map': shape (N, 1, H, W)
                - 'threshold_mask': shape (N, H, W)

        Returns:
            dict: Computed losses:
                - 'loss_shrink_maps': loss for shrink maps
                - 'loss_threshold_maps': loss for threshold maps
                - 'loss_binary_maps': loss for binary maps (only during training)
                - 'loss': total loss
        """
        shrink_maps = pred[:, 0, :, :]
        threshold_maps = pred[:, 1, :, :]
        binary_maps = pred[:, 2, :, :]

        loss_shrink_maps = self.bce_loss(shrink_maps, batch['shrink_map'], batch['shrink_mask'])
        loss_threshold_maps = self.l1_loss(threshold_maps, batch['threshold_map'], batch['threshold_mask'])
        metrics = dict(loss_shrink_maps=loss_shrink_maps, loss_threshold_maps=loss_threshold_maps)
        if pred.size()[1] > 2:
            loss_binary_maps = self.dice_loss(binary_maps, batch['shrink_map'], batch['shrink_mask'])
            metrics['loss_binary_maps'] = loss_binary_maps
            loss_all = self.alpha * loss_shrink_maps + self.beta * loss_threshold_maps + loss_binary_maps
            metrics['loss'] = loss_all
        else:
            metrics['loss'] = loss_shrink_maps
        return metrics