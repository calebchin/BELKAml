import torch
import torch.nn as nn
import torch.nn.functional as f


class MultiLabelLoss(nn.Module):
    """Macro- or Micro-averaged Weighted Masked Binary Focal loss.
    Dynamic mini-batch class weights "alpha".
    Used for binary multilabel classification.
    """

    def __init__(
        self, epsilon: float, macro: bool, gamma: float = 2.0, nan_mask: int = 2
    ):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = gamma
        self.macro = macro
        self.nan_mask = nan_mask

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """Compute loss"""
        # Cast y_true to int
        y_true = y_true.long()

        # Compute class weights (alpha)
        if self.macro:
            freq = torch.stack(
                [
                    torch.bincount(y_true[:, i], minlength=2).float()
                    for i in range(y_true.shape[1])
                ],
                dim=0,
            )
        else:
            freq = torch.bincount(y_true.view(-1), minlength=2).float()

        alpha = torch.where(freq == 0.0, torch.zeros_like(freq), torch.rsqrt(freq))
        ax = 1 if self.macro else 0
        norm = freq.sum(dim=ax, keepdim=True) / (alpha * freq).sum(dim=ax, keepdim=True)
        alpha = alpha * norm

        one_hot = f.one_hot(y_true, num_classes=2).float()
        alpha = (alpha.unsqueeze(0) * one_hot).sum(-1)

        # Mask missing labels
        y_true = y_true.float()
        mask = (y_true != self.nan_mask).float()
        y_true = y_true * mask

        # Clip predictions
        y_pred = torch.clamp(y_pred, min=self.epsilon, max=1.0 - self.epsilon)

        pt = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        loss = -alpha * ((1.0 - pt) ** self.gamma) * torch.log(pt) * mask

        if self.macro:
            loss = loss.sum(dim=1) / (alpha * mask).sum(dim=1)
        else:
            loss = loss.sum() / (alpha * mask).sum()

        return loss.mean()


class BinaryLoss(nn.Module):
    """Binary Focal loss.
    Used for FPs training.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """Comput loss"""
        y_true = y_true.float().view(-1, 1)
        y_pred = y_pred.view(-1, 1)

        # Binary focal loss
        bce = f.binary_cross_entropy(y_pred, y_true, reduction="none")
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()
