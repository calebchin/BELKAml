import torch
import torch.nn as nn
import torch.nn.functional as F

class CategoricalLoss(nn.Module):
    """
    Masked Categorical Focal loss.
    Dynamic mini-batch class weights ("alpha").
    Used for MLM training.
    """
    def __init__(self, epsilon: float, mask: int, vocab_size: int, gamma: float = 2.0):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = gamma
        self.mask = mask
        self.vocab_size = vocab_size

    def forward(self, y_pred, y_true):
        # Unpack y_true into masked and unmasked arrays
        unmasked = y_true[:, :, 1]
        y_true = y_true[:, :, 0]

        # Flatten
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1, self.vocab_size)

        # Drop non-masked
        mask = (y_true != self.mask)
        y_true = y_true[mask]

        # Compute class weights (alpha)
        freq = torch.bincount(unmasked.view(-1), minlength=self.vocab_size).float()
        freq[:2] = 0  # set [PAD], [MASK] frequencies to 0
        alpha = torch.where(freq == 0.0, torch.zeros_like(freq), torch.rsqrt(freq))

        # One-hot encode targets
        y_true_oh = F.one_hot(y_true, num_classes=self.vocab_size).float()
        y_pred = y_pred[mask]

        # Clip predictions
        y_pred = torch.clamp(y_pred, min=self.epsilon, max=1.0 - self.epsilon)

        pt = y_true_oh * y_pred + (1.0 - y_true_oh) * (1.0 - y_pred)
        loss = -alpha * ((1.0 - pt) ** self.gamma) * (y_true_oh * torch.log(y_pred))
        loss = loss.sum() / (alpha * y_true_oh).sum()

        return loss
