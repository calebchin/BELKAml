import torch
from torchmetrics.classification import MultilabelAUROC, MulticlassAUROC, BinaryAUROC

class MaskedAUC(nn.Module):
    def __init__(self, mode: str, mask: int, multi_label: bool, num_labels: int = None, vocab_size: int = None):
        super().__init__()
        self.mode = mode
        self.mask = mask
        self.multi_label = multi_label
        self.num_labels = num_labels
        self.vocab_size = vocab_size

        if multi_label:
            self.metric = MultilabelAUROC(num_labels=num_labels, average="macro")
        elif mode == "mlm":
            self.metric = MulticlassAUROC(num_classes=vocab_size, average="macro")
        else:
            self.metric = BinaryAUROC()

    def forward(self, y_pred, y_true):
        if self.mode == "mlm":
            unmasked = y_true[:, :, 1]
            y_true = y_true[:, :, 0]
            y_true = y_true.reshape(-1)
            y_pred = y_pred.reshape(-1, self.vocab_size)
            mask = (y_true != self.mask)
            y_true = y_true[mask]
            y_pred = y_pred[mask]

        elif self.mode == "clf":
            mask = (y_true != self.mask)
            y_pred = y_pred[mask]
            y_true = y_true[mask]

        else:
            y_true = y_true.view(-1)
            y_pred = y_pred.view(-1)

        return self.metric(y_pred, y_true)