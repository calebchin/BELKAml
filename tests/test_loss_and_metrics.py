import pytest
import torch

from losses.categorical import MultiLabelLoss, BinaryLoss
from losses.focal import CategoricalLoss
from metrics.auc import MaskedAUC


@pytest.mark.parametrize("macro", [True, False])
def test_multilabel_loss_basic(macro: bool) -> None:
    """Test MultiLabelLoss"""
    # Two samples, two labels
    y_true = torch.tensor([[1, 0], [0, 1]])
    y_pred_perfect = torch.tensor([[0.999, 0.001], [0.001, 0.999]])
    y_pred_wrong = torch.tensor([[0.001, 0.999], [0.999, 0.001]])

    loss_fn = MultiLabelLoss(epsilon=1e-6, macro=macro)

    # Perfect predictions should give near zero loss
    loss_perfect = loss_fn(y_pred_perfect, y_true)
    assert loss_perfect.item() < 0.01

    # Completely wrong predictions should give higher loss
    loss_wrong = loss_fn(y_pred_wrong, y_true)

    # Check loss of completely wrong predictions is higher than fully accurate predictions
    assert loss_wrong.item() > loss_perfect.item()


def test_binary_loss_basic() -> None:
    """Test Binary Loss"""
    y_true = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    y_pred_perfect = torch.tensor([0.99, 0.01, 0.99, 0.01])
    y_pred_wrong = torch.tensor([0.01, 0.99, 0.01, 0.99])

    loss_fn = BinaryLoss()

    loss_perfect = loss_fn(y_pred_perfect, y_true)
    assert loss_perfect.item() < 0.01

    loss_wrong = loss_fn(y_pred_wrong, y_true)
    assert loss_wrong.item() > loss_perfect.item()


def test_categorical_loss_basic() -> None:
    """Test CategorcialLoss"""
    vocab_size = 5
    mask_token = -1
    y_true = torch.tensor(
        [
            [[1, 2], [mask_token, 0]],
            [[2, 1], [3, 3]],
        ]
    )
    y_pred_perfect = torch.zeros(2, 2, vocab_size)
    y_pred_perfect[0, 0, 1] = 0.999
    y_pred_perfect[1, 0, 2] = 0.999
    y_pred_perfect[1, 1, 3] = 0.999

    y_pred_wrong = torch.ones_like(y_pred_perfect) / vocab_size

    loss_fn = CategoricalLoss(epsilon=1e-6, mask=mask_token, vocab_size=vocab_size)

    loss_perfect = loss_fn(y_pred_perfect, y_true)
    loss_wrong = loss_fn(y_pred_wrong, y_true)

    assert loss_perfect.item() < loss_wrong.item()


@pytest.mark.parametrize(
    "mode, multi_label", [("mlm", False), ("clf", True), ("other", False)]
)
def test_masked_auc_basic(mode: str, multi_label: bool) -> None:
    """Test MaskedAUC"""
    metric_fn: MaskedAUC
    y_pred: torch.Tensor
    y_true: torch.Tensor
    if mode == "mlm":
        vocab_size = 4
        mask_token = -1
        y_true = torch.tensor([[[1, 0], [mask_token, 2]], [[2, 1], [3, 3]]])
        y_pred = torch.rand(2, 2, vocab_size)
        metric_fn = MaskedAUC(
            mode=mode, mask=mask_token, multi_label=multi_label, vocab_size=vocab_size
        )
    elif mode == "clf" and multi_label:
        num_labels = 3
        y_true = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.long)
        y_pred = torch.rand(2, num_labels)
        metric_fn = MaskedAUC(
            mode=mode, mask=-1, multi_label=multi_label, num_labels=num_labels
        )
    else:  # regular binary classification ("other" mode)
        y_true = torch.tensor([0, 1, 1, 0], dtype=torch.long)
        y_pred = torch.rand_like(y_true, dtype=torch.float)
        metric_fn = MaskedAUC(mode=mode, mask=-1, multi_label=multi_label, num_labels=1)

    if mode == "clf" and multi_label:
        auc_score = metric_fn(y_pred, y_true)
    else:
        auc_score = metric_fn(y_pred.float(), y_true)

    assert 0 <= auc_score <= 1
