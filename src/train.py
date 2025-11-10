import torch
import os
from typing import Union
from model import Belka
from losses import CategoricalLoss, BinaryLoss, MultiLabelLoss
from metrics import MaskedAUC
from utils.data_utils import train_val_set
from utils.model_utils import load_model


def train_model(
    model: Union[str, None],
    epochs: int,
    initial_epoch: int,
    mode: str,
    model_name: str,
    patience: int,
    steps_per_epoch: int,
    validation_steps: int,
    working: str,
    **parameters
):
    """
    Train a PyTorch model.
    """

    # Data loaders
    train_loader, val_loader = train_val_set(**parameters, mode=mode, working=working)

    # Model setup
    if model is not None:
        model = load_model(model)  # Load PyTorch model from file
    else:
        model = Belka(**parameters, mode=mode)
        if mode == 'mlm':
            loss_fn = CategoricalLoss(mask=-1, **parameters)
            metrics = MaskedAUC(mask=-1, multi_label=False, num_labels=None, **parameters, mode=mode)
        elif mode == 'fps':
            loss_fn = BinaryLoss(**parameters)
            metrics = MaskedAUC(mask=-1, multi_label=False, num_labels=None, **parameters, mode=mode)
        else:  # clf mode - binary classification
            loss_fn = BinaryLoss(**parameters)  # Use BinaryLoss for single binary label
            metrics = MaskedAUC(mask=-1, multi_label=False, num_labels=None, **parameters, mode=mode)

    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.get("lr", 1e-3))
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, mode="min")

    # Checkpointing
    def save_checkpoint(epoch, val_loss):
        suffix = {
            'mlm': f"_{epoch:03d}_{val_loss:.4f}.pt",
            'fps': f"_{epoch:03d}_{val_loss:.4f}.pt",
            'clf': f"_{epoch:03d}_{val_loss:.4f}.pt"
        }
        filepath = os.path.join(working, model_name + suffix[mode])
        torch.save(model.state_dict(), filepath)

    # Early stopping
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # Print summary and sample batch
    model.eval()
    batch = next(iter(train_loader))
    # Extract tokenized SMILES as input
    sample_input = batch['smiles']
    with torch.no_grad():
        y_pred = model(sample_input)
    print(model)

    # Training loop
    for epoch in range(initial_epoch, epochs):
        model.train()
        train_loss = 0
        for step, batch in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            # Extract inputs and targets from batch dictionary
            x = batch['smiles']  # Use tokenized SMILES as input
            y = batch['binds']

            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= steps_per_epoch

        # Validation
        model.eval()
        val_loss = 0
        for step, batch in enumerate(val_loader):
            if mode == 'mlm':
                val_steps = None
            else:
                val_steps = validation_steps
            if val_steps and step >= val_steps:
                break
            # Extract inputs and targets from batch dictionary
            x = batch['smiles']  # Use tokenized SMILES
            y = batch['binds']

            with torch.no_grad():
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                val_loss += loss.item()
        val_loss /= (validation_steps if validation_steps else len(val_loader))

        # Metrics (optional)
        # val_auc = metrics(y_pred, y) # Implement your own

        lr_scheduler.step(val_loss)
        print(f"Epoch {epoch} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

        # Save checkpoint
        save_checkpoint(epoch, val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Optionally, save best weights separately
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return model
