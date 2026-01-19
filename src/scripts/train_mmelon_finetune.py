"""Training script for finetuning MMELON foundation model on binding prediction.

This script implements a two-stage training pipeline:
1. Extract molecule embeddings from frozen MMELON foundation model
2. Train protein-aware classification head for binding prediction

Usage:
    python src/scripts/train_mmelon_finetune.py --config configs/mmelon_finetune_config.yaml
"""

import argparse
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from core.model.finetune import (
    MMELONFoundationModel,
    ProteinAwareBinaryClassificationHead
)
from core.model.finetune.dataset import (
    MoleculeProteinBindingDataset,
    collate_fn_mmelon
)
from core.losses.categorical import BinaryLoss
from core.metrics.auc import MaskedAUC


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config YAML file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(device_str: str = "auto") -> torch.device:
    """Setup compute device.

    Args:
        device_str: Device string ("auto", "cpu", "cuda", "mps")

    Returns:
        PyTorch device
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    print(f"Using device: {device}")
    return device


def train_epoch(
    foundation_model: MMELONFoundationModel,
    head: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    freeze_foundation: bool = True
) -> float:
    """Train for one epoch.

    Args:
        foundation_model: MMELON foundation model (frozen)
        head: Classification head to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Compute device
        epoch: Current epoch number
        freeze_foundation: Whether to freeze foundation model (default True)

    Returns:
        Average training loss
    """
    head.train()
    if freeze_foundation:
        foundation_model.model.eval()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, (smiles_list, protein_ids, labels) in enumerate(dataloader):
        # Move to device
        protein_ids = protein_ids.to(device)
        labels = labels.to(device)

        # Extract embeddings from foundation model (no grad)
        with torch.no_grad():
            molecule_embeddings = foundation_model.get_embedding(smiles_list)

        # Forward pass through head
        optimizer.zero_grad()
        predictions = head(molecule_embeddings, protein_ids)

        # Compute loss
        loss = criterion(predictions, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(
                f"Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(
    foundation_model: MMELONFoundationModel,
    head: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    metric: Optional[nn.Module],
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Validate model.

    Args:
        foundation_model: MMELON foundation model
        head: Classification head
        dataloader: Validation data loader
        criterion: Loss function
        metric: Evaluation metric (e.g., MaskedAUC)
        device: Compute device
        epoch: Current epoch number

    Returns:
        Dictionary with validation metrics
    """
    foundation_model.model.eval()
    head.eval()

    total_loss = 0.0
    num_batches = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for smiles_list, protein_ids, labels in dataloader:
            # Move to device
            protein_ids = protein_ids.to(device)
            labels = labels.to(device)

            # Extract embeddings
            molecule_embeddings = foundation_model.get_embedding(smiles_list)

            # Forward pass
            predictions = head(molecule_embeddings, protein_ids)

            # Compute loss
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            num_batches += 1

            # Collect for metrics
            all_predictions.append(predictions)
            all_labels.append(labels)

    # Compute metrics
    avg_loss = total_loss / num_batches
    results = {'val_loss': avg_loss}

    if metric is not None:
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Compute AUC
        auc_score = metric(all_predictions, all_labels.long())
        results['val_auc'] = auc_score.item()

        print(
            f"Epoch {epoch} | Val Loss: {avg_loss:.4f} | Val AUC: {auc_score:.4f}"
        )
    else:
        print(f"Epoch {epoch} | Val Loss: {avg_loss:.4f}")

    return results


def save_checkpoint(
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    checkpoint_dir: str,
    prefix: str = "mmelon_head"
):
    """Save model checkpoint.

    Args:
        head: Classification head to save
        optimizer: Optimizer state
        epoch: Current epoch
        val_loss: Validation loss
        checkpoint_dir: Directory to save checkpoint
        prefix: Checkpoint filename prefix
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"{prefix}_epoch{epoch:03d}_valloss{val_loss:.4f}.pt"
    )

    torch.save({
        'epoch': epoch,
        'head_state_dict': head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, checkpoint_path)

    print(f"Saved checkpoint: {checkpoint_path}")


def main(config: Dict[str, Any]):
    """Main training function.

    Args:
        config: Configuration dictionary
    """
    # Set random seed
    torch.manual_seed(config['seed'])

    # Setup device
    device = setup_device(config.get('device', 'auto'))

    # Create datasets
    print("Loading datasets...")
    train_dataset = MoleculeProteinBindingDataset(
        data_path=config['data_path'],
        split='train',
        val_ratio=config.get('val_ratio', 0.1),
        seed=config['seed'],
        protein_id_column=config.get('protein_id_column', 'protein_id'),
        smiles_column=config.get('smiles_column', 'smiles'),
        label_column=config.get('label_column', 'binds')
    )

    val_dataset = MoleculeProteinBindingDataset(
        data_path=config['data_path'],
        split='val',
        val_ratio=config.get('val_ratio', 0.1),
        seed=config['seed'],
        protein_id_column=config.get('protein_id_column', 'protein_id'),
        smiles_column=config.get('smiles_column', 'smiles'),
        label_column=config.get('label_column', 'binds')
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 2),
        collate_fn=collate_fn_mmelon,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        collate_fn=collate_fn_mmelon,
        pin_memory=True
    )

    # Initialize foundation model
    print("Loading MMELON foundation model...")
    foundation_model = MMELONFoundationModel(
        model_path=config.get('foundation_model_path', 'ibm/biomed.sm.mv-te-84m'),
        device=device
    )

    # Determine embedding dimension
    print("Determining embedding dimension...")
    test_smiles = "CCO"
    test_emb = foundation_model.get_embedding(test_smiles)
    embedding_dim = test_emb.shape[-1]
    print(f"MMELON embedding dimension: {embedding_dim}")

    # Initialize classification head
    print("Initializing classification head...")
    head = ProteinAwareBinaryClassificationHead(
        molecule_embedding_dim=embedding_dim,
        num_proteins=config.get('num_proteins', 3),
        protein_embedding_dim=config.get('protein_embedding_dim', 16),
        hidden_dims=config.get('hidden_dims', [128, 64]),
        dropout_rate=config.get('dropout_rate', 0.1),
        activation=config.get('activation', 'relu'),
        use_normalization=config.get('use_normalization', False),
        normalization_type=config.get('normalization_type', 'layer')
    ).to(device)

    # Loss function and metrics
    criterion = BinaryLoss(
        gamma=config.get('focal_gamma', 2.0),
        alpha=config.get('focal_alpha', 0.25)
    )

    metric = MaskedAUC(mode='clf')

    # Optimizer
    optimizer = Adam(
        head.parameters(),
        lr=config['learning_rate'],
        eps=config.get('epsilon', 1e-7)
    )

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.get('lr_factor', 0.5),
        patience=config.get('lr_patience', 5),
        verbose=True
    )

    # Training loop
    print(f"\nStarting training for {config['epochs']} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"{'='*60}")

        # Train
        train_loss = train_epoch(
            foundation_model=foundation_model,
            head=head,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            freeze_foundation=config.get('freeze_foundation', True)
        )
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

        # Validate
        val_results = validate(
            foundation_model=foundation_model,
            head=head,
            dataloader=val_loader,
            criterion=criterion,
            metric=metric,
            device=device,
            epoch=epoch
        )

        val_loss = val_results['val_loss']

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            save_checkpoint(
                head=head,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                checkpoint_dir=config['checkpoint_dir'],
                prefix='mmelon_head_best'
            )
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.get('patience', 10):
            print(f"\nEarly stopping after {epoch} epochs")
            break

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MMELON foundation model with finetuning head"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file'
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Run training
    main(config)
