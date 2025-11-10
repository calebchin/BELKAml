from torch.utils.data import DataLoader

from .torch_data_utils import BelkaDataset
import torch
import numpy as np
import os
def train_val_set(batch_size: int, buffer_size: int, masking_rate: float, max_length: int, mode: str, seed: int,
                  vocab_size: int, working: str, num_workers: int = 4, val_split: float = 0.1, **kwargs) -> tuple:
    """Make train and validation DataLoaders.

    Args:
        batch_size: Batch size
        buffer_size: Not used (kept for API compatibility)
        masking_rate: Not used here
        max_length: Not used here
        mode: 'mlm', 'fps', or 'clf'
        seed: Random seed
        vocab_size: Not used here
        working: Working directory with belka.parquet
        num_workers: Number of workers for DataLoader
        val_split: Validation split fraction (default 0.1)
        **kwargs: Additional arguments

    Returns:
        (train_loader, val_loader)

    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # Parquet path
    parquet_path = os.path.join(working, "belka.parquet")

    # Determine subsets based on mode
    if mode == 'mlm':
        train_subset = 'train'
        val_subset = None
    else:  # fps or clf
        train_subset = 'train'
        val_subset = 'val'

    # Construct vocab path
    vocab_path = os.path.join(working, "vocab.txt") if os.path.exists(os.path.join(working, "vocab.txt")) else None

    # Create training DataLoader
    train_dataset = BelkaDataset(
        parquet_path=parquet_path,
        subset=train_subset,
        val_split=val_split,
        seed=seed,
        vocab_path=vocab_path,
        max_length=max_length
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        generator=g
    )

    # Create validation DataLoader if needed
    if val_subset is not None:
        val_dataset = BelkaDataset(
            parquet_path=parquet_path,
            subset=val_subset,
            val_split=val_split,
            seed=seed,
            vocab_path=vocab_path,
            max_length=max_length
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
    else:
        val_loader = None

    return train_loader, val_loader
