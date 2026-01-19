"""Dataset classes for foundation model finetuning.

This module provides dataset classes that load molecular binding data
and prepare it for finetuning foundation models like MMELON.
"""

from typing import Dict, Optional, Tuple, Union
import pandas as pd
import torch
from torch.utils.data import Dataset


class MoleculeProteinBindingDataset(Dataset):
    """Dataset for molecule-protein binding prediction.

    This dataset loads data with SMILES strings, protein targets, and binding labels.
    It's designed for finetuning foundation models with protein-aware classification heads.

    The dataset expects a parquet or CSV file with columns:
    - smiles: SMILES string representation of molecule
    - protein_id: Integer protein ID (0, 1, 2, ... for BRD4, HSA, sEH, etc.)
    - binds: Binary binding label (0 or 1)
    - Optional: protein_name for human-readable protein names

    Example:
        >>> dataset = MoleculeProteinBindingDataset(
        ...     data_path="data/train.parquet",
        ...     split="train"
        ... )
        >>> sample = dataset[0]
        >>> print(sample.keys())
        dict_keys(['smiles', 'protein_id', 'binds'])
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
        protein_id_column: str = "protein_id",
        smiles_column: str = "smiles",
        label_column: str = "binds"
    ):
        """Initialize molecule-protein binding dataset.

        Args:
            data_path: Path to parquet or CSV file
            split: Dataset split ("train", "val", or "all")
            val_ratio: Validation split ratio (default 0.1)
            seed: Random seed for train/val splitting
            protein_id_column: Name of protein ID column
            smiles_column: Name of SMILES column
            label_column: Name of binding label column

        Raises:
            FileNotFoundError: If data_path doesn't exist
            ValueError: If required columns are missing
        """
        super().__init__()

        self.data_path = data_path
        self.split = split
        self.val_ratio = val_ratio
        self.seed = seed
        self.protein_id_column = protein_id_column
        self.smiles_column = smiles_column
        self.label_column = label_column

        # Load data
        self._load_data()

    def _load_data(self):
        """Load and split data from file."""
        # Load from file
        if self.data_path.endswith('.parquet'):
            df = pd.read_parquet(self.data_path)
        elif self.data_path.endswith('.csv'):
            df = pd.read_csv(self.data_path)
        else:
            raise ValueError(
                f"Unsupported file format: {self.data_path}. "
                "Use .parquet or .csv"
            )

        # Validate required columns
        required_cols = [self.smiles_column, self.protein_id_column, self.label_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

        # Train/val split
        if self.split in ["train", "val"]:
            # Stratified split by protein_id to ensure balanced protein distribution
            train_dfs = []
            val_dfs = []

            for protein_id in df[self.protein_id_column].unique():
                protein_df = df[df[self.protein_id_column] == protein_id]

                # Shuffle and split
                protein_df = protein_df.sample(frac=1, random_state=self.seed)
                n_val = int(len(protein_df) * self.val_ratio)

                val_dfs.append(protein_df.iloc[:n_val])
                train_dfs.append(protein_df.iloc[n_val:])

            train_df = pd.concat(train_dfs, ignore_index=True)
            val_df = pd.concat(val_dfs, ignore_index=True)

            if self.split == "train":
                self.df = train_df
            else:  # val
                self.df = val_df

        else:  # all
            self.df = df

        # Reset index
        self.df = self.df.reset_index(drop=True)

        print(f"Loaded {len(self.df)} samples for split='{self.split}'")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, int, float]]:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with keys:
            - smiles: SMILES string
            - protein_id: Integer protein ID
            - binds: Binary binding label (0 or 1)
        """
        row = self.df.iloc[idx]

        return {
            'smiles': row[self.smiles_column],
            'protein_id': int(row[self.protein_id_column]),
            'binds': float(row[self.label_column])
        }


def collate_fn_mmelon(
    batch: list,
) -> Tuple[list, torch.Tensor, torch.Tensor]:
    """Collate function for MMELON finetuning batches.

    This collate function prepares batches for the two-step workflow:
    1. Extract embeddings from foundation model (requires raw SMILES)
    2. Pass embeddings + protein IDs to classification head

    Args:
        batch: List of sample dicts from dataset

    Returns:
        Tuple of (smiles_list, protein_ids, labels):
        - smiles_list: List of SMILES strings (for foundation model)
        - protein_ids: Tensor of protein IDs (batch,)
        - labels: Tensor of binding labels (batch,)
    """
    smiles_list = [sample['smiles'] for sample in batch]
    protein_ids = torch.tensor(
        [sample['protein_id'] for sample in batch],
        dtype=torch.long
    )
    labels = torch.tensor(
        [sample['binds'] for sample in batch],
        dtype=torch.float32
    ).unsqueeze(-1)  # (batch, 1)

    return smiles_list, protein_ids, labels
